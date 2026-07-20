"""Atom-rearrangement coordinator.

Ports the standalone ``rearrangement_node_server.py`` (a ZMQ REP server that held
its own camera and talked to the SLM over another socket) into a composite-device
:class:`~pytweezer.coordinators.base.Coordinator`. The camera and SLM are now
sub-devices of the same process, so this coordinator drives them with direct
in-process calls: the GPU-computed phase sequence goes straight to
``slm.update_mask`` with no frame ever serialized onto a socket.

Two roles are required in the composite's ``"devices"`` block:

* ``camera`` — an ImagEM-X2-style camera (`setup_acquisition`, `set_roi`,
  `enable_em_gain`, `acquire_n_frames`, ...).
* ``slm``    — a :class:`pytweezer.drivers.slm.SLM` (`update_mask`).

Lifecycle over RPC (all synchronous; each stalls the server for its duration, per
the coordinator contract):

* :meth:`initialise` — build the GPU phasemask generator, configure the camera,
  precompute the initial array. Takes the two trap parameter sets (``data1``/
  ``data2``, shape ``(4, N)``: w, phi, x, y).
* :meth:`arm_rearrangement` — grab an image, extract the occupancy mask, generate
  the interpolated phase sequence
  (``OptimisationBasedPhasemaskGeneratorGPU.generate_rearrangement_sequence``), play
  it to the SLM, then grab a reset image. Returns the before/after images.
* :meth:`test` / :meth:`status` / :meth:`shutdown`.

``cupy``/``lap`` and the heavy GPU math are imported lazily, so this module imports
and :meth:`status` works on any machine; :meth:`initialise`/:meth:`arm_rearrangement`
raise a clear error where the GPU stack is absent.

The remaining timing optimisation — preloading the whole sequence into the SLM's
on-board memory and clocking it out with a hardware trigger instead of the
software-timed :meth:`SLM.run_sequence` — is described in
``docs/rearrangement_coordinator.md``.
"""

import time

import numpy as np

from pytweezer.coordinators.base import Coordinator
from pytweezer.logging_utils import get_logger

from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.drivers.slm import SLM

LOGGER = get_logger("rearrangement")

try:
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover - depends on the machine's GPU stack
    cp = None
    _HAS_CUPY = False


#: Default phasemask-generator geometry (the lab's Rb SLM); overridable via config.
DEFAULT_PHASEMASK = dict(
    wavelength_um=0.852,
    focal_length_mm=17.3,
    slm_pitch_um=17,
    slm_res=(1024, 1024),
    input_beam_waist_mm=16,
    fresnel_f_mm=1072,
    blaze_dx_dy_um=(48, -4),
    zernike_coeff_dict={
        5: 1.195, 6: 0.725, 7: 0.970, 8: 0.478, 9: -1.091,
        10: 0.303, 11: 0.021, 12: 0.072, 13: 0.049,
    },
)

#: Default camera ROI (x0, y0, width, height) if ``initialise`` isn't given one.
DEFAULT_ROI = [50, 70, 384, 384]


class Rearrangement(Coordinator):
    """Camera + SLM rearrangement loop, run entirely in one process."""

    camera_role = "camera"
    slm_role = "slm"

    def __init__(self, targets, conf):
        super().__init__(targets, conf)
        self.camera: ImagEMX2Camera = self.require_role(self.camera_role)
        self.slm: SLM = self.require_role(self.slm_role)

        self.phasemask_kwargs = {**DEFAULT_PHASEMASK, **(conf.get("phasemask") or {})}

        self._initialised = False
        self._state = None  # populated by initialise()

    def _require_gpu(self):
        if not _HAS_CUPY:
            raise RuntimeError(
                "Rearrangement needs cupy/lap and a CUDA GPU, which are not available "
                "on this machine. Run the rig on the GPU host, or use status()/test() "
                "only here."
            )

    def _require_initialised(self):
        if not self._initialised:
            raise RuntimeError("Rearrangement not initialised; call initialise() first.")

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def status(self) -> dict:
        """Report readiness. Works with or without the GPU stack."""
        return {
            "gpu_available": _HAS_CUPY,
            "initialised": self._initialised,
            "roles": sorted(self.targets),
            "camera_connected": bool(getattr(self.camera, "is_connected", lambda: None)()),
            "slm_connected": bool(getattr(self.slm, "is_connected", lambda: None)()),
        }

    def get_slm_temperature(self) -> float:
        return float(self.slm.get_temperature())

    # ------------------------------------------------------------------ #
    # Initialise
    # ------------------------------------------------------------------ #

    def initialise(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        array_shape1,
        array_shape2,
        d0: float,
        fps: float,
        threshold: float,
        grid_positions,
        roi=None,
    ) -> None:
        """Build the phasemask generator, configure the camera, precompute masks.

        ``data1``/``data2`` are ``(4, N)`` arrays of trap parameters (w, phi, x, y)
        for the initial and target arrays. Everything GPU-side is kept on the device
        between here and :meth:`arm_rearrangement`.
        """
        self._require_gpu()
        from pytweezer import phasemask as pm

        roi = list(roi) if roi is not None else list(DEFAULT_ROI)

        PM = pm.OptimisationBasedPhasemaskGeneratorGPU(**self.phasemask_kwargs)
        LOGGER.info("Phasemask generator initialised.")

        # Camera setup (ImagEM-X2 specific; simulated backend stubs these).
        x0, y0, width, height = roi
        self.camera.setup_acquisition("snap", 1)
        self.camera.set_trigger_source("ext")
        self.camera.set_external_exposure_mode()
        self.camera.enable_em_gain(True)
        self.camera.enable_direct_em_gain(True)
        self.camera.set_sensitivity(1200)
        self.camera.set_roi(x0, width, y0, height)
        LOGGER.info("Camera configured for rearrangement (roi=%s).", roi)

        w1, phi1, x1, y1 = cp.asarray(np.asarray(data1))
        w2, phi2, x2, y2 = cp.asarray(np.asarray(data2))
        terms1 = (w1, phi1, x1, y1, array_shape1)
        terms2 = (w2, phi2, x2, y2, array_shape2)

        # The initial array to load onto the SLM before each rearrangement.
        pm_array_init = PM.generate_phasemask(list(terms1))
        pm_init = PM.superimpose([pm_array_init, PM.fresnel, PM.blaze, PM.zernike])
        pm_init_uint8 = PM.transform_phase_8bit(pm_init).get()

        self._state = dict(
            PM=PM,
            terms1=terms1, terms2=terms2,
            pm_init_uint8=pm_init_uint8,
            d0=d0, fps=fps, threshold=threshold, grid_positions=grid_positions, roi=roi,
        )
        self._initialised = True
        LOGGER.info("Rearrangement node initialised.")

    # ------------------------------------------------------------------ #
    # Arm / run one rearrangement
    # ------------------------------------------------------------------ #

    def arm_rearrangement(self):
        """Run one rearrangement and return the ``(before, after)`` camera images.

        Loads the initial array, grabs an occupancy image, generates the full
        interpolated phase sequence on the GPU, plays it to the SLM, and grabs a
        reset image. Timing breakdown is logged.
        """
        self._require_gpu()
        self._require_initialised()
        from pytweezer.analysis import analysis as an

        s = self._state
        PM = s["PM"]
        arr_shape1 = s["terms1"][4]

        # Load the initial array onto the SLM.
        self.slm.update_mask(s["pm_init_uint8"])

        # 1. Acquire the occupancy image.
        self.camera.start_acquisition()
        img_array0 = self.camera.acquire_n_frames(1)[0]
        t1 = time.time()

        # 2. Occupancy mask.
        img = an.morphological_tophat_high_pass(img_array0, feature_size=10)
        pixel_sums = np.fliplr(
            an.sum_pixel_values(img, s["grid_positions"], arr_shape1, window_size=3)
        )
        occ_mask = pixel_sums.flatten() > s["threshold"]
        t2 = time.time()

        # 3. Optimal pairing + full interpolated sequence (host-side uint8 frames).
        sequence, _debug = PM.generate_rearrangement_sequence(
            s["terms1"], s["terms2"], occ_mask, d0=s["d0"]
        )
        t3 = time.time()

        # 4. Play the sequence to the SLM.
        self.slm.run_sequence(sequence, fps=s["fps"])
        t4 = time.time()

        # 5. Reset image.
        try:
            self.camera.start_acquisition()
            img_array1 = self.camera.acquire_n_frames(1)[0]
        except Exception:
            LOGGER.exception("Reset-image acquisition failed; returning zeros.")
            img_array1 = np.zeros_like(img_array0)
        t5 = time.time()

        LOGGER.info(
            "Rearrangement complete: %d frames, %.4fs total "
            "(occupancy %.4fs, sequence %.4fs, slm %.4fs, reset %.4fs).",
            len(sequence), t5 - t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
        )
        return np.asarray(img_array0), np.asarray(img_array1)

    # ------------------------------------------------------------------ #
    # Test / teardown
    # ------------------------------------------------------------------ #

    def test(self, delay_s: float = 0.0):
        """Round-trip smoke test: return two random ``(before, after)`` images.

        No GPU needed — exercises the RPC path and image marshalling only.
        """
        if delay_s:
            time.sleep(delay_s)
        rng = np.random.default_rng()
        return (
            rng.integers(0, 256, (10, 10), dtype=np.uint8),
            rng.integers(0, 256, (10, 10), dtype=np.uint8),
        )

    def shutdown(self) -> None:
        """Release rearrangement state. The camera/SLM backends close themselves."""
        if self._initialised:
            try:
                self.camera.stop_acquisition()
            except Exception:
                LOGGER.debug("camera.stop_acquisition() during shutdown failed", exc_info=True)
        self._state = None
        self._initialised = False
