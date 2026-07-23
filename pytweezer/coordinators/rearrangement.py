"""Atom-rearrangement coordinator.

Ports the standalone ``rearrangement_node_server.py`` (a ZMQ REP server that held
its own camera and talked to the SLM over another socket) into a composite-device
:class:`~pytweezer.coordinators.base.Coordinator`. The camera and SLM are now
sub-devices of the same process, so this coordinator drives them with direct
in-process calls: the GPU-computed phase sequence goes straight to
``slm.update_mask`` with no frame ever serialized onto a socket.

Roles in the composite's ``"devices"`` block:

* ``slm``    — a :class:`pytweezer.drivers.slm.SLM` (`update_mask`). Required.
* ``camera`` — an ImagEM-X2-style camera (`setup_acquisition`, `set_roi`,
  `enable_em_gain`, `acquire_n_frames`, ...). Optional: without it the
  coordinator constructs SLM-only and :meth:`initialise` raises, but the phase
  sequence generation and upload path is fully usable. Benchmarks that only
  time frame delivery run this way.

Lifecycle over RPC (all synchronous; each stalls the server for its duration, per
the coordinator contract):

* :meth:`initialise` — build the GPU phasemask generator, configure the camera,
  precompute the initial array. Takes the two trap parameter sets (``data1``/
  ``data2``, shape ``(4, N)``: w, phi, x, y).
* :meth:`arm_rearrangement` — grab an image, extract the occupancy mask, then
  generate the interpolated phase sequence
  (``OptimisationBasedPhasemaskGeneratorGPU.iter_rearrangement_sequence``) and
  upload it to the SLM concurrently, then grab a reset image. Returns the
  before/after images.
* :meth:`test` / :meth:`status` / :meth:`shutdown`.

Generation and upload are pipelined: the GPU loop pushes each finished frame onto a
bounded queue and a writer thread copies it to the host and DMAs it to the board, so
synthesis of frame *n+1* overlaps the upload of frame *n*. The queue depth
(:data:`UPLOAD_QUEUE_DEPTH`) bounds how far the GPU may run ahead.

``cupy``/``lap`` and the heavy GPU math are imported lazily, so this module imports
and :meth:`status` works on any machine; :meth:`initialise`/:meth:`arm_rearrangement`
raise a clear error where the GPU stack is absent.

The remaining timing optimisation — preloading frames into the SLM's on-board memory
and clocking them out with a hardware trigger (``preload_sequence`` /
``start_auto_increment``) instead of per-frame software writes — is described in
``docs/rearrangement_coordinator.md``.
"""

import queue
import threading
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

#: Set ``False`` to force the numpy occupancy path instead of the C++ extension.
USE_SUM_CPP = True

_sum_cpp_module = None
_sum_cpp_loaded = False

#: How many generated frames may queue ahead of the SLM before the GPU loop blocks.
UPLOAD_QUEUE_DEPTH = 5


def _sum_cpp():
    global _sum_cpp_module, _sum_cpp_loaded
    if not USE_SUM_CPP:
        return None
    if not _sum_cpp_loaded:
        _sum_cpp_loaded = True
        try:
            from pytweezer.cpp import sum_pixel_values_cpp

            _sum_cpp_module = sum_pixel_values_cpp
        except Exception:  # pragma: no cover 
            LOGGER.debug(
                "sum_pixel_values_cpp unavailable; using numpy.",
                exc_info=True,
            )
            _sum_cpp_module = None
    return _sum_cpp_module


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
        self.camera: ImagEMX2Camera | None = self.targets.get(self.camera_role)
        self.slm: SLM = self.require_role(self.slm_role)

        self.phasemask_kwargs = {**DEFAULT_PHASEMASK, **(conf.get("phasemask") or {})}

        self._initialised = False
        self._state = None  # populated by initialise()

        #: ``time.perf_counter()`` at which the most recent
        #: :meth:`_play_sequence_pipelined` finished displaying its *first* frame,
        #: or ``None`` if it never did. Frame 0 must be synthesised, copied to the
        #: host and DMA'd before anything reaches the panel, so this splits a
        #: pipelined run into time-to-first-frame and the move proper.
        self.last_first_frame_at = None

    def _require_gpu(self):
        if not _HAS_CUPY:
            raise RuntimeError(
                "Rearrangement needs cupy/lap and a CUDA GPU, which are not available "
                "on this machine. Run the rig on the GPU host, or use status()/test() "
                "only here."
            )

    def _require_camera(self):
        if self.camera is None:
            raise RuntimeError(
                f"{type(self).__name__} needs a sub-device with role "
                f"{self.camera_role!r} for this call; this composite provides roles: "
                f"{sorted(self.targets) or '(none)'}. Set \"role\": "
                f"{self.camera_role!r} on the relevant entry in the composite's "
                '"devices" block.'
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
        profile: str = "minimum_jerk",
    ) -> None:
        """Build the phasemask generator, configure the camera, precompute masks.

        ``data1``/``data2`` are ``(4, N)`` arrays of trap parameters (w, phi, x, y)
        for the initial and target arrays. Everything GPU-side is kept on the device
        between here and :meth:`arm_rearrangement`.

        ``profile`` picks the transport trajectory used by every subsequent
        :meth:`arm_rearrangement`: ``"minimum_jerk"`` (smoother on the atoms) or
        ``"linear"``, which needs 1.875x fewer frames for the same ``d0`` and so
        completes the move in proportionally less time.
        """
        self._require_gpu()
        self._require_camera()
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
            profile=profile,
        )
        self._initialised = True
        LOGGER.info("Rearrangement node initialised.")

    # ------------------------------------------------------------------ #
    # Arm / run one rearrangement
    # ------------------------------------------------------------------ #

    def _extract_occupancy(self, image, array_shape, threshold):
        """Threshold per-site pixel sums into a flat boolean occupancy mask.

        Uses the compiled ``sum_pixel_values`` when it is available and the image is
        ``uint16`` (the dtype the extension is built for), else the numpy version.
        """
        from pytweezer.analysis import analysis as an

        img = an.morphological_tophat_high_pass(image, feature_size=10)
        grid_positions = self._state["grid_positions"]

        cpp = _sum_cpp()
        if cpp is not None and getattr(img, "dtype", None) == np.uint16:
            pixel_sums = cpp.sum_pixel_values(
                img, grid_positions, array_shape, window_size=3
            )
        else:
            pixel_sums = an.sum_pixel_values(
                img, grid_positions, array_shape, window_size=3
            )
        return np.fliplr(pixel_sums).flatten() > threshold

    def _play_sequence_pipelined(self, frames):
        """Upload each frame to the SLM as it is produced; return the frame count.

        ``frames`` is an iterator of ``cupy`` (or numpy) uint8 masks. A writer thread
        drains a bounded queue, so the GPU keeps synthesising while the previous frame
        is copied to the host and DMA'd to the board. The GPU->host ``.get()`` runs on
        the writer thread to keep the PCIe transfer off the generation loop.

        The queue depth bounds how far generation may run ahead. On an upload error
        the writer keeps draining (without touching hardware) so the producer cannot
        deadlock on a full queue; the first error is re-raised here.

        :attr:`last_first_frame_at` is set to the moment the first frame reached the
        panel, so a caller can separate the latency before the move starts from the
        move itself.
        """
        upload_queue = queue.Queue(maxsize=UPLOAD_QUEUE_DEPTH)
        errors = []
        first_frame_at = []
        self.last_first_frame_at = None

        def writer():
            while True:
                frame = upload_queue.get()
                if frame is None:
                    return
                if errors:
                    continue  # drain the rest so the producer never blocks
                try:
                    # cupy arrays expose .get(); numpy frames pass straight through.
                    host_frame = frame.get() if hasattr(frame, "get") else frame
                    self.slm.update_mask(host_frame)
                    if not first_frame_at:
                        first_frame_at.append(time.perf_counter())
                except Exception as exc:
                    errors.append(exc)

        thread = threading.Thread(target=writer, name="slm-upload", daemon=True)
        thread.start()

        n_frames = 0
        try:
            for frame in frames:
                upload_queue.put(frame)
                n_frames += 1
        finally:
            upload_queue.put(None)
            thread.join()

        if first_frame_at:
            self.last_first_frame_at = first_frame_at[0]
        if errors:
            raise errors[0]
        return n_frames

    def arm_rearrangement(self):
        """Run one rearrangement and return the ``(before, after)`` camera images.

        Loads the initial array, grabs an occupancy image, then generates the
        interpolated phase sequence and uploads it to the SLM *concurrently* - each
        frame goes to the board as soon as the GPU produces it - and finally grabs a
        reset image. Timing breakdown is logged.

        Generation and upload overlap, so they are timed together; splitting them
        would only measure where the pipeline happened to stall.
        """
        self._require_gpu()
        self._require_initialised()

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
        occ_mask = self._extract_occupancy(img_array0, arr_shape1, s["threshold"])
        t2 = time.time()

        # 3. Pairing, interpolation and SLM upload, pipelined.
        frames = PM.iter_rearrangement_sequence(
            s["terms1"], s["terms2"], occ_mask,
            d0=s["d0"],
            profile=s.get("profile", "minimum_jerk"),
            to_host=False,
        )
        n_frames = self._play_sequence_pipelined(frames)
        t3 = time.time()

        # 4. Reset image.
        try:
            self.camera.start_acquisition()
            img_array1 = self.camera.acquire_n_frames(1)[0]
        except Exception:
            LOGGER.exception("Reset-image acquisition failed; returning zeros.")
            img_array1 = np.zeros_like(img_array0)
        t4 = time.time()

        LOGGER.info(
            "Rearrangement complete: %d frames, %.4fs total "
            "(occupancy %.4fs, sequence+upload %.4fs, reset %.4fs).",
            n_frames, t4 - t1, t2 - t1, t3 - t2, t4 - t3,
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
        if self._initialised and self.camera is not None:
            try:
                self.camera.stop_acquisition()
            except Exception:
                LOGGER.debug("camera.stop_acquisition() during shutdown failed", exc_info=True)
        self._state = None
        self._initialised = False
