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

from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.drivers.slm import SLM

try:
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover - depends on the machine's GPU stack
    cp = None
    _HAS_CUPY = False


USE_SUM_CPP = False

_sum_cpp_module = None
_sum_cpp_loaded = False

_morph_cpp_module = None
_morph_cpp_loaded = False

#: How many generated frames may queue ahead of the SLM before the GPU loop blocks.
UPLOAD_QUEUE_DEPTH = 5

#: Minimum interval between successive ``update_mask`` calls on the display path,
#: in seconds. ``Write_image``/``ImageWriteComplete`` acknowledge the transfer into
#: the board's memory, NOT a panel flip, so without this the pipeline happily
#: writes faster than the liquid crystal can refresh and each frame overwrites the
#: one still being displayed — the atoms then see fewer, larger jumps than ``d0``
#: asks for. Override per instance with ``conf["panel_period_s"]``; set to 0.0 to
#: disable pacing entirely. Only the display path needs this: preload+trigger
#: writes to frame memory and is clocked by the Arduino.
PANEL_PERIOD_S = 0.70e-3

#: Slack left for the spin tail in :func:`_wait_until`. Must exceed the Windows
#: timer quantum (~0.5 ms), or ``time.sleep`` rounds straight past the deadline,
#: but stay below :data:`PANEL_PERIOD_S`, or nothing ever sleeps. At a 0.70 ms
#: period a 0.10 ms margin overshoots to 1.00 ms; 0.60 ms lands on 0.7006 ms.
SLEEP_MARGIN_S = 0.60e-3


def _wait_until(deadline):
    """Block until ``deadline`` (a ``time.perf_counter()`` value).

    Sleeps the bulk and spins only the last :data:`SLEEP_MARGIN_S`. ``time.sleep``
    releases the GIL, so the generation thread keeps running for most of the wait,
    while the spin covers the tail where the OS timer is too coarse to land on.
    Measured at a 0.70 ms period: mean 0.7006 ms, 3.4 us sd, asleep ~74% of the wait.
    """
    remaining = deadline - time.perf_counter()
    if remaining > SLEEP_MARGIN_S:
        time.sleep(remaining - SLEEP_MARGIN_S)
    while time.perf_counter() < deadline:
        pass


def _pinned_like(frame):
    """Page-locked host array matching for a frame to be copied from the GPU.

    Falls back to ordinary memory if the pinned allocation fails, since pinning is
    a throughput optimisation and not for correctness - should be faster than pageable.
    """
    shape = tuple(frame.shape)
    dtype = np.dtype(frame.dtype)
    count = int(np.prod(shape))
    try:
        mem = cp.cuda.alloc_pinned_memory(count * dtype.itemsize)
        return np.frombuffer(mem, dtype, count).reshape(shape)
    except Exception:
        print("Pinned allocation failed; using pageable memory.")
        return np.empty(shape, dtype)


def _morph_cpp():
    global _morph_cpp_module, _morph_cpp_loaded
    if not USE_SUM_CPP:
        return None
    if not _morph_cpp_loaded:
        _morph_cpp_loaded = True
        try:
            from pytweezer.cpp import morph_tophat_cpp

            _morph_cpp_module = morph_tophat_cpp
        except Exception:  # pragma: no cover
            # pragme no cover stops coverage.py complaining about failing imports here on machine 
            # without the C++ extension built. The extension is optional, so this is not a test failure.
            print("morph_tophat_cpp unavailable; using numpy.")
            _morph_cpp_module = None
    return _morph_cpp_module

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
            print("sum_pixel_values_cpp unavailable; using numpy.")
            _sum_cpp_module = None
    return _sum_cpp_module


dx, dy = 8.0, 12.0

#: Default phasemask-generator geometry (the lab's Rb SLM); overridable via config.
DEFAULT_PHASEMASK = dict(
    wavelength_um=0.852,
    focal_length_mm=17.3,
    slm_pitch_um=17,
    slm_res=(1024,1024),
    input_beam_waist_mm=16,
    fresnel_f_mm=1072,
    blaze_dx_dy_um=(40+dx, -8+dy),
    zernike_coeff_dict={5:1.195, 6:0.725, 7:0.970, 8:0.478, 9:-1.091, 10:0.303, 11:0.021, 12:0.072, 13:0.049}
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

        #: Enforced minimum interval between panel writes; see :data:`PANEL_PERIOD_S`.
        self.panel_period_s = float(conf.get("panel_period_s", PANEL_PERIOD_S))

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
        
        self._require_gpu()
        self._require_camera()
        from pytweezer import phasemask as pm

        roi = list(roi) if roi is not None else list(DEFAULT_ROI)

        PM = pm.OptimisationBasedPhasemaskGeneratorGPU(**self.phasemask_kwargs)
        print("Phasemask generator initialised.")

        # Camera setup (ImagEM-X2 specific; simulated backend stubs these).
        x0, y0, width, height = roi
        self.camera.setup_acquisition("snap", 1)
        self.camera.set_trigger_source("ext")
        self.camera.set_external_exposure_mode()
        self.camera.enable_em_gain(True)
        self.camera.enable_direct_em_gain(True)
        self.camera.set_sensitivity(600)
        self.camera.set_roi(x0, width, y0, height)
        self.camera.timeout = 5
        print("Camera configured for rearrangement (roi=%s).", roi)

        # cp.asarray takes host or device arrays: a no-op for the cupy terms that
        # come straight out of superposition_optimization, a host->device copy for
        # numpy ones. np.asarray cannot sit in front of it - it is not dispatchable
        # under NEP-18, so a cupy input falls through to __array__ and raises.
        w1, phi1, x1, y1 = cp.asarray(data1)
        w2, phi2, x2, y2 = cp.asarray(data2)
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
        print("Rearrangement node initialised.")

    # ------------------------------------------------------------------ #
    # Arm / run one rearrangement
    # ------------------------------------------------------------------ #

    def _extract_occupancy(self, image, array_shape, threshold):
        """Threshold per-site pixel sums into a flat boolean occupancy mask.

        Uses the compiled ``sum_pixel_values`` when it is available and the image is
        ``uint16`` (the dtype the extension is built for), else the numpy version.
        """
        from pytweezer import analysis_old as an

        grid_positions = self._state["grid_positions"]

        cpp_morph = _morph_cpp()
        if cpp_morph is not None and getattr(image, "dtype", None) == np.uint16:
            print(image.dtype)
            img = cpp_morph.tophat(image, feature_size=10)
            print("Using C++ morphological top-hat for occupancy extraction.")
        else:
            img = an.morphological_tophat_high_pass(image, feature_size=10)

        cpp_sum = _sum_cpp()
        if cpp_sum is not None and getattr(img, "dtype", None) == np.uint16:
            pixel_sums = cpp_sum.sum_pixel_values(
                img, grid_positions, array_shape, window_size=3
            )
            print("Using C++ pixel-sum for occupancy extraction.")
        else:
            pixel_sums = an.sum_pixel_values(
                img, grid_positions, array_shape, window_size=3
            )
        return np.fliplr(pixel_sums).flatten() > threshold

    def _play_sequence_pipelined(self, frames):
        """Display each frame on the SLM as it is produced."""
        first_frame_at = []
        self.last_first_frame_at = None
        next_slot = None

        def display(index, host_frame):
            nonlocal next_slot
            self.slm.update_mask(host_frame)
            if not first_frame_at:
                first_frame_at.append(time.perf_counter())
            if self.panel_period_s:
                now = time.perf_counter()
                if next_slot is not None and now < next_slot:
                    _wait_until(next_slot)
                    now = next_slot
                next_slot = now + self.panel_period_s

        n_frames = self._drain_pipelined(frames, display, "slm-upload")
        if first_frame_at:
            self.last_first_frame_at = first_frame_at[0]
        return n_frames

    def _preload_sequence_pipelined(self, frames):
        """Preload each frame into the SLM's on-board memory as it is produced."""
        first_frame_at = []
        self.last_first_frame_at = None

        def store(index, host_frame):
            self.slm.preload_image(host_frame, index)
            if not first_frame_at:
                first_frame_at.append(time.perf_counter())

        n_frames = self._drain_pipelined(frames, store, "slm-preload")
        if first_frame_at:
            self.last_first_frame_at = first_frame_at[0]
        return n_frames

    def _drain_pipelined(self, frames, sink, thread_name):
        """Drain a generator of GPU frames into a sink function on a writer thread."""
        upload_queue = queue.Queue(maxsize=UPLOAD_QUEUE_DEPTH)
        errors = []

        def writer():
            index = 0
            staging = None
            while True:
                frame = upload_queue.get()
                if frame is None:
                    return
                if errors:
                    continue  # drain the rest so the producer never blocks
                try:
                    if hasattr(frame, "get"):  # cupy: copy into pinned host memory
                        if staging is None:
                            staging = _pinned_like(frame)
                        frame.get(out=staging)
                        host_frame = staging
                    else:  # already numpy, nothing to copy
                        host_frame = frame
                    sink(index, host_frame)
                    index += 1
                except Exception as exc:
                    errors.append(exc)

        thread = threading.Thread(target=writer, name=thread_name, daemon=True)
        thread.start()

        n_frames = 0
        try:
            for frame in frames:
                upload_queue.put(frame)
                n_frames += 1
        finally:
            upload_queue.put(None)
            thread.join()

        if errors:
            raise errors[0]
        return n_frames

    def arm_rearrangement(self):
        """Run one rearrangement via software upload."""
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
            print("Reset-image acquisition failed; returning zeros.")
            img_array1 = np.zeros_like(img_array0)

        print(
            f"Rearrangement complete: {n_frames} frames, {(t3 - t1)*1000:.4f}ms total "
            f"(occupancy {(t2 - t1)*1000:.4f}ms, calculation+upload {(t3 - t2)*1000:.4f}ms)."
        )
        # Generation and upload are pipelined (see docstring), so they share one
        # measured span; both keys report it rather than a fabricated split.
        timings = {
            "n_frames": n_frames,
            "occupancy_extraction_ms": (t2 - t1)*1000,
            "calculation_and_upload_ms": (t3 - t2)*1000,
            "total_rearrangement_ms": (t3 - t1)*1000
        }
        return np.asarray(img_array0), np.asarray(img_array1), timings

    def arm_rearrangement_preload_trigger(self, pulser, period_us: float = 700, profile: str = None, restore_initial_array: bool = True):
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

        # 3. Pairing, interpolation and on-board preload, pipelined.
        frames = PM.iter_rearrangement_sequence(
            s["terms1"], s["terms2"], occ_mask,
            d0=s["d0"],
            profile=profile or s.get("profile", "minimum_jerk"),
            to_host=False,
        )
        self.slm.set_wait_for_trigger(False)  # must be off while preloading
        n_frames = self._preload_sequence_pipelined(frames)
        t3 = time.time()

        # 4. Clock the preloaded sequence out on hardware triggers.
        self.slm.set_wait_for_trigger(True)
        try:
            self.slm.start_auto_increment(n_frames)
            try:
                pulser.send_pulses(n_frames - 1, period_us=period_us)
                t4 = time.time()
            finally:
                self.slm.stop_auto_increment()
        finally:
            self.slm.set_wait_for_trigger(False)
        

        # 5. Reset image.
        try:
            self.camera.start_acquisition()
            img_array1 = self.camera.acquire_n_frames(1)[0]
        except Exception:
            print("Reset-image acquisition failed; returning zeros.")
            img_array1 = np.zeros_like(img_array0)

        if restore_initial_array:
            self.slm.update_mask(s["pm_init_uint8"])  # restore the initial array for the next run

        print(
            f"Rearrangement complete (preload+trigger): {n_frames} frames, {(t4 - t1)*1000:.4f}ms total "
            f"(occupancy {(t2 - t1)*1000:.4f}ms, calculation and preload {(t3 - t2)*1000:.4f}ms, hardware trigger upload {(t4 - t3)*1000:.4f}ms)."
        )
        timings = {
            "n_frames": n_frames,
            "occupancy_extraction_ms": (t2 - t1)*1000,
            "calculation_and_preload_ms": (t3 - t2)*1000,
            "hardware_trigger_upload_ms": (t4 - t3)*1000,
            "total_rearrangement_ms": (t4 - t1)*1000
        }
        return np.asarray(img_array0), np.asarray(img_array1), timings

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
                print("camera.stop_acquisition() during shutdown failed", exc_info=True)
        self._state = None
        self._initialised = False
