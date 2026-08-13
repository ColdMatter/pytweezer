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
import asyncio
import numpy as np


from pytweezer.coordinators.base import Coordinator
from pytweezer import analysis as an
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


def _steady_median(values):
    
    tail = values[1:] if len(values) > 1 else values
    return float(np.median(tail)) if tail else float("nan")


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


class Rearrangement(Coordinator):
    """Camera + SLM rearrangement loop, run entirely in one process."""

    camera_role = "camera"
    slm_role = "slm"
    pm_gen_role = "phasemask_generator"

    def __init__(self, targets, conf):
        super().__init__(targets, conf)
        self.camera: ImagEMX2Camera | None = self.targets.get(self.camera_role)
        self.slm: SLM = self.require_role(self.slm_role)
        self.pm_gen = self.targets.get(self.pm_gen_role)

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
        window_size: int = 5,
        feature_size: int = 10,
        profile: str = "linear",
        detector=None,
    ) -> None:

        self._require_gpu()
        self._require_camera()
        self.camera.timeout = 2

        # cp.asarray takes host or device arrays: a no-op for the cupy terms that
        # come straight out of superposition_optimization, a host->device copy for
        # numpy ones. np.asarray cannot sit in front of it - it is not dispatchable
        # under NEP-18, so a cupy input falls through to __array__ and raises.
        w1, phi1, x1, y1 = cp.asarray(data1)
        w2, phi2, x2, y2 = cp.asarray(data2)
        terms1 = (w1, phi1, x1, y1, array_shape1)
        terms2 = (w2, phi2, x2, y2, array_shape2)

        # The initial array to load onto the SLM before each rearrangement.
        pm_array_init = self.pm_gen.generate_phasemask(list(terms1))
        pm_init = self.pm_gen.superimpose([pm_array_init, self.pm_gen.fresnel, self.pm_gen.blaze, self.pm_gen.zernike])
        pm_init_uint8 = self.pm_gen.transform_phase_8bit(pm_init).get()

        self._state = dict(
            PM=self.pm_gen,
            terms1=terms1, terms2=terms2,
            pm_init_uint8=pm_init_uint8,
            d0=d0, fps=fps, threshold=threshold, grid_positions=grid_positions, window_size=window_size, feature_size=feature_size,
            profile=profile, detector=detector,
        )
        self._initialised = True
        if detector is not None:
            print(f"Occupancy via {detector.name} detector "
                  f"(window {detector.window_size}, threshold {detector.threshold:.1f}).")
        print("Rearrangement node initialised.")

    # ------------------------------------------------------------------ #
    # Arm / run one rearrangement
    # ------------------------------------------------------------------ #

    def _extract_occupancy(self, image, array_shape, threshold):
        
        detector = self._state.get("detector")
        if detector is not None:
            return detector.occupancy(image)

        from pytweezer import analysis_old as an

        grid_positions = self._state["grid_positions"]
        window_size = self._state["window_size"]
        feature_size = self._state["feature_size"]

        cpp_morph = _morph_cpp()
        cpp_morph = None
        if cpp_morph is not None and getattr(image, "dtype", None) == np.uint16:
            img = cpp_morph.tophat(np.ascontiguousarray(image, dtype=np.uint16),
                                   feature_size=feature_size)
        else:
            img = an.morphological_tophat_high_pass(image, feature_size=feature_size)

        cpp_sum = _sum_cpp()
        cpp_sum = None
        if cpp_sum is not None and getattr(img, "dtype", None) == np.uint16:
            pixel_sums = cpp_sum.sum_pixel_values(
                img, grid_positions, array_shape, window_size=window_size
            )
        else:
            pixel_sums = an.sum_pixel_values(
                img, grid_positions, array_shape, window_size=window_size
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
        # perf_counter, not time(): time() is quantised to ~1 ms on Windows, which is
        # the same order as the spans measured here.
        t1 = time.perf_counter()

        # 2. Occupancy mask.
        occ_mask = self._extract_occupancy(img_array0, arr_shape1, s["threshold"])
        t2 = time.perf_counter()

        # 3. Pairing, interpolation and SLM upload, pipelined.
        frames = PM.iter_rearrangement_sequence(
            s["terms1"], s["terms2"], occ_mask,
            d0=s["d0"],
            profile=s.get("profile", "minimum_jerk"),
            to_host=False,
        )
        n_frames = self._play_sequence_pipelined(frames)
        t3 = time.perf_counter()

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


    def _play_sequence_pipelined_timings(self, frames):
        
        upload_queue = queue.Queue(maxsize=UPLOAD_QUEUE_DEPTH)
        errors = []
        first_frame_at = []
        events = []
        gpu_wait_ms, transfer_ms, display_ms, pacing_ms = [], [], [], []
        pinned_alloc_ms = [float("nan")]
        self.last_first_frame_at = None

        def writer():
            staging = None
            next_slot = None
            while True:
                item = upload_queue.get()
                if item is None:
                    return
                if errors:
                    continue  # drain the rest so the producer never blocks
                frame, start_ev, end_ev = item
                try:
                    # First frame only: page-locked staging buffer. Timed on its
                    # own because it is a per-run cost that lands entirely on
                    # frame 0 and would otherwise inflate that frame's transfer.
                    if staging is None:
                        a0 = time.perf_counter()
                        staging = _pinned_like(frame)
                        pinned_alloc_ms[0] = (time.perf_counter() - a0) * 1000

                    w0 = time.perf_counter()
                    end_ev.synchronize()
                    w1 = time.perf_counter()
                    frame.get(out=staging)
                    w2 = time.perf_counter()
                    self.slm.update_mask(staging)
                    w3 = time.perf_counter()

                    if not first_frame_at:
                        first_frame_at.append(w3)

                    if self.panel_period_s:
                        now = w3
                        if next_slot is not None and now < next_slot:
                            _wait_until(next_slot)
                            now = next_slot
                        next_slot = now + self.panel_period_s
                    w4 = time.perf_counter()

                    gpu_wait_ms.append((w1 - w0) * 1000)
                    transfer_ms.append((w2 - w1) * 1000)
                    display_ms.append((w3 - w2) * 1000)
                    pacing_ms.append((w4 - w3) * 1000)
                    events.append((start_ev, end_ev))
                except Exception as exc:
                    errors.append(exc)

        thread = threading.Thread(target=writer, name="slm-upload-timed", daemon=True)
        thread.start()

        n_frames = 0
        try:
            for item in frames:
                upload_queue.put(item)
                n_frames += 1
        finally:
            upload_queue.put(None)
            thread.join()

        if errors:
            raise errors[0]
        if first_frame_at:
            self.last_first_frame_at = first_frame_at[0]

        # Read the events only now that every frame has completed. Calling
        # get_elapsed_time on a still-pending event would block and perturb the
        # very run it is measuring.
        gpu_compute_ms = [float(cp.cuda.get_elapsed_time(s, e)) for s, e in events]

        return n_frames, {
            "gpu_compute_ms": gpu_compute_ms,
            "gpu_wait_ms": gpu_wait_ms,
            "transfer_ms": transfer_ms,
            "display_ms": display_ms,
            "pacing_ms": pacing_ms,
            "pinned_alloc_ms": pinned_alloc_ms[0],
        }

    @staticmethod
    def _pairing_record(plan):
        """Host-side copy of the JV assignment produced by ``plan_rearrangement``.

        Indices are flat trap indices: ``moving_idx``/``occ_mask`` index the
        initial array in the same order as the occupancy mask (i.e. the scorer's
        ``np.fliplr(score_grid).flatten()`` order), ``final_idx`` indexes the
        target array. Together they say which atom was sent to which target site,
        which is what a per-atom survival analysis needs afterwards.

        Called only once the run is over - every ``.get()`` here is a device sync
        and inside a timed span would be billed as setup or frame time.
        """
        occ_mask = cp.asnumpy(plan["occ_mask"]).astype(bool)
        moving_idx = cp.asnumpy(plan["moving_idx"]).astype(np.int32)
        final_idx = cp.asnumpy(plan["final_idx"]).astype(np.int32)
        n_targets = int(plan["n_targets"])

        occupied_idx = np.flatnonzero(occ_mask).astype(np.int32)
        filled = np.zeros(n_targets, dtype=bool)
        filled[final_idx] = True

        return {
            "occ_mask": occ_mask,
            "occupied_idx": occupied_idx,
            "moving_idx": moving_idx,
            "final_idx": final_idx,
            "pos_init": cp.asnumpy(plan["pos_init"]).astype(np.float32),
            "pos_final": cp.asnumpy(plan["pos_final"]).astype(np.float32),
            # Loaded sites the assignment left behind (more atoms than targets);
            # these traps are switched off by the sequence.
            "discarded_idx": np.setdiff1d(occupied_idx, moving_idx).astype(np.int32),
            # Target sites no atom was available for (more targets than atoms).
            "unfilled_target_idx": np.flatnonzero(~filled).astype(np.int32),
            "n_occupied": int(occupied_idx.size),
            "n_moving": int(moving_idx.size),
            "n_targets": n_targets,
        }

    def arm_rearrangement_timings(self):

        self._require_gpu()
        self._require_initialised()

        s = self._state
        PM = s["PM"]
        arr_shape1 = s["terms1"][4]

        self.slm.update_mask(s["pm_init_uint8"])

        # 1. Acquire the occupancy image.
        self.camera.start_acquisition()
        img_array0 = self.camera.acquire_n_frames(1)[0]
        t1 = time.perf_counter()

        # 2. Occupancy mask.
        occ_mask = self._extract_occupancy(img_array0, arr_shape1, s["threshold"])
        t2 = time.perf_counter()

        # 3. One-time pairing and interpolation setup, eager so it lands in its
        #    own span instead of on frame 0.
        plan = PM.plan_rearrangement(
            s["terms1"], s["terms2"], occ_mask,
            d0=s["d0"],
            profile=s.get("profile", "minimum_jerk"),
        )
        t_plan = time.perf_counter()

        # 4. Per-frame generation, transfer and upload, still pipelined.
        frames = PM.iter_rearrangement_sequence_timings(plan, to_host=False)
        n_frames, stage = self._play_sequence_pipelined_timings(frames)
        t3 = time.perf_counter()

        # 5. Reset image.
        try:
            self.camera.start_acquisition()
            img_array1 = self.camera.acquire_n_frames(1)[0]
        except Exception:
            print("Reset-image acquisition failed; returning zeros.")
            img_array1 = np.zeros_like(img_array0)

        # After the reset image, so the syncs it costs land outside every timed
        # span and never delay arming the camera.
        pairing = self._pairing_record(plan)

        ttff_ms = (
            (self.last_first_frame_at - t2) * 1000
            if self.last_first_frame_at is not None else float("nan")
        )
        med = {k: _steady_median(stage[k])
               for k in ("gpu_compute_ms", "gpu_wait_ms", "transfer_ms",
                         "display_ms", "pacing_ms")}

        print(
            f"Rearrangement complete: {n_frames} frames, {(t3 - t1)*1000:.4f}ms total "
            f"(occupancy {(t2 - t1)*1000:.4f}ms, pairing+setup {(t_plan - t2)*1000:.4f}ms, "
            f"streaming {(t3 - t_plan)*1000:.4f}ms).\n"
            f"  per-frame medians (frame 0 excluded): compute {med['gpu_compute_ms']:.4f}ms, "
            f"gpu-wait {med['gpu_wait_ms']:.4f}ms, transfer {med['transfer_ms']:.4f}ms, "
            f"board-write {med['display_ms']:.4f}ms, pacing {med['pacing_ms']:.4f}ms.\n"
            f"  time to first frame {ttff_ms:.4f}ms, "
            f"pinned alloc {stage['pinned_alloc_ms']:.4f}ms."
        )
        timings = {
            "n_frames": n_frames,
            "occupancy_extraction_ms": (t2 - t1) * 1000,
            "calculation_and_upload_ms": (t3 - t2) * 1000,
            "total_rearrangement_ms": (t3 - t1) * 1000,

            # One-time costs, now separated out of calculation_and_upload_ms.
            "pairing_and_setup_ms": (t_plan - t2) * 1000,
            "streaming_ms": (t3 - t_plan) * 1000,
            "time_to_first_frame_ms": ttff_ms,
            "pinned_alloc_ms": stage["pinned_alloc_ms"],
            "n_moving": plan["n_moving"],

            # Per-frame series, one entry per frame, in frame order.
            "gpu_compute_ms": stage["gpu_compute_ms"],
            "gpu_wait_ms": stage["gpu_wait_ms"],
            "transfer_ms": stage["transfer_ms"],
            "display_ms": stage["display_ms"],
            "pacing_ms": stage["pacing_ms"],

            # Steady-state medians of the above, frame 0 dropped.
            "gpu_compute_median_ms": med["gpu_compute_ms"],
            "gpu_wait_median_ms": med["gpu_wait_ms"],
            "transfer_median_ms": med["transfer_ms"],
            "display_median_ms": med["display_ms"],
            "pacing_median_ms": med["pacing_ms"],
        }
        return np.asarray(img_array0), np.asarray(img_array1), timings, pairing

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
        # perf_counter, not time(): time() is quantised to ~1 ms on Windows, which is
        # the same order as the spans measured here.
        t1 = time.perf_counter()

        # 2. Occupancy mask.
        occ_mask = self._extract_occupancy(img_array0, arr_shape1, s["threshold"])
        t2 = time.perf_counter()

        # 3. Pairing, interpolation and on-board preload, pipelined.
        frames = PM.iter_rearrangement_sequence(
            s["terms1"], s["terms2"], occ_mask,
            d0=s["d0"],
            profile=profile or s.get("profile", "minimum_jerk"),
            to_host=False,
        )
        self.slm.set_wait_for_trigger(False)  # must be off while preloading
        n_frames = self._preload_sequence_pipelined(frames)
        t3 = time.perf_counter()

        # 4. Clock the preloaded sequence out on hardware triggers.
        trigger_span_s = float("nan")
        self.slm.set_wait_for_trigger(True)
        try:
            self.slm.start_auto_increment(n_frames)
            try:
                trigger_span_s = pulser.send_pulses(n_frames - 1, period_us=period_us)
            finally:
                self.slm.stop_auto_increment()
        finally:
            self.slm.set_wait_for_trigger(False)
        t4 = time.perf_counter()

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
            "total_rearrangement_ms": (t4 - t1)*1000, 
            "trigger_span_s": trigger_span_s,
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
