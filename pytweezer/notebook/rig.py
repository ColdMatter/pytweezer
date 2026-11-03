"""Run MOTMaster shots from a notebook and get every camera's frames back.

::

    from pytweezer.notebook import Rig

    rig = Rig("CaF MotMaster", {"CaF": "CaF ThorCam", "Rb": "Rb HamCam"})
    shot = rig.run("LinkedRbCaFMOT", dummy=0.0)
    images, params = shot.images["CaF"], shot.params

Each camera is armed once and runs free; a shot's frames are whatever each camera
caught between the mark taken just before ``Go`` and the moment its frame count
stops changing afterwards, so no frame count is configured anywhere. When saving,
the frames are added to the zip MOTMaster wrote for the run (see
:mod:`pytweezer.notebook.zips`), so ``analysis3.readFromZip`` keeps working.
"""

import functools
import json
import re
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from pytweezer.drivers.shot_frames import settle_frame_counts
from pytweezer.experiment.archive import fetch_measurement
from pytweezer.experiment.client import ExperimentManagerClient, submit, wait
from pytweezer.experiment.optimisers import (
    accept_proposals,
    changed_summary,
    live_task_using,
    proposals,
)
from pytweezer.experiment.task import Task, TaskStatus
from pytweezer.experiments.caf_fluorescence import LASERS
from pytweezer.logging_utils import get_logger
from pytweezer.notebook import zips
from pytweezer.optimisers.optimise_mot_lasers import LASER_KEYS, OptimiseMOTLasers
from pytweezer.servers.device_client import get_device, get_device_config

logger = get_logger("notebook rig")


@dataclass
class Shot:
    """One MOTMaster run: each camera's frames, the parameters, and where it was saved.

    ``images[key]`` is a float ``(n, H, W)`` stack, as ``readFromZip`` returns, and
    holds every iteration's frames in order.
    """

    script: str
    params: dict[str, Any]
    images: dict[str, np.ndarray]
    zips: dict[str, Path] = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0

    @property
    def zip_number(self) -> int | None:
        """The run number MOTMaster gave the saved zip, or ``None`` if unsaved."""
        if not self.zips:
            return None
        return int(re.search(r"_(\d+)\.zip$", next(iter(self.zips.values())).name)[1])


def _caller_stacklevel() -> int:
    """``warnings.warn`` stacklevel of the first frame outside this module."""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename == __file__:
        frame, level = frame.f_back, level + 1
    return level


def _reconnect_if_interrupted(method):
    # sipyco reads a call's reply after sending it; an interrupt in between leaves
    # the reply queued, and every later call on that client gets the one before's.
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException as error:
            if not isinstance(error, Exception) and not getattr(
                error, "_rig_reconnected", False
            ):
                self._reconnect()
                error._rig_reconnected = True
            raise

    return wrapper


def _motmaster_file_config(name: str) -> dict[str, Any]:
    from pytweezer.drivers.motmaster import CONFIG_DIR

    with open(CONFIG_DIR / get_device_config(name)["config_file"]) as file:
        return json.load(file)


class Rig:
    """One MOTMaster and the cameras imaging its shots, driven from a notebook.

    Args:
        motmaster: Device name, or an object with the MOTMaster driver's methods.
        cameras: ``{key: device name or camera object}``. The camera whose key is
            ``element`` saves into MOTMaster's zip; every other key gets a sibling.
        linked: Run scripts in MOTMaster's linked mode (see :attr:`linked`).
        element: MOTMaster's data prefix (``"CaF"``); read from the device's
            config file when ``motmaster`` is a name.
        data_root: Where MOTMaster saves its zips; as for ``element``.
        buffer_frames: Ring-buffer size each camera is armed with.
        frame_settle_s: How long every camera's frame count must stay unchanged
            after ``Go`` before the frames are read.
        frame_timeout_s: Longest wait for the counts to settle.
        frame_poll_s: Interval between frame-count polls while settling.
    """

    def __init__(
        self,
        motmaster: str | Any = "CaF MotMaster",
        cameras: dict[str, str | Any] | None = None,
        *,
        linked: bool = False,
        element: str | None = None,
        data_root: Path | str | None = None,
        buffer_frames: int = 1000,
        frame_settle_s: float = 0.3,
        frame_timeout_s: float = 5.0,
        frame_poll_s: float = 0.05,
    ) -> None:
        self._owned: list[Any] = []
        self._motmaster_name = motmaster if isinstance(motmaster, str) else None
        self._camera_names = {
            key: camera
            for key, camera in (cameras or {}).items()
            if isinstance(camera, str)
        }
        self.simulated = False
        if isinstance(motmaster, str):
            file_config = _motmaster_file_config(motmaster)
            element = element or file_config["data_prefix"]
            data_root = data_root or file_config["data_root_path"]
            self.simulated = bool(get_device_config(motmaster).get("simulate"))
            motmaster = self._connect(motmaster)
        if element is None or data_root is None:
            raise ValueError(
                "Give element and data_root when passing a MOTMaster object"
            )
        self.motmaster = motmaster
        self.element = element
        self.data_root = Path(data_root)
        self.cameras = {
            key: self._connect(camera) if isinstance(camera, str) else camera
            for key, camera in (cameras or {}).items()
        }
        self.linked = linked
        self.buffer_frames = buffer_frames
        self.frame_settle_s = frame_settle_s
        self.frame_timeout_s = frame_timeout_s
        self.frame_poll_s = frame_poll_s
        self.last_shots: list[Shot] = []
        self._linked_set: bool | None = None
        self._last_counts: dict[tuple[str, str], int] = {}
        self._shots_run = 0
        self._warned_simulated = False
        for camera in self.cameras.values():
            camera.arm(buffer_frames)

    def _connect(self, name: str) -> Any:
        client = get_device(name)
        self._owned.append(client)
        return client

    def _reconnect(self) -> None:
        for client in self._owned:
            try:
                client.close_rpc()
            except Exception:
                logger.warning("Could not close a stale client", exc_info=True)
        self._owned = []
        if self._motmaster_name is not None:
            self.motmaster = self._connect(self._motmaster_name)
            self._linked_set = None
        for key, name in self._camera_names.items():
            self.cameras[key] = self._connect(name)

    @_reconnect_if_interrupted
    def run(
        self, script: str, *, save: bool = True, iterations: int = 1, **params: Any
    ) -> Shot:
        """Run ``script`` once (all ``iterations`` in one ``Go``) and collect frames.

        Args:
            script: MOTMaster script name, without ``.cs``.
            save: Let MOTMaster save the run and add the frames to its zip.
            iterations: MOTMaster iterations; their frames share one shot and zip.
            **params: Script parameters to override.
        """
        self.last_shots = []
        return self._run(script, save, iterations, params)

    @_reconnect_if_interrupted
    def scan(
        self,
        script: str,
        parameter: str,
        values: Sequence[float],
        *,
        save: bool = True,
        iterations: int = 1,
        **params: Any,
    ) -> list[Shot]:
        """One :meth:`run` per value of the script parameter ``parameter``.

        Shots taken so far stay in :attr:`last_shots` if the scan is interrupted.
        """
        self.last_shots = []
        return [
            self._run(script, save, iterations, {**params, parameter: value})
            for value in values
        ]

    @_reconnect_if_interrupted
    def scan_laser(
        self,
        script: str,
        laser: str,
        values: Sequence[float],
        *,
        save: bool = True,
        iterations: int = 1,
        settle_s: float = 0.0,
        **params: Any,
    ) -> list[Shot]:
        """One :meth:`run` per wavemeter-lock set point (THz) of ``laser``.

        The set point is walked to the first value, then held ``settle_s`` before
        the first shot. Each shot's ``params[laser]`` is its set point.
        """
        self.last_shots = []
        values = [float(value) for value in values]
        self.motmaster.ramp_laser_set_point_wml(laser, values[0])
        time.sleep(settle_s)
        shots = []
        for value in values:
            self.motmaster.set_laser_set_point_wml(laser, value)
            shot = self._run(script, save, iterations, params)
            shot.params[laser] = value
            shots.append(shot)
        return shots

    def auto_mot(
        self,
        stages: Sequence[str],
        *,
        spans: dict[str, float] | None = None,
        points: dict[str, int] | None = None,
        centres: dict[str, float] | None = None,
        display: bool = False,
        accept: bool = False,
        **arguments: Any,
    ) -> dict[str, dict[str, Any]]:
        """Tune the CaF lasers for the most fluorescence with the queued optimiser.

        Submits :class:`~pytweezer.optimisers.optimise_mot_lasers.OptimiseMOTLasers`
        scanning exactly the lasers in ``stages`` and waits for it. Its proposals
        are saved only with ``accept``, or when accepted in the Optimisers tab.
        It does not use this rig's devices, so it must not run alongside a shot
        from this rig.
        Interrupting it (Ctrl-C) terminates the run, which then returns the lasers.
        Run :func:`~pytweezer.optimisers.optimise_mot_lasers.create_saved_set_points`
        once before the first run.

        Args:
            stages: The lasers to scan, a subset of
                :data:`~pytweezer.experiments.caf_fluorescence.LASERS`; they are
                scanned in that order whatever order is given here.
            spans: Half-width of each laser's scan, in THz, by laser; the
                optimiser's default for lasers not given.
            points: Number of set points scanned, by laser.
            centres: Set point each laser's scan is centred on, in THz, by laser;
                the saved ``caf_laser_*`` for lasers not given.
            display: Plot the result with :func:`plot_auto_mot_results`.
            accept: Accept every proposal of the run as the Optimisers tab's
                Accept does, saving it for the whole lab, then ramp each
                scanned laser to its new set point. The ramp is skipped, with a
                warning, while a running task uses a device the optimiser
                reserves.
            **arguments: Any other argument of the optimiser, e.g. ``mot_field``.

        Returns:
            Per laser in ``stages``: ``scan_range`` (set points in THz, one per
            point), ``numbers`` (background-subtracted counts), ``errors``,
            ``set_point`` proposed and ``start_set_point`` (the lock's value when
            the run started). Pass it to :func:`plot_auto_mot_results`, or set
            ``display``.

        Raises:
            TypeError: If ``stages`` is a single string.
            KeyError: If a laser is not in ``LASERS``.
            RuntimeError: If the run did not complete, or if accepting, when
                the lasers could not be ramped; the accepted values stay saved.
        """
        if isinstance(stages, str):
            raise TypeError(
                f"auto_mot takes a list of laser names, e.g. ['v0', 'v1'], not "
                f"{stages!r}; the lasers are {list(LASERS)}"
            )
        spans, points, centres = spans or {}, points or {}, centres or {}
        named = {*stages, *spans, *points, *centres}
        if unknown := sorted(named - set(LASERS)):
            raise KeyError(f"unknown lasers {unknown}; the lasers are {list(LASERS)}")
        laser_arguments = {f"{laser}_run": laser in stages for laser in LASERS}
        for suffix, by_laser in (
            ("span", spans),
            ("points", points),
            ("centre", centres),
        ):
            laser_arguments.update(
                {f"{laser}_{suffix}": value for laser, value in by_laser.items()}
            )
        rid = submit(OptimiseMOTLasers, **laser_arguments, **arguments)
        try:
            task = wait(rid)
        except KeyboardInterrupt:
            client = ExperimentManagerClient()
            try:
                client.terminate(rid)
            finally:
                client.close()
            logger.warning(
                "Terminated optimiser run %d; it is returning the lasers to where "
                "they started",
                rid,
            )
            raise
        if task.status != TaskStatus.COMPLETED:
            raise RuntimeError(
                f"optimiser run {rid} {task.status.value}: {task.error or 'no error recorded'}"
            )

        measurement = fetch_measurement(rid)
        recorded = measurement.results
        start_set_points = measurement.constants["start_set_points"]
        results: dict[str, dict[str, Any]] = {}
        for laser in LASERS:
            if laser not in stages:
                continue
            on_laser = recorded["stage"] == LASERS.index(laser)
            results[laser] = {
                "scan_range": recorded["setting"][on_laser].tolist(),
                "numbers": recorded["counts"][on_laser].tolist(),
                "errors": recorded["counts_error"][on_laser].tolist(),
                "set_point": float(
                    measurement.constants[f"proposed.{LASER_KEYS[laser]}"]
                ),
                "start_set_point": start_set_points[laser],
            }
        if display:
            plot_auto_mot_results(results)
        if accept:
            _accept_and_apply(rid, measurement.constants)
        return results

    @_reconnect_if_interrupted
    def laser_set_points(self) -> dict[str, float]:
        """Every laser's wavemeter-lock set point, in THz."""
        return self.motmaster.laser_set_points_wml()

    def _run(
        self, script: str, save: bool, iterations: int, params: dict[str, Any]
    ) -> Shot:
        mm = self.motmaster
        # numpy scalars from np.arange/np.linspace don't all survive the RPC.
        params = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in params.items()
        }
        if save and self.simulated:
            if not self._warned_simulated:
                warnings.warn(
                    "The simulated MOTMaster writes no zips; running with save=False",
                    stacklevel=_caller_stacklevel(),
                )
                self._warned_simulated = True
            save = False
        # Re-claimed every run: MOTMaster drops the link if the secondary takes
        # back control, and a linked script must not then run on one machine.
        if self.linked or self._linked_set is not False:
            mm.set_linked(self.linked)
            self._linked_set = self.linked
        mm.set_motmaster_experiment(script)
        mm.set_run_until_stopped(False)
        mm.set_trigger_mode(False)
        mm.set_iterations(int(iterations))
        mm.set_save_toggle(save)

        # Another user of a shared camera may have stopped it since the last shot.
        for camera in self.cameras.values():
            camera.arm(self.buffer_frames)
        marks = {key: camera.frames_acquired() for key, camera in self.cameras.items()}
        today = date.today()
        folders = [
            zips.day_folder(self.data_root, today),
            zips.day_folder(self.data_root, today + timedelta(days=1)),
        ]
        before = zips.snapshot(folders, self.element) if save else set()

        self._shots_run += 1
        t_start = time.time()
        mm.start_motmaster_experiment(params)
        t_end = time.time()
        self._settle()
        images = {
            key: np.asarray(camera.read_frames_from(marks[key]), dtype=float)
            for key, camera in self.cameras.items()
        }
        self._check_counts(script, images)

        shot = Shot(
            script=script,
            params={**mm.get_params(), **params},
            images=images,
            t_start=t_start,
            t_end=t_end,
        )
        if save:
            shot.zips = self._save(
                zips.find_new_zip(before, folders, self.element), images
            )
        self.last_shots.append(shot)
        return shot

    def _settle(self) -> None:
        settle_frame_counts(
            self.cameras, self.frame_settle_s, self.frame_timeout_s, self.frame_poll_s
        )

    def _check_counts(self, script: str, images: dict[str, np.ndarray]) -> None:
        for key, stack in images.items():
            count = len(stack)
            previous = self._last_counts.get((script, key))
            if previous is not None and previous != count:
                # The shot number keeps Python from showing a repeat only once.
                message = (
                    f"{key}: {count} frames from {script!r}, {previous} last time "
                    f"(shot {self._shots_run})"
                )
                logger.warning(message)
                warnings.warn(message, stacklevel=_caller_stacklevel())
            self._last_counts[(script, key)] = count

    def _save(self, main_zip: Path, images: dict[str, np.ndarray]) -> dict[str, Path]:
        saved = {}
        for key, stack in images.items():
            settings = self.cameras[key].settings()
            if key == self.element:
                zips.inject_frames(main_zip, key, stack, settings)
                saved[key] = main_zip
            else:
                saved[key] = zips.write_sibling(
                    main_zip, self.element, key, stack, settings
                )
        return saved

    def close(self) -> None:
        """Release a link this rig made, and close the device clients it opened."""
        if self._linked_set:
            self.motmaster.set_linked(False)
            self._linked_set = False
        for client in self._owned:
            client.close_rpc()
        self._owned = []


def _accept_and_apply(rid: int, constants: dict[str, Any]) -> None:
    """Accept every proposal of :class:`OptimiseMOTLasers` run ``rid``, then apply them."""
    values = proposals(constants, OptimiseMOTLasers.proposes())
    reserved = OptimiseMOTLasers.reserved_devices()
    client = ExperimentManagerClient()
    try:
        # Read before saving, so a failure here leaves nothing saved.
        snapshot = client.snapshot()
        running = [
            Task.model_validate(entry) for entry in snapshot.get("running") or []
        ]
        busy = live_task_using(running, reserved)
        changes = accept_proposals(
            values, rid, OptimiseMOTLasers.optimiser_schema(), client=client
        )
    finally:
        client.close()
    accepted = f"Accepted run {rid}: {changed_summary(changes)}"
    if busy:
        busy_rid, shared = busy
        warnings.warn(
            f"{accepted}; not applied: run {busy_rid} is using {', '.join(shared)}",
            stacklevel=_caller_stacklevel(),
        )
        return
    try:
        OptimiseMOTLasers.apply(values)
    except Exception as error:
        reason = str(error) or type(error).__name__
        raise RuntimeError(
            f"Accepted run {rid}, but applying OptimiseMOTLasers failed: {reason}"
        ) from error
    logger.info("%s and applied: %s", accepted, values)


def plot_auto_mot_results(results: dict[str, dict[str, Any]]):
    """Plot the scans returned by :meth:`Rig.auto_mot`: one panel per laser,
    normalised number against detuning from the set point the laser started on, with
    an arrow marking the set point chosen. Returns the ``(figure, axes)`` pair."""
    import matplotlib.pyplot as plt

    n_cols = min(3, max(1, len(results)))
    n_rows = max(1, -(-len(results) // n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True, squeeze=False
    )
    flat_axes = axes.flatten()
    for ax, (laser, scan) in zip(flat_axes, results.items()):
        numbers = np.array(scan["numbers"])
        errors = np.array(scan["errors"])
        scale = np.max(numbers) if np.max(numbers) > 0 else 1.0
        # THz set points shown as MHz detunings from where the laser started.
        start = scan["start_set_point"]
        detunings = (np.array(scan["scan_range"]) - start) * 1e6
        ax.errorbar(detunings, numbers / scale, yerr=errors / scale, fmt="ok")
        ax.arrow(
            (scan["set_point"] - start) * 1e6,
            1.15,
            0,
            -0.1,
            head_width=2.0,
            head_length=0.03,
            width=1.0,
            fc="r",
            ec="r",
        )
        ax.set_title(f"{laser} set @ {scan['set_point']} THz")
        ax.set_xlabel("Detuning to previous set point [MHz]")
        ax.set_ylim((0, 1.2))
    for ax in flat_axes[len(results) :]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("Norm. number")
    fig.tight_layout()
    return fig, axes
