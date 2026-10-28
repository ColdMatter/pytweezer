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

import json
import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from pytweezer.logging_utils import get_logger
from pytweezer.notebook import zips
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
        if isinstance(motmaster, str):
            file_config = _motmaster_file_config(motmaster)
            element = element or file_config["data_prefix"]
            data_root = data_root or file_config["data_root_path"]
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
        self.frame_settle_s = frame_settle_s
        self.frame_timeout_s = frame_timeout_s
        self.frame_poll_s = frame_poll_s
        self.last_shots: list[Shot] = []
        self._linked_set: bool | None = None
        self._last_counts: dict[tuple[str, str], int] = {}
        for camera in self.cameras.values():
            camera.arm(buffer_frames)

    def _connect(self, name: str) -> Any:
        client = get_device(name)
        self._owned.append(client)
        return client

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
        script: str,
        scan_ranges: dict[str, Sequence[float]],
        *,
        camera: str | None = None,
        iterations: int = 1,
        field_parameter: str = "MOTCoilsCurrentValue",
        bg_field_value: float = 0.0,
        mot_field_value: float = 1.0,
        save: bool = False,
        display: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Tune the wavemeter-lock set points laser by laser for the most fluorescence.

        A background shot is taken with ``field_parameter`` at ``bg_field_value``;
        then each laser in ``scan_ranges`` (set points in THz) is scanned with the
        field at ``mot_field_value`` and left on the point with the most
        background-subtracted counts before the next laser is scanned.

        Args:
            camera: Key of the camera to count on; defaults to :attr:`element`'s.

        Returns:
            Per laser: ``scan_range``, ``numbers`` (mean counts over a shot's
            frames), ``errors`` (their standard error; NaN for one frame),
            ``set_point`` chosen and ``start_set_point``. Pass it to
            :func:`plot_auto_mot_results`, or set ``display``.
        """
        key = camera or self.element
        start_set_points = self.laser_set_points()
        unknown = [laser for laser in scan_ranges if laser not in start_set_points]
        if unknown:
            raise KeyError(
                f"{unknown} not among the MOTMaster's lasers {sorted(start_set_points)}"
            )
        background = (
            self.run(
                script,
                save=save,
                iterations=iterations,
                **{field_parameter: bg_field_value},
            )
            .images[key]
            .mean(axis=0)
        )

        results: dict[str, dict[str, Any]] = {}
        for laser, scan_range in scan_ranges.items():
            scan_range = [float(value) for value in scan_range]
            shots = self.scan_laser(
                script,
                laser,
                scan_range,
                save=save,
                iterations=iterations,
                **{field_parameter: mot_field_value},
            )
            numbers, errors = [], []
            for shot in shots:
                counts = np.sum(shot.images[key] - background, axis=(1, 2))
                numbers.append(float(np.mean(counts)))
                errors.append(
                    float(np.std(counts, ddof=1) / np.sqrt(len(counts)))
                    if len(counts) > 1
                    else float("nan")
                )
            set_point = scan_range[int(np.argmax(numbers))]
            self.motmaster.set_laser_set_point_wml(laser, set_point)
            results[laser] = {
                "scan_range": scan_range,
                "numbers": numbers,
                "errors": errors,
                "set_point": set_point,
                "start_set_point": start_set_points[laser],
            }
        if display:
            plot_auto_mot_results(results)
        return results

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
        if self._linked_set != self.linked:
            mm.set_linked(self.linked)
            self._linked_set = self.linked
        mm.set_motmaster_experiment(script)
        mm.set_run_until_stopped(False)
        mm.set_trigger_mode(False)
        mm.set_iterations(int(iterations))
        mm.set_save_toggle(save)

        marks = {key: camera.frames_acquired() for key, camera in self.cameras.items()}
        today = date.today()
        folders = [
            zips.day_folder(self.data_root, today),
            zips.day_folder(self.data_root, today + timedelta(days=1)),
        ]
        before = zips.snapshot(folders, self.element) if save else set()

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
        def counts():
            return {
                key: camera.frames_acquired() for key, camera in self.cameras.items()
            }

        last = counts()
        now = time.monotonic()
        stable_since, deadline = now, now + self.frame_timeout_s
        while now - stable_since < self.frame_settle_s and now < deadline:
            time.sleep(self.frame_poll_s)
            current = counts()
            now = time.monotonic()
            if current != last:
                last, stable_since = current, now

    def _check_counts(self, script: str, images: dict[str, np.ndarray]) -> None:
        for key, stack in images.items():
            count = len(stack)
            previous = self._last_counts.get((script, key))
            if previous is not None and previous != count:
                warnings.warn(
                    f"{key}: {count} frames from {script!r}, {previous} last time",
                    stacklevel=4,
                )
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
