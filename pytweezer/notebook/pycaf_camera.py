"""Save a pytweezer camera's frames into the zips of MOTMaster runs fired by pycaf.

::

    from pytweezer.notebook import PycafCamera

    hamcam = PycafCamera(expt, "Rb HamCam")
    expt.scan_motmaster_parameter(script, "tBlueMOTField", values)
    hamcam_images = hamcam.store_images()

``expt`` is a connected :class:`pycaf.Experiment`. The camera is put in external
exposure and live mode, and every MOTMaster ``Go`` that ``expt`` makes from then on
(any scan method, or ``motmaster_single_run``) is watched: the frames that arrive
during the run are read and paired with the zip MOTMaster saved for it.
:meth:`PycafCamera.store_images` writes each run's frames to a zip beside
MOTMaster's (``Rb{ddMMMyy}{batch}_{nnn}.zip`` for the default ``key``) and
returns them, one stack per run, then starts afresh for the next scan.
"""

import json
import re
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from pytweezer.drivers.shot_frames import settle_frame_counts
from pytweezer.logging_utils import get_logger
from pytweezer.notebook import zips
from pytweezer.servers.device_client import get_device_config

logger = get_logger("notebook pycaf camera")


def _motmaster_file_config(name: str) -> dict[str, Any]:
    from pytweezer.drivers.motmaster import CONFIG_DIR

    with open(CONFIG_DIR / get_device_config(name)["config_file"]) as file:
        return json.load(file)


@dataclass
class _Run:
    frames: np.ndarray
    zip: Path | None


class PycafCamera:
    """A pytweezer camera whose frames are saved with each pycaf MOTMaster run.

    Args:
        expt: A connected ``pycaf.Experiment``; its ``Go`` calls are watched.
        camera: Device name, or an object with the camera driver's methods.
        key: Prefix of the zips the frames are saved in, in place of MOTMaster's
            element (``"Rb"`` makes ``Rb.readFromZip(n)`` find run ``n``'s frames).
        external_exposure: Put the camera in external exposure (the trigger
            pulse is the exposure) before going live.
        motmaster: pytweezer device name of the MOTMaster ``expt`` drives, for
            its data folder and element.
        element: MOTMaster's data prefix; read from ``motmaster``'s config if not given.
        data_root: Where MOTMaster saves its zips; as for ``element``.
        buffer_frames: Ring-buffer size of the camera's live mode.
        frame_settle_s: How long the frame count must stay unchanged after a
            ``Go`` before the run's frames are read.
        frame_timeout_s: Longest wait for the count to settle.
        frame_poll_s: Interval between frame-count polls while settling.
    """

    def __init__(
        self,
        expt: Any,
        camera: str | Any = "Rb HamCam",
        *,
        key: str = "Rb",
        external_exposure: bool = True,
        motmaster: str = "CaF MotMaster",
        element: str | None = None,
        data_root: Path | str | None = None,
        buffer_frames: int = 1000,
        frame_settle_s: float = 0.3,
        frame_timeout_s: float = 5.0,
        frame_poll_s: float = 0.05,
    ) -> None:
        if element is None or data_root is None:
            file_config = _motmaster_file_config(motmaster)
            element = element or file_config["data_prefix"]
            data_root = data_root or file_config["data_root_path"]
        if key == element:
            raise ValueError(
                f"key {key!r} is MOTMaster's own prefix; its zips already hold "
                "another camera's frames, so pick another key"
            )
        self.expt = expt
        self.key = key
        self.element = element
        self.data_root = Path(data_root)
        self.buffer_frames = buffer_frames
        self.frame_settle_s = frame_settle_s
        self.frame_timeout_s = frame_timeout_s
        self.frame_poll_s = frame_poll_s
        self.last_zips: list[Path | None] = []
        self._camera_name = camera if isinstance(camera, str) else None
        self._owns_camera = isinstance(camera, str)
        self.camera = self._connect() if isinstance(camera, str) else camera
        self._runs: list[_Run] = []
        self._original_go = None

        if external_exposure:
            # DCAM refuses trigger settings while acquiring (DCAMERR_BUSY).
            self.camera.stop_live()
            self.camera.set_external_exposure_mode()
        self.camera.start_live(self.buffer_frames)
        self._attach()

    # ------------------------------------------------------------------ #
    # Watching expt's Go.
    # ------------------------------------------------------------------ #

    def _attach(self) -> None:
        # Re-running the notebook cell makes a new one: retire the old one watching
        # the same key, so a run's frames are not collected twice.
        watchers = self.expt.__dict__.setdefault("_pycaf_cameras", {})
        if (previous := watchers.get(self.key)) is not None:
            previous.close()
        self._original_go = self.expt._go
        self.expt._go = self._go
        watchers[self.key] = self

    def _detach(self) -> None:
        if self._original_go is None:
            return
        watchers = self.expt.__dict__.get("_pycaf_cameras", {})
        if self.expt.__dict__.get("_go") == self._go:
            if self._original_go == getattr(type(self.expt), "_go", None).__get__(
                self.expt
            ):
                # The class's own method: drop the instance attribute altogether.
                del self.expt.__dict__["_go"]
            else:
                self.expt._go = self._original_go
        else:
            # Another camera wrapped ours since: skip ours inside its chain.
            for watcher in watchers.values():
                if watcher._original_go == self._go:
                    watcher._original_go = self._original_go
        if watchers.get(self.key) is self:
            del watchers[self.key]
        self._original_go = None

    def _go(self, *args: Any, **kwargs: Any) -> Any:
        try:
            # Another user of a shared camera may have taken it out of live mode.
            self.camera.start_live(self.buffer_frames)
            mark = self.camera.frames_acquired()
            folders = self._day_folders()
            before = zips.snapshot(folders, self.element)
        except BaseException as error:
            self._reconnect_if_interrupted(error)
            raise
        result = self._original_go(*args, **kwargs)
        try:
            settle_frame_counts(
                {self.key: self.camera},
                self.frame_settle_s,
                self.frame_timeout_s,
                self.frame_poll_s,
            )
            if self.camera.frames_acquired() < mark:
                raise RuntimeError(
                    f"{self.key} camera's acquisition restarted during the run; "
                    "its frames are lost"
                )
            frames = np.asarray(self.camera.read_frames_from(mark), dtype=float)
        except BaseException as error:
            self._reconnect_if_interrupted(error)
            raise
        new = sorted(zips.snapshot(folders, self.element) - before)
        self._runs.append(_Run(frames, new[0] if len(new) == 1 else None))
        return result

    def _day_folders(self) -> list[Path]:
        today = date.today()
        return [
            zips.day_folder(self.data_root, today),
            zips.day_folder(self.data_root, today + timedelta(days=1)),
        ]

    # ------------------------------------------------------------------ #
    # Saving.
    # ------------------------------------------------------------------ #

    def store_images(self, save: bool = True) -> list[np.ndarray]:
        """Each run's frames since the last call, saved beside MOTMaster's zip.

        Args:
            save: Write the zips; ``False`` only returns the frames.

        Returns:
            One float ``(n, H, W)`` stack per MOTMaster run, in run order, as
            ``readFromZip`` gives them. The zips written are in :attr:`last_zips`
            (``None`` for a run MOTMaster did not save).
        """
        runs, self._runs = self._runs, []
        if not runs:
            warnings.warn(
                f"No MOTMaster runs since the last store_images(); nothing from {self.key}",
                stacklevel=2,
            )
            self.last_zips = []
            return []
        counts = sorted({len(run.frames) for run in runs})
        if len(counts) > 1:
            warnings.warn(
                f"{self.key}: runs had different frame counts {counts}", stacklevel=2
            )
        settings = self.camera.settings() if save else {}
        saved: list[Path | None] = []
        for run in runs:
            if not save or run.zip is None:
                saved.append(None)
                continue
            saved.append(
                zips.write_sibling(
                    run.zip, self.element, self.key, run.frames, settings
                )
            )
        self.last_zips = saved
        if save:
            self._report(saved, counts)
        return [run.frames for run in runs]

    def _report(self, saved: list[Path | None], counts: list[int]) -> None:
        written = [path for path in saved if path is not None]
        unsaved = len(saved) - len(written)
        frames = "/".join(str(count) for count in counts)
        if written:
            numbers = [
                int(re.search(r"_(\d+)\.zip$", path.name)[1]) for path in written
            ]
            span = (
                f"{written[0].name}"
                if len(written) == 1
                else f"{written[0].name} ... _{numbers[-1]:03d}"
            )
            print(
                f"{self.key}: saved {len(written)} runs ({frames} frames each) to {span}"
            )
        if unsaved:
            warnings.warn(
                f"{self.key}: {unsaved} of {len(saved)} runs had no new MOTMaster zip "
                "to save beside (is saving on in MOTMaster?); their frames are only "
                "returned",
                stacklevel=3,
            )

    # ------------------------------------------------------------------ #
    # Connection.
    # ------------------------------------------------------------------ #

    def _connect(self) -> Any:
        from pytweezer.servers.device_client import get_device

        return get_device(self._camera_name)

    def _reconnect_if_interrupted(self, error: BaseException) -> None:
        # sipyco reads a call's reply after sending it; an interrupt in between
        # leaves the reply queued, and every later call gets the one before's.
        if isinstance(error, Exception) or not self._owns_camera:
            return
        try:
            self.camera.close_rpc()
        except Exception:
            logger.warning("Could not close a stale camera client", exc_info=True)
        self.camera = self._connect()

    def close(self) -> None:
        """Stop watching ``expt`` and close the camera client this opened.

        Runs not yet stored are dropped. The camera stays live.
        """
        self._detach()
        if self._owns_camera:
            self.camera.close_rpc()
            self._owns_camera = False
