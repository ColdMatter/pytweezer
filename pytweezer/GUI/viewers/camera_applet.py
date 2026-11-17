"""One camera's live image, with the camera's settings in a sidebar on the left.

Pick the camera in the sidebar. The applet subscribes to the camera's own image
stream (its CONFIG ``stream_name``) and reads its settings over RPC. An edit is
applied on Enter, focus-out or selection; the sidebar then shows the value the
camera took, and that value is stored as a saved parameter
(``<camera key>_<control>``, reason "Camera applet").

While an Experiment Manager task has reserved the camera, the settings are
read-only and refreshed every few seconds from the values the driver keeps,
without touching the hardware. The same holds after start-up until the
manager's state has arrived (a few seconds at most), since a task may already
have the camera. A notebook using the camera is not noticed.
If the manager stops responding while a task holds the camera, the settings
stay read-only until it reports back. If it cannot be reached and no task was
known to hold the camera, the settings stay editable, and the sidebar says it
cannot tell whether a task holds the camera.

Saved ROIs are shared with everyone through Properties, under
``/Camera ROI presets/<camera>/<name>``.

The Image Monitor's context menu (levels, ROIs, *Copy as camera ROI*) is all
still there; its data panel starts hidden.

Properties (besides the Image Monitor's):
    *   camera: (str) CONFIG name of the camera shown.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from PyQt6 import QtCore
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel

from pytweezer.drivers.camera_catalogue import camera_conf, camera_driver, camera_names
from pytweezer.experiment.camera_store import store_camera_settings
from pytweezer.experiment.reservations import canonical_device
from pytweezer.GUI.applet import run_applet
from pytweezer.GUI.background import clear_tracebacks
from pytweezer.GUI.components import set_state
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.GUI.viewers.camera_sidebar import CameraSidebar
from pytweezer.GUI.viewers.image_monitor import CameraROI, ImageDisplay, camera_roi
from pytweezer.logging_utils import get_logger
from pytweezer.servers import Properties, PropertyAttribute
from pytweezer.servers.device_client import get_device

logger = get_logger(__name__)

#: Socket timeout of a camera call; long enough for a snap at the ImagEM's 20 s frame timeout.
CALL_TIMEOUT_S = 30.0
REFRESH_MS = 3000
TEMPERATURE_MS = 5000
PRESET_POLL_MS = 1000
#: How long the manager's state may take to arrive before the hold is called unknown.
FEED_GRACE_MS = 3000
FPS_WINDOW_S = 2.0
#: How sipyco reports a server that is gone, refused, reset or timed out (``OSError``),
#: or closed mid-reply (``EOFError``).
CONNECTION_ERRORS = (OSError, EOFError)

PRESETS_ROOT = "Camera ROI presets"
ROI_NAMES = ("roi_x0", "roi_width", "roi_y0", "roi_height")
UNKNOWN_HOLD = "Can't tell whether a task holds the camera"
STALE_HOLD = "Manager not responding; read-only until it reports back"
EXTERNAL_TRIGGER = "waiting for external trigger"


def error_text(error: BaseException) -> str:
    """One line saying what went wrong."""
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error) or type(error).__name__


def control_values(controls: list[dict]) -> dict[str, Any]:
    """``{name: value}`` of a control list."""
    return {control["name"]: control["value"] for control in controls}


def model_name(driver: type) -> str:
    """The camera model as the driver class names it: ``ImagEMX2Camera`` gives ``ImagEMX2``."""
    return driver.__name__.removesuffix("Camera") or driver.__name__


class CameraLink(QtCore.QObject):
    """Calls to one camera's device server, made one at a time on a worker thread.

    Only the worker touches the sipyco client, since a client is not
    thread-safe, so calls run in the order they were queued. Each outcome, the
    return value or the exception raised, goes to its callback on the GUI
    thread; after :meth:`close`, outcomes are dropped. A connection error
    (``OSError``, which includes a timeout, or ``EOFError``) closes the client
    and the next call reconnects.

    Args:
        name: CONFIG name of the camera.
    """

    _finished = QtCore.pyqtSignal(object, object)

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._calls: queue.Queue = queue.Queue()
        self._closed = False
        self._finished.connect(self._deliver)
        self._thread = threading.Thread(
            target=self._run, name=f"camera link {name}", daemon=True
        )
        self._thread.start()

    def call(self, method: str, *args: Any, callback: Callable[[Any], None]) -> None:
        """Queue the camera's ``method(*args)``; ``callback(outcome)`` runs on the GUI thread."""
        self.submit(lambda device: getattr(device, method)(*args), callback=callback)

    def submit(
        self, function: Callable[[Any], Any], callback: Callable[[Any], None]
    ) -> None:
        """Queue ``function(client)``, for work that must run in one go on the worker."""
        self._calls.put((function, callback))

    def close(self) -> None:
        """Drop every outcome from now on; calls already queued still run, then the thread ends."""
        self._closed = True
        self._calls.put(None)

    def is_running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        device = None
        while (item := self._calls.get()) is not None:
            function, callback = item
            try:
                if device is None:
                    device = get_device(self.name, timeout=CALL_TIMEOUT_S)
                outcome = function(device)
            except Exception as error:
                outcome = error
                if isinstance(error, CONNECTION_ERRORS):
                    _close_client(device)
                    device = None
            self._finished.emit(callback, outcome)
        _close_client(device)

    def _deliver(self, callback: Callable[[Any], None], outcome: Any) -> None:
        try:
            if not self._closed:
                callback(outcome)
        finally:
            if isinstance(outcome, BaseException):
                clear_tracebacks(outcome)


def _close_client(device: Any) -> None:
    if device is None:
        return
    try:
        device.close_rpc()
    except Exception:
        pass


class CameraApplet(ImageDisplay):
    """The Image Monitor with a camera's settings, Live and Snap in a sidebar on its left."""

    _camera = PropertyAttribute("camera", "")
    _show_sidebar = PropertyAttribute("show_sidebar", False)

    def init_gui(self):
        self._camera_name = ""
        self._model = ""
        self._link: CameraLink | None = None
        self._retired_links: list[CameraLink] = []
        self._controls: list[dict] = []
        self._loaded = False
        self._unavailable = ""
        self._holder: tuple[str, Any] | None = None
        self._live = False
        self._in_flight: set[str] = set()
        self._applying: dict[int, set[str]] = {}
        self._apply_serial = 0
        self._live_serial = 0
        self._releasing = False
        self._read_due = False
        self._presets: dict[str, dict[str, int]] | None = None
        self._last_poll_error = ""
        self._fps_samples: deque = deque()
        self._fps: float | None = None
        self._frame_interval = 0.0
        self._last_frame_at = 0.0
        self._frame_max: int | None = None
        self._cursor: tuple[float, float] | None = None
        super().init_gui()

        self.camera_sidebar = CameraSidebar()
        self.panel_layout.insertWidget(0, self.camera_sidebar)
        self.status_bar = QFrame()
        self.status_bar.setObjectName("CameraStatusBar")
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 4, 10, 4)
        self.readout = QLabel("No frame yet")
        self.readout.setObjectName("CameraReadout")
        self.message = QLabel()
        self.message.setProperty("role", "regionHint")
        self.message.setWordWrap(True)
        status_layout.addWidget(self.readout)
        status_layout.addStretch(1)
        status_layout.addWidget(self.message)
        self.image_layout.insertWidget(1, self.status_bar)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)

        sidebar = self.camera_sidebar
        sidebar.camera_chosen.connect(self._choose)
        sidebar.apply_requested.connect(self._apply)
        sidebar.live_toggled.connect(self._toggle_live)
        sidebar.snap_requested.connect(self._snap)
        sidebar.roi_from_drawn_requested.connect(self._roi_from_drawn)
        sidebar.full_sensor_requested.connect(self._full_sensor)
        sidebar.preset_chosen.connect(self._preset_chosen)
        sidebar.preset_save_requested.connect(self._preset_save)
        sidebar.preset_delete_requested.connect(self._preset_delete)

        self._preset_props = Properties(PRESETS_ROOT)
        self.feed = ExperimentFeed(parent=self)
        self._feed_connected = self.feed.connected
        # Until the manager's state arrives, a running task may hold the camera:
        # only the kept values are read and nothing is applied.
        self._checking = not self._feed_connected
        self.feed.queue_changed.connect(self._queue_changed)
        self.feed.connection_changed.connect(self._feed_connection)
        QtCore.QTimer.singleShot(FEED_GRACE_MS, self._end_check)

        self._timers = []
        for interval, slot in (
            (REFRESH_MS, self._refresh_tick),
            (TEMPERATURE_MS, self._temperature_tick),
            (PRESET_POLL_MS, self._presets_tick),
        ):
            timer = QtCore.QTimer(self)
            timer.timeout.connect(slot)
            timer.start(interval)
            self._timers.append(timer)

        names = camera_names()
        sidebar.set_cameras(names, self._inactive(names))
        sidebar.set_current_camera(self._camera or None)
        if (chosen := sidebar.current_camera()) is not None:
            self._choose(chosen)

    def sizeHint(self):
        return QtCore.QSize(1120, 720)

    @staticmethod
    def _inactive(names: list[str]) -> dict[str, str]:
        return {
            name: "Inactive in CONFIG"
            for name in names
            if not camera_conf(name).get("active", True)
        }

    # ---- choosing a camera ---------------------------------------------

    def _choose(self, name: str) -> None:
        self._camera = name
        self.camera_sidebar.set_current_camera(name)
        self._retire_link()
        self._controls = []
        self._loaded = False
        self._live = False
        self._in_flight.clear()
        self._applying.clear()
        self._releasing = False
        self._read_due = False
        self._presets = None
        self._reset_fps()
        self._show_message("")
        self.image_head = None
        self._frame_max = None
        self.image_item.clear()
        self._update_readout()
        self.camera_sidebar.set_temperature(None)
        try:
            self._camera_name = canonical_device(name)
            conf = camera_conf(name)
            self._model = model_name(camera_driver(name))
        except Exception as error:
            logger.warning("Camera %r: %s", name, error_text(error))
            self._camera_name, self._model = name, ""
            self._imagestreams = ["--None--"]
            self.update_subscriptions()
            self.camera_sidebar.set_controls([])
            self._set_unavailable("not in CONFIG")
            return
        stream = conf.get("stream_name")
        self._imagestreams = [stream] if stream else ["--None--"]
        self.update_subscriptions()
        self.camera_sidebar.set_controls([], self._model)
        self._show_presets()
        self._holder = self._holder_in(self.feed.snapshot())
        if not conf.get("active", True):
            self._set_unavailable("inactive in CONFIG")
            return
        self._unavailable = ""
        self._link = CameraLink(self._camera_name)
        self._load()
        self._temperature_tick()
        self._render_state()

    def _retire_link(self) -> None:
        if self._link is not None:
            self._link.close()
            self._retired_links.append(self._link)
            self._link = None

    def _prune_links(self) -> None:
        # A closed link is dropped here, on the GUI thread, once its thread has
        # ended, so its QObject is never deleted from the worker.
        self._retired_links = [
            link for link in self._retired_links if link.is_running()
        ]

    # ---- calls to the camera -------------------------------------------

    def _request(
        self,
        kind: str | None,
        method: str,
        *args: Any,
        callback: Callable[[Any], None],
    ) -> bool:
        """Queue ``method`` unless a call of the same ``kind`` is still in flight.

        ``kind=None`` always queues.
        """
        if self._link is None or (kind is not None and kind in self._in_flight):
            return False
        if kind is not None:
            self._in_flight.add(kind)

        def done(outcome):
            self._in_flight.discard(kind)
            callback(outcome)

        self._link.call(method, *args, callback=done)
        return True

    def _connection_failed(self, outcome: Any) -> bool:
        """Whether ``outcome`` is a lost connection, marking the camera unavailable if so.

        A timeout while a task holds the camera (or may) is the server busy with
        that task, so the camera is left as it is.
        """
        if not isinstance(outcome, CONNECTION_ERRORS):
            return False
        if isinstance(outcome, TimeoutError):
            if not self._hold_locks():
                self._set_unavailable("server not responding")
        else:
            self._set_unavailable("server not running")
        return True

    def _set_unavailable(self, reason: str) -> None:
        self._unavailable = reason
        self._loaded = False
        self._controls = []
        self._live = False
        self._applying.clear()
        self._releasing = False
        self.camera_sidebar.set_controls([], self._model)
        self.camera_sidebar.set_temperature(None)
        self._reset_fps()
        self._render_state()

    def _load(self) -> None:
        """Read every setting and whether the camera is live; only the kept values while it may be held."""
        locked = self._hold_locks()
        method = "controls" if locked else "read_controls"
        self._request("load", method, callback=self._loaded_controls)
        if not locked:
            self._request_live()

    def _loaded_controls(self, outcome: Any) -> None:
        if self._connection_failed(outcome):
            return
        if isinstance(outcome, Exception):
            logger.warning(
                "%s: reading its settings failed: %s",
                self._camera_name,
                error_text(outcome),
            )
            self._set_unavailable(f"settings unreadable: {error_text(outcome)}")
            return
        self._unavailable = ""
        self._loaded = True
        self._controls = outcome
        self.camera_sidebar.set_controls(outcome, self._model)
        self._refresh_busy()
        self._render_state()
        if self._read_due and not self._hold_locks():
            self._released()

    def _show(self, controls: list[dict]) -> None:
        self._controls = controls
        self.camera_sidebar.show_values(controls)
        self._render_state()

    def _refresh_tick(self) -> None:
        self._prune_links()
        if self._link is None:
            return
        if self._unavailable or not self._loaded:
            self._load()
            return
        serial = self._apply_serial
        self._request(
            "poll", "controls", callback=lambda outcome: self._polled(outcome, serial)
        )
        if not self._hold_locks():
            self._request_live()

    def _polled(self, outcome: Any, apply_serial: int) -> None:
        if self._connection_failed(outcome):
            return
        if isinstance(outcome, Exception):
            self._log_poll_error(outcome)
            return
        if apply_serial == self._apply_serial and not self._applying:
            self._show(outcome)

    def _log_poll_error(self, error: Exception) -> None:
        text = error_text(error)
        if text != self._last_poll_error:
            logger.warning("%s: refreshing failed: %s", self._camera_name, text)
            self._last_poll_error = text

    # ---- holds -----------------------------------------------------------

    def _holder_in(self, snapshot: dict) -> tuple[str, Any] | None:
        for task in snapshot.get("running") or ():
            if self._camera_name in (task.get("devices") or ()):
                return task.get("class_name") or "an experiment", task.get("rid")
        return None

    def _held(self) -> bool:
        """Whether a task holds the camera, as last heard: a hold outlasts a silent manager."""
        return self._holder is not None

    def _hold_locks(self) -> bool:
        """Whether the camera must be left alone: a task holds it, or may and the manager has not said yet."""
        return self._checking or self._held()

    def _hold_unknown(self) -> bool:
        return not self._feed_connected and not self._checking and not self._held()

    def _queue_changed(self, snapshot: dict) -> None:
        was_held = self._held()
        self._holder = self._holder_in(snapshot)
        if self._checking:
            self._end_check()
        elif was_held and not self._held():
            self._released()
        self._render_state()

    def _feed_connection(self, connected: bool) -> None:
        self._feed_connected = connected
        if connected:
            was_held = self._held()
            self._holder = self._holder_in(self.feed.snapshot())
            if self._checking:
                self._end_check()
            elif was_held and not self._held():
                self._released()
        self._render_state()

    def _end_check(self) -> None:
        """The manager has said whether a task holds the camera, or has not in time."""
        if not self._checking:
            return
        self._checking = False
        if not self._held():
            self._released()
        self._render_state()

    def _released(self) -> None:
        """Read the camera once after a task lets it go; the fields stay locked until then.

        Before the first settings have arrived, the read is made when they do.
        """
        if self._link is None:
            return
        if not self._loaded:
            self._read_due = True
            return
        self._read_due = False
        self._releasing = True
        self._refresh_busy()
        self._request("release", "read_controls", callback=self._release_read)
        self._request_live()

    def _release_read(self, outcome: Any) -> None:
        self._releasing = False
        self._refresh_busy()
        self._reread(outcome)

    def _refresh_busy(self) -> None:
        busy = set().union(*self._applying.values())
        if self._releasing:
            busy |= {control["name"] for control in self._controls}
        self.camera_sidebar.set_busy(busy)

    def _render_state(self) -> None:
        sidebar = self.camera_sidebar
        if self._unavailable:
            sidebar.set_state("unavailable", self._unavailable)
        elif self._held():
            class_name, rid = self._holder
            detail = f"{class_name} (rid {rid})"
            if not self._feed_connected:
                detail += f"\n{STALE_HOLD}"
            sidebar.set_state("held", detail)
        elif self._checking:
            sidebar.set_state("checking")
        elif self._live:
            notes = []
            if self._control_value("trigger_source") not in (None, "int"):
                notes.append(EXTERNAL_TRIGGER)
            if self._hold_unknown():
                notes.append(UNKNOWN_HOLD)
            sidebar.set_state("live", "\n".join(notes))
        elif self._hold_unknown():
            sidebar.set_state("unknown_hold", UNKNOWN_HOLD)
        elif not self._loaded and self._link is not None:
            sidebar.set_state("idle", "Reading the camera's settings…")
        else:
            sidebar.set_state("idle")

    def _control_value(self, name: str) -> Any:
        return control_values(self._controls).get(name)

    # ---- edits -----------------------------------------------------------

    def _apply(self, values: dict) -> None:
        if self._link is None or not self._loaded or self._hold_locks():
            # Disabling a field mid-edit commits it; put the camera's values back.
            self.camera_sidebar.show_values(self._controls)
            return
        self._apply_serial += 1
        serial = self._apply_serial
        self._applying[serial] = set(values)
        self._refresh_busy()
        camera = self._camera_name

        def apply_and_store(device):
            # One worker item, so the store happens even if this link is closed
            # (camera switched, window shut) before the outcome arrives.
            controls = device.apply_controls(dict(values))
            taken = control_values(controls)
            stored = {name: taken[name] for name in values if name in taken}
            try:
                store_camera_settings(camera, stored, "Camera applet")
            except Exception as error:
                logger.warning(
                    "%s: %s applied but not stored: %s",
                    camera,
                    sorted(stored),
                    error_text(error),
                )
                clear_tracebacks(error)
                return controls, stored, error
            return controls, stored, None

        self._link.submit(
            apply_and_store,
            callback=lambda outcome: self._applied(outcome, serial, values),
        )

    def _applied(self, outcome: Any, serial: int, values: dict) -> None:
        self._applying.pop(serial, None)
        self._refresh_busy()
        if self._connection_failed(outcome):
            self.camera_sidebar.show_values(self._controls)
            return
        if isinstance(outcome, Exception):
            self.camera_sidebar.show_error(next(iter(values)), error_text(outcome))
            if not self._hold_locks():
                self._request(None, "read_controls", callback=self._reread)
            return
        controls, stored, store_error = outcome
        self._show(controls)
        if store_error is not None:
            for name in stored:
                self.camera_sidebar.show_error(
                    name, f"Applied, not stored: {error_text(store_error)}"
                )

    def _reread(self, outcome: Any) -> None:
        if self._connection_failed(outcome):
            return
        if isinstance(outcome, Exception):
            self._log_poll_error(outcome)
            return
        self._show(outcome)

    def _toggle_live(self, start: bool) -> None:
        if self._hold_locks() or "toggle" in self._in_flight:
            return
        self._live_serial += 1
        self._request(
            "toggle",
            "start_live" if start else "stop_live",
            callback=lambda outcome: self._live_toggled(outcome, start),
        )

    def _live_toggled(self, outcome: Any, start: bool) -> None:
        self._live_serial += 1
        if self._connection_failed(outcome):
            return
        if isinstance(outcome, Exception):
            action = "Starting" if start else "Stopping"
            self._show_message(f"{action} live failed: {error_text(outcome)}")
            self._request_live()
            return
        self._show_message("")
        self._set_live(start)

    def _request_live(self) -> None:
        serial = self._live_serial
        self._request(
            "is_live",
            "is_live",
            callback=lambda outcome: self._live_read(outcome, serial),
        )

    def _live_read(self, outcome: Any, serial: int) -> None:
        if self._connection_failed(outcome) or isinstance(outcome, Exception):
            return
        if serial == self._live_serial and "toggle" not in self._in_flight:
            self._set_live(bool(outcome))

    def _set_live(self, live: bool) -> None:
        if live != self._live:
            self._live = live
            self._reset_fps()
        self._render_state()

    def _snap(self) -> None:
        if self._hold_locks() or self._live or "snap" in self._in_flight:
            return
        self._request("snap", "snap", True, callback=self._snapped)

    def _snapped(self, outcome: Any) -> None:
        if self._connection_failed(outcome):
            return
        if isinstance(outcome, Exception):
            self._show_message(f"Snap failed: {error_text(outcome)}")
        else:
            self._show_message("")

    def _roi_from_drawn(self) -> None:
        rois = [item for item in self.plot.items if isinstance(item, CameraROI)]
        if not rois:
            self._show_message("From drawn ROI: draw an ROI on the image first.")
        elif self.image_head is None:
            self._show_message(
                "From drawn ROI: no frame yet; the ROI is placed on the last frame."
            )
        elif (roi := camera_roi(rois[0], self.image_item, self.image_head)) is None:
            self._show_message("From drawn ROI: the drawn ROI is off the frame.")
        else:
            self._show_message("")
            self._apply(dict(zip(ROI_NAMES, roi, strict=True)))

    def _full_sensor(self) -> None:
        maxima = {control["name"]: control.get("maximum") for control in self._controls}
        width, height = maxima.get("roi_width"), maxima.get("roi_height")
        if width is None or height is None:
            return
        self._apply(
            {
                "roi_x0": 0,
                "roi_width": int(width),
                "roi_y0": 0,
                "roi_height": int(height),
            }
        )

    def _show_message(self, text: str) -> None:
        self.message.setText(text)
        set_state(self.message, "error" if text else "")

    # ---- saved ROIs ------------------------------------------------------

    def _read_presets(self) -> dict[str, dict[str, int]]:
        tree = self._preset_props.get("/" + PRESETS_ROOT, {})
        saved = tree.get(self._camera_name) if isinstance(tree, dict) else None
        presets = {}
        for name, values in (saved if isinstance(saved, dict) else {}).items():
            if isinstance(values, dict) and all(
                isinstance(values.get(key), int | float)
                and not isinstance(values.get(key), bool)
                for key in ROI_NAMES
            ):
                presets[name] = {key: int(values[key]) for key in ROI_NAMES}
        return presets

    def _show_presets(self) -> None:
        presets = self._read_presets()
        if presets != self._presets:
            self._presets = presets
            self.camera_sidebar.set_presets(presets)

    def _presets_tick(self) -> None:
        changed = self._preset_props.changes()
        if "/" in changed or "/" + PRESETS_ROOT in changed:
            self._show_presets()

    def _preset_chosen(self, name: str) -> None:
        values = (self._presets or {}).get(name)
        current = control_values(self._controls)
        if values is None or all(current.get(k) == v for k, v in values.items()):
            self.camera_sidebar.show_values(self._controls)
            return
        self._apply(values)

    def _preset_save(self, name: str) -> None:
        current = control_values(self._controls)
        if not all(key in current for key in ROI_NAMES):
            return
        self._preset_props.set(
            [PRESETS_ROOT, self._camera_name, name],
            {key: int(current[key]) for key in ROI_NAMES},
        )
        self._show_presets()

    def _preset_delete(self, name: str) -> None:
        self._preset_props.delete([PRESETS_ROOT, self._camera_name, name])
        self._show_presets()

    # ---- sensor temperature ----------------------------------------------

    def _temperature_tick(self) -> None:
        if self._link is None or self._unavailable:
            return
        self._request(
            "temperature", "sensor_temperature", callback=self._temperature_read
        )

    def _temperature_read(self, outcome: Any) -> None:
        valid = isinstance(outcome, dict) and all(
            isinstance(outcome.get(key), int | float) for key in ("celsius", "time")
        )
        self.camera_sidebar.set_temperature(outcome if valid else None)

    # ---- frames, frame rate and the readout under the image ---------------

    def poll(self):
        head = self.image_head
        super().poll()
        if self.image_head is not None and self.image_head is not head:
            self._frame_arrived(self.image_head, self.imgdata)
        elif self._fps is not None and time.monotonic() - self._last_frame_at > max(
            FPS_WINDOW_S, 3 * self._frame_interval
        ):
            self._reset_fps()

    def _frame_arrived(self, head: dict, frame) -> None:
        now = time.monotonic()
        self._last_frame_at = now
        self._frame_max = int(frame.max()) if frame.size else None
        self._sample_fps(head, now)
        self._update_readout()

    def _sample_fps(self, head: dict, now: float) -> None:
        """Frames per second over the last few seconds, by the frames' own clock and numbers.

        Counting frame numbers rather than arrivals keeps the rate right when
        several frames arrive between two polls.
        """
        on_camera_clock = "camera_timestamp" in head
        clock = head.get("camera_timestamp", head.get("timestamp"))
        index = head.get("index")
        if not isinstance(clock, int | float) or not isinstance(index, int):
            return
        samples = self._fps_samples
        if samples and (samples[-1][0] != on_camera_clock or index <= samples[-1][1]):
            samples.clear()
        samples.append((on_camera_clock, index, float(clock), now))
        while len(samples) > 2 and now - samples[0][3] > FPS_WINDOW_S:
            samples.popleft()
        frames = samples[-1][1] - samples[0][1]
        seconds = samples[-1][2] - samples[0][2]
        if frames > 0 and seconds > 0:
            self._frame_interval = seconds / frames
            self._fps = frames / seconds
            self.camera_sidebar.set_fps(self._fps)

    def _reset_fps(self) -> None:
        self._fps_samples.clear()
        self._fps = None
        self._frame_interval = 0.0
        self.camera_sidebar.set_fps(None)

    def _mouse_moved(self, position) -> None:
        if self.plot.sceneBoundingRect().contains(position):
            point = self.image_item.mapFromScene(position)
            self._cursor = (point.x(), point.y())
        else:
            self._cursor = None
        self._update_readout()

    def _update_readout(self) -> None:
        head = self.image_head
        frame = getattr(self, "imgdata", None) if head is not None else None
        if frame is None or frame.ndim != 2:
            text = "No frame yet"
        else:
            rows, cols = frame.shape
            binning = int(head.get("binning", 1))
            parts = []
            if self._cursor is not None:
                col, row = (math.floor(coordinate) for coordinate in self._cursor)
                if 0 <= row < rows and 0 <= col < cols:
                    x = int(head.get("roi_x0", 0)) + col * binning
                    y = int(head.get("roi_y0", 0)) + row * binning
                    parts.append(f"x {x}  y {y}  value {frame[row, col]}")
            if self._frame_max is not None:
                parts.append(f"max {self._frame_max}")
            if "index" in head:
                parts.append(f"frame {head['index']}")
            parts.append(f"{rows}×{cols}, {binning}×{binning}")
            text = "   ·   ".join(parts)
        if text != self.readout.text():
            self.readout.setText(text)

    # ---- teardown ----------------------------------------------------------

    def closeEvent(self, event):
        for timer in self._timers:
            timer.stop()
        self._retire_link()
        self.feed.close()
        super().closeEvent(event)


def main(name):
    run_applet(CameraApplet, default_name=name)


if __name__ == "__main__":
    run_applet(CameraApplet, default_name="Camera")
