"""Screenshot every applet with synthetic data, for comparing renders before and after a change.

Not a test: run it, then look at the PNGs.

    QT_QPA_PLATFORM=offscreen uv run python tests/applet_screenshots.py <out_dir>

Each applet is built with no hubs, Properties server or experiment manager
behind it, is fed fixed synthetic messages through the stream clients it
subscribes with, and is saved as ``<out_dir>/<Applet>.png`` at a fixed window
size. Everything is seeded, so two runs of the same code give the same images.
"""

import importlib
import os
import sys
from pathlib import Path
from typing import ClassVar

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["PYTWEEZER_LOG_PUBLISH"] = "0"

import numpy as np
from PyQt6 import QtCore, QtTest, QtWidgets

from pytweezer import servers
from pytweezer.GUI.theme import apply_theme

WINDOW_SIZE = (1000, 700)
SEED = 20260101
FRAME_SHAPE = (96, 128)
RESOLUTION = [1e-6, 1e-6]
OFFSET = [10, 20]

VIEWERS = "pytweezer.GUI.viewers."


class FakeProps:
    """Properties with no server: ``seed`` answers first, then each default."""

    seed: ClassVar[dict] = {}

    def __init__(self, name=""):
        self.name = name
        self.values = dict(self.seed)

    def get(self, key, default=None):
        return self.values.setdefault(key, default)

    def set(self, key, value):
        self.values[key] = value

    def delete(self, key):
        self.values.pop(key, None)

    def changes(self, includeparent=True):
        return []


class FakeStream:
    """A stream client that hands over what :func:`publish` puts on channels it subscribed to."""

    live: ClassVar[list["FakeStream"]] = []

    def __init__(self, name="", **kwargs):
        self.queue = []
        self.channels = set()
        FakeStream.live.append(self)

    def subscribe(self, channel):
        self.channels.update([channel] if isinstance(channel, str) else channel)

    def unsubscribe(self, channel=None):
        self.channels.clear()

    def has_new_data(self, timeout_ms=1):
        return bool(self.queue)

    def recv(self, **kwargs):
        return self.queue.pop(0)


class FakeFeed(QtCore.QObject):
    """An experiment feed that never hears from a manager."""

    queue_changed = QtCore.pyqtSignal(dict)
    point_received = QtCore.pyqtSignal(dict)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(parent)
        self.connected = False

    def snapshot(self):
        return {}

    def held_points(self):
        return {}

    def points(self, rid):
        return []

    def close(self):
        pass


def fake_hubs(trap_sites):
    """Replace the hub, Properties and manager clients wherever pytweezer imported them."""
    replacements = {
        "Properties": FakeProps,
        "ImageClient": FakeStream,
        "DataClient": FakeStream,
        "ExperimentFeed": FakeFeed,
        "get_saved_parameters": lambda **kwargs: {
            "trap_sites": trap_sites,
            "trap_window_size": 8,
        },
    }
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("pytweezer."):
            continue
        for name, fake in replacements.items():
            if hasattr(module, name):
                setattr(module, name, fake)


def publish(message):
    """Deliver ``message`` (``(channel, ...)``) to every stream subscribed to its channel."""
    for stream in FakeStream.live:
        if message[0] in stream.channels:
            stream.queue.append(message)


def gaussian_frame(shape=FRAME_SHAPE, centre=(0.45, 0.55), sigma=(14.0, 10.0)):
    rows, cols = np.indices(shape, dtype=float)
    y0, x0 = centre[0] * shape[0], centre[1] * shape[1]
    peak = np.exp(
        -0.5 * (((rows - y0) / sigma[0]) ** 2 + ((cols - x0) / sigma[1]) ** 2)
    )
    noise = np.random.default_rng(SEED).normal(0, 4, shape)
    return 100.0 + 800.0 * peak + noise


#: ROI placement in the saved ``pos``/``size`` keys every ROI reads, over the frame.
ROI_SEED = {"maskstreams": ["mask"], "pos": [5e-5, 3e-5], "size": [4e-5, 3e-5]}
FRAME_HEAD = {"_imgresolution": RESOLUTION, "_offset": OFFSET, "index": 0}


def mask_message():
    """A ring around the cloud; it also sets the mask's extent, which the view's range follows."""
    rows, cols = np.indices(FRAME_SHAPE, dtype=float)
    radius = np.hypot(rows - 0.45 * FRAME_SHAPE[0], cols - 0.55 * FRAME_SHAPE[1])
    ring = (np.abs(radius - 30) < 1.0).astype(float)
    return ("mask", dict(FRAME_HEAD), ring)


def image_messages(channel="cam"):
    return [(channel, dict(FRAME_HEAD), gaussian_frame()), mask_message()]


def camera_messages():
    """A frame on the stream of the camera the applet opens on: the first one in CONFIG."""
    from pytweezer.drivers.camera_catalogue import camera_conf, camera_names

    stream = camera_conf(camera_names()[0])["stream_name"]
    return image_messages(stream)


def projection_messages():
    frame = gaussian_frame()
    x = (OFFSET[0] + np.arange(FRAME_SHAPE[1]) + 0.5) * RESOLUTION[0]
    y = (OFFSET[1] + np.arange(FRAME_SHAPE[0]) + 0.5) * RESOLUTION[1]
    messages = list(image_messages())
    for axis, coords, sums in (
        ("_x", x, frame.sum(axis=0)),
        ("_y", y, frame.sum(axis=1)),
    ):
        smooth = np.convolve(sums, np.ones(9) / 9, mode="same")
        head = {"center": 0.0, "sigma": 1.0, "amplitude": 1.0, "offset": 0.0}
        head["success"] = True
        messages.append(("cloud" + axis, head, np.vstack([coords, sums, smooth])))
    return messages


def slm_messages():
    rows, cols = np.indices((64, 64), dtype=float)
    spots = np.zeros((64, 64))
    for row in range(8, 64, 16):
        for col in range(8, 64, 16):
            spots += np.exp(-((rows - row) ** 2 + (cols - col) ** 2) / 8.0)
    phase = (rows * 3.0 + cols * 5.0 + 40 * np.sin(cols / 6.0)) % 256.0
    head = {"_imgresolution": [1e-6, 1e-6], "_offset": [0, 0]}
    return [
        ("slm_target", dict(head), (spots > 0.5).astype(float)),
        ("slm_simulated", dict(head), spots),
        ("slm_top_down", {}, spots[::2, :] ** 0.5),
        ("slm_phasemask", {}, phase),
    ]


def live_plot_messages():
    rng = np.random.default_rng(SEED)
    x = np.arange(40.0)
    messages = [("analysis", {}, np.vstack([x, np.exp(-(((x - 20) / 6) ** 2) / 2)]))]
    for shot in range(40):
        messages.append(
            ("counts", {"atom_number": 100 + 20 * np.sin(shot / 5) + rng.normal(0, 3)})
        )
    return messages


def scalar_messages():
    rng = np.random.default_rng(SEED)
    messages = []
    for shot in range(60):
        for stream, level in (("first", 100.0), ("second", 60.0)):
            value = level + 10 * np.sin(shot / 7 + level) + rng.normal(0, 2)
            messages.append((stream, {"atom_number": value}))
    return messages


def loading_message():
    rng = np.random.default_rng(SEED)
    rows, cols = np.meshgrid(np.arange(4), np.arange(5), indexing="ij")
    n_sites = rows.size
    loaded = rng.random((300, n_sites)) < np.linspace(0.4, 0.7, n_sites)
    photons = rng.poisson(5 + 40 * loaded).astype(float)
    head = {
        "site_rows": rows.ravel().tolist(),
        "site_cols": cols.ravel().tolist(),
        "site_loading": loaded.mean(axis=0).tolist(),
        "threshold": 25.0,
        "loading": float(loaded.mean()),
        "loading_spread": float(loaded.mean(axis=0).std()),
        "fidelity": 0.995,
        "background": [5.0, 5.0, 0.4],
        "signal": [45.0, 60.0, 0.6],
        "rid": 7,
        "shots": len(photons),
        "sites": n_sites,
    }
    return [("loading", head, photons)]


SITES = np.array(
    [[48 + 20 * row, 64 + 20 * col] for row in (-1, 0, 1) for col in (-1, 0, 1)],
    dtype=float,
)

#: (class name, module, Properties seed, messages). The seed uses the keys the
#: applets save under, so it holds for any implementation of them.
APPLETS = [
    (
        "ImageDisplay",
        "image_monitor",
        ROI_SEED | {"imagestreams": ["cam"]},
        image_messages,
    ),
    (
        "ImageProjectionDisplay",
        "image_plot_monitor",
        ROI_SEED | {"imagestreams": ["cam"], "projectionstreams": ["cloud"]},
        projection_messages,
    ),
    (
        "TweezerDisplay",
        "tweezer_monitor",
        ROI_SEED | {"imagestreams": ["cam"], "show_site_labels": True},
        image_messages,
    ),
    (
        "CameraApplet",
        "camera_applet",
        ROI_SEED | {"camera": ""},
        camera_messages,
    ),
    ("SlmMonitor", "slm_monitor", {}, slm_messages),
    (
        "LivePlot",
        "live_plot",
        {
            "plots": ["plot0", "plot1"],
            "plot0/datastreams": ["analysis"],
            "plot1/datastreams": ["counts"],
            "plot1/fields": ["atom_number"],
        },
        live_plot_messages,
    ),
    (
        "StackedPlotter",
        "stacked_plotter",
        {
            "plots": ["plot0", "plot1"],
            "plot0/datastreams": ["analysis"],
            "plot1/datastreams": ["counts"],
            "plot1/fields": ["atom_number"],
        },
        live_plot_messages,
    ),
    (
        "ScalarHistory",
        "scalar_history",
        {"datastreams": ["first", "second"], "field": "atom_number"},
        scalar_messages,
    ),
    (
        "TweezerLoadingViewer",
        "tweezer_loading",
        {"datastreams": ["loading"], "show_site_labels": True},
        loading_message,
    ),
]


def screenshot(class_name, module_name, seed, messages, out_dir):
    module = importlib.import_module(VIEWERS + module_name)
    fake_hubs(SITES)
    FakeStream.live.clear()
    FakeProps.seed = seed
    widget = getattr(module, class_name)(class_name)
    widget.resize(*WINDOW_SIZE)
    widget.show()
    QtWidgets.QApplication.processEvents()
    for message in messages():
        publish(message)
    for _ in range(3):
        widget.poll()
        QtTest.QTest.qWait(100)
    path = out_dir / f"{class_name}.png"
    if not widget.grab().save(str(path)):
        raise RuntimeError(f"could not write {path}")
    widget.hide()
    widget.deleteLater()
    return path


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app)
    for module_name in {entry[1] for entry in APPLETS}:
        importlib.import_module(VIEWERS + module_name)
    assert servers.Properties is not FakeProps
    for class_name, module_name, seed, messages in APPLETS:
        print(screenshot(class_name, module_name, seed, messages, out_dir))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
