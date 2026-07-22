#!/usr/bin/env python
"""Build an applet against fake streams and save a screenshot of it.

An applet normally needs a Propertyhub, a Datahub/Imagehub, and something
publishing on them before it shows anything. This driver replaces all three with
in-process fakes, so any applet script can be built and *looked at* from a plain
shell:

    poetry run python .claude/skills/add-applet/scripts/preview_applet.py \
        pytweezer/GUI/viewers/live_plot.py

It writes a PNG (default ``<skill>/shots/<ClassName>.png``) and prints what the
applet subscribed to and how many messages it consumed — which is how you tell a
window that merely *built* from one that is actually drawing.

The fakes reproduce the parts of the real transport that applets get wrong:

* ``recv()`` returns a **2-tuple** ``(channel, header)`` for a header-only
  message and a **3-tuple** ``(channel, header, array)`` when an array follows,
  exactly as ``GenericClient.recv`` does. Run with ``--scalars`` to feed
  header-only messages; an applet that unpacks three values unconditionally
  raises there.
* messages arrive in bursts, so several can be waiting per poll.
* ``props.get(key, default)`` writes the default back, as the real Properties
  does — that is what puts a knob in the configure dialog.

Options:
    --class NAME        pick one Applet subclass when the module defines several
    --out PATH          where to write the PNG
    --name NAME         applet instance name (default: the class name)
    --prop KEY=JSON     seed a property, repeatable (e.g. --prop "rotated=true")
    --scalars           feed header-only messages instead of arrays
    --ticks N           feed N message bursts before grabbing (default 20)
    --size WxH          resize the window before grabbing

Runs on the native Qt backend so text renders. On a box with no desktop session,
set ``QT_QPA_PLATFORM=offscreen`` — the layout and colours are still right, but
Windows' offscreen font database draws no glyphs, so labels come out blank.
"""

import argparse
import copy
import importlib.util
import inspect
import json
import os
import sys
import time
import zlib

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shots")

# Seeded by --prop and read by every FakeProperties instance, so a seeded value
# reaches sub-widgets (ROIs, sidebars) that build their own connection.
_SEED_PROPS = {}
# Every fake stream client built during construction, so the feeder can find
# them without the applet having to expose them.
_STREAMS = []


class FakeProperties:
    """Dict-backed stand-in for :class:`pytweezer.servers.Properties`.

    No sockets and no threads, so the applet builds instantly and the process
    can exit normally. ``get`` writes its default back like the real one, which
    is what makes a ``PropertyAttribute`` show up in the configure dialog.
    """

    def __init__(self, name, initfromfile=False):
        self.name = name
        self.properties = dict(_SEED_PROPS)
        self.crashed = False

    def get(self, key, defaultvalue=None):
        if key not in self.properties:
            self.properties[key] = defaultvalue
        return copy.deepcopy(self.properties[key])

    def set(self, key, value):
        self.properties[key] = copy.deepcopy(value)

    def delete(self, key):
        self.properties.pop(key, None)

    def changes(self, includeparent=True):
        return set()


class FakeStream:
    """Stand-in for DataClient/ImageClient: records subscriptions, queues messages."""

    kind = "data"

    def __init__(self, name="noname", recvtimeout=1000, subscribe=None, **kwargs):
        self.name = name
        self.subscriptions = []
        self.queue = []
        self.received = 0
        self.sent = 0
        _STREAMS.append(self)
        if subscribe is not None:
            self.subscribe(subscribe)

    def subscribe(self, channel):
        channels = [channel] if isinstance(channel, str) else list(channel)
        for chan in channels:
            # "--None--" is the placeholder an unconfigured stream list carries;
            # the real hub would simply never match it.
            if chan and chan != "--None--" and chan not in self.subscriptions:
                self.subscriptions.append(chan)

    def unsubscribe(self, channel=None):
        if channel is None:
            self.subscriptions = []
        elif channel in self.subscriptions:
            self.subscriptions.remove(channel)

    def has_new_data(self):
        return bool(self.queue)

    def recv(self, flags=0, copy=True, track=False):
        if not self.queue:
            return None
        self.received += 1
        return self.queue.pop(0)

    def send(self, *args, **kwargs):
        # Applets are pure consumers; a send() here means the applet is
        # publishing, which is worth knowing about.
        self.sent += 1

    def push(self, message):
        self.queue.append(message)


class FakeDataClient(FakeStream):
    kind = "data"


class FakeImageClient(FakeStream):
    kind = "image"


def install_fakes():
    """Patch the transport before any GUI module imports it.

    Applet modules do ``from pytweezer.servers import DataClient``, which binds
    the name at import time — so the attributes have to be replaced on the
    package before the applet module is loaded.
    """
    import pytweezer.servers as servers

    servers.Properties = FakeProperties
    servers.DataClient = FakeDataClient
    servers.ImageClient = FakeImageClient
    servers.CommandClient = FakeDataClient


def _channel_phase(channel):
    """A stable per-channel offset in [0, 1), so two streams in one preview don't
    draw identical curves on top of each other.

    ``zlib.crc32`` rather than ``hash`` because str hashing is salted per
    process, and a preview you re-run should look the same.
    """
    return (zlib.crc32(channel.encode()) % 100) / 100.0


def synth_image(tick, channel=""):
    """A drifting gaussian spot on noise — enough structure to see colormap,
    levels and ROI placement at a glance."""
    phase = _channel_phase(channel) * 6.28
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w]
    cx = w / 2 + 18 * np.sin(tick / 6.0 + phase)
    cy = h / 2 + 10 * np.cos(tick / 5.0 + phase)
    spot = 900 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * 14.0**2)))
    noise = np.random.default_rng(tick).normal(60, 12, size=(h, w))
    return np.clip(spot + noise, 0, None).astype(np.uint16)


def synth_trace(tick, channel=""):
    """A 2xN ``[coords, values]`` array — the shape the analyses publish and the
    plotting viewers expect."""
    offset = _channel_phase(channel)
    coords = np.linspace(0, 100, 200)
    centre = 30 + 40 * offset + 12 * np.sin(tick / 4.0)
    values = (60 + 40 * offset) * np.exp(-((coords - centre) ** 2) / (2 * 9.0**2))
    values = values + np.random.default_rng(tick).normal(0, 2.0, size=coords.size)
    return np.vstack([coords, values])


def synth_header(tick, channel="", array=None, physical=False):
    offset = _channel_phase(channel)
    head = {
        "timestamp": time.time(),
        "index": 0,
        "_imgindex": tick % 3,
    }
    if physical:
        # Only some publishers set these; an applet must treat them as optional.
        # They are off by default because a metres-per-pixel scale shrinks the
        # image into the corner of a pixel-ranged view, which looks like a bug
        # in the applet when it is really just a units mismatch.
        head["_offset"] = [0, 0]
        head["_imgresolution"] = [1e-6, 1e-6]
    if array is not None:
        values = array[1] if array.ndim == 2 and array.shape[0] == 2 else array
        head["mean"] = float(np.mean(values))
        head["max"] = float(np.max(values))
    # Scalars an analysis would publish, so header-consuming applets have
    # something plausible to read.
    head["atom_number"] = float(120 + 60 * offset + 25 * np.sin(tick / 3.0))
    head["centre_x"] = float(50 + 12 * np.sin(tick / 4.0 + offset))
    return head


def feed(tick, scalars=False, physical=False):
    """Push one message per subscription onto every fake stream."""
    for stream in _STREAMS:
        for channel in stream.subscriptions:
            if scalars:
                # Header-only: recv() gives a 2-tuple, as the real client does.
                stream.push((channel, synth_header(tick, channel, physical=physical)))
                continue
            if stream.kind == "image":
                array = synth_image(tick, channel)
            else:
                array = synth_trace(tick, channel)
            head = synth_header(tick, channel, array, physical=physical)
            stream.push((channel, head, array))


def load_module(path):
    """Import an applet script by file path, without needing it to be a package."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        sys.exit(f"[preview] no such file: {path}")
    spec = importlib.util.spec_from_file_location("applet_under_preview", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_applet_class(module, wanted=None):
    from pytweezer.GUI.applet import Applet

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Applet) and obj is not Applet and obj.__module__ == module.__name__
    ]
    if wanted:
        for obj in candidates:
            if obj.__name__ == wanted:
                return obj
        sys.exit(f"[preview] no Applet subclass named {wanted}; found {[c.__name__ for c in candidates]}")
    if not candidates:
        sys.exit("[preview] the module defines no Applet subclass")
    if len(candidates) > 1:
        sys.exit(
            f"[preview] several Applet subclasses ({[c.__name__ for c in candidates]}); "
            "pick one with --class"
        )
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", help="path to the applet script")
    parser.add_argument("--class", dest="cls", default=None, help="Applet subclass to build")
    parser.add_argument("--out", default=None, help="PNG path")
    parser.add_argument("--name", default=None, help="applet instance name")
    parser.add_argument("--prop", action="append", default=[], metavar="KEY=JSON",
                        help="seed a property, repeatable")
    parser.add_argument("--scalars", action="store_true",
                        help="feed header-only messages instead of arrays")
    parser.add_argument("--physical", action="store_true",
                        help="include _offset/_imgresolution so coordinates are in metres")
    parser.add_argument("--ticks", type=int, default=20, help="message bursts before grabbing")
    parser.add_argument("--size", default=None, metavar="WxH", help="resize before grabbing")
    parser.add_argument("--show", action="store_true",
                        help="map the window on the real desktop instead of painting off-screen")
    args = parser.parse_args()

    for item in args.prop:
        key, _, raw = item.partition("=")
        try:
            _SEED_PROPS[key] = json.loads(raw)
        except json.JSONDecodeError:
            _SEED_PROPS[key] = raw  # a bare string is the common case

    install_fakes()

    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication
    from pytweezer.GUI.theme import apply_theme

    module = load_module(args.script)
    applet_cls = find_applet_class(module, args.cls)
    name = args.name or applet_cls.__name__

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)

    applet = applet_cls(name)
    if args.size:
        width, height = (int(part) for part in args.size.lower().split("x"))
        applet.resize(width, height)
    else:
        applet.resize(applet.sizeHint())
    if not args.show:
        # Lay out and paint into the backing store without ever mapping a window
        # on the desktop — nothing flashes over whatever the user is doing.
        # Preferred over QT_QPA_PLATFORM=offscreen, which on Windows renders no
        # text glyphs and so produces screenshots with every label blank.
        applet.setAttribute(Qt.WA_DontShowOnScreen, True)
    applet.show()

    # Feed on a timer rather than all at once: an applet that keeps history
    # (a strip chart, a running average) only looks right after several
    # messages, and bursts of one-per-tick match how a real stream arrives.
    state = {"tick": 0}

    def tick():
        feed(state["tick"], scalars=args.scalars, physical=args.physical)
        state["tick"] += 1
        if state["tick"] >= args.ticks:
            timer.stop()
            QTimer.singleShot(400, app.quit)  # let the last messages be polled+painted

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(50)
    app.exec_()

    out = args.out or os.path.join(DEFAULT_SHOTS, f"{applet_cls.__name__}.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    applet.grab().save(out)

    subscribed = sorted({chan for s in _STREAMS for chan in s.subscriptions})
    consumed = sum(s.received for s in _STREAMS)
    pending = sum(len(s.queue) for s in _STREAMS)
    published = sum(s.sent for s in _STREAMS)

    print(f"[preview] {applet_cls.__name__}  ->  {out}")
    print(f"[preview] size        {applet.width()}x{applet.height()}")
    print(f"[preview] subscribed  {subscribed or 'NOTHING - check update_subscriptions()'}")
    print(f"[preview] consumed    {consumed} messages ({pending} left unread)")
    if published:
        print(f"[preview] published   {published} messages — applets should be consumers only")
    sys.stdout.flush()

    # Mirrors run_applet: skip interpreter teardown so a stray non-daemon thread
    # from a real sub-widget cannot hang the process.
    os._exit(0)


if __name__ == "__main__":
    main()
