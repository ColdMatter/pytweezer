#!/usr/bin/env python
"""Headless launch + screenshot driver for the pytweezer PyQt6 GUIs.

The GUIs (``pytweezer-server`` / ``pytweezer-client``) are PyQt6 QMainWindows.
This driver builds one of them under the *offscreen* Qt platform, spins the
real event loop briefly so text/layout actually paints, then grabs a PNG of the
whole window and of each tab.

Why offscreen + grab instead of ``pytweezer-server``:
  * No display is needed (works over SSH / in a non-interactive shell).
  * ``pytweezer-server``'s ControlPanel is controllable=True and would
    ``subprocess.Popen(['python3', ...])`` every active hub on construction —
    heavy, and ``python3`` isn't on PATH on Windows. The *client* GUI is
    view-only (probes servers over TCP, spawns nothing), so it is the safe
    default. Pass ``server`` only if you accept those subprocess spawns.

Panels that connect to the (usually unreachable) Propertyhub/Analysis Manager
are wrapped in ``_safe_panel`` upstream, so they degrade to a placeholder label
instead of aborting the window — the screenshot still succeeds offscreen.

Usage (always via Poetry, from the repo root):
    poetry run python .claude/skills/run-pytweezer/driver.py client [outdir]
    poetry run python .claude/skills/run-pytweezer/driver.py server [outdir]

Default outdir is ``.claude/skills/run-pytweezer/shots``. Writes:
    <role>_full.png            whole window, active tab raised
    <role>_tab_<label>.png     one per tab, that dock raised
"""

import os
import sys

# Platform choice (must be decided before any PyQt6 import):
#   * default (unset) -> native "windows" backend. On a machine with an active
#     desktop session (the usual lab PC) this renders text/fonts correctly AND
#     the panels reach the live hubs, so the screenshot shows real device state.
#     The window briefly flashes on the real desktop.
#   * QT_QPA_PLATFORM=offscreen -> truly headless (SSH / no session). Renders
#     layout, colors and status dots, but the offscreen font database on Windows
#     does NOT render normal text glyphs — labels come out blank. Use only when
#     there is no display; prefer the default when a session exists.
# Respect an explicit override; otherwise leave Qt's default (native).

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from bin.gui import build_client_gui, build_server_gui
from pytweezer.GUI.theme import DARK_STYLESHEET


def _paint(app, ms):
    """Spin the real event loop for ``ms`` so widgets lay out and paint text.

    A handful of ``processEvents()`` calls is not enough — text stays unpainted
    until the loop has actually run. A single-shot quit gives a bounded spin.
    """
    QTimer.singleShot(ms, app.quit)
    app.exec()


def main():
    role = sys.argv[1] if len(sys.argv) > 1 else "client"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "shots"
    )
    os.makedirs(outdir, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)

    win = build_server_gui() if role == "server" else build_client_gui()
    win.resize(1400, 900)
    win.show()
    _paint(app, 1200)

    full = os.path.join(outdir, f"{role}_full.png")
    win.grab().save(full)
    saved = [full]

    # Docks are tabified; raise each in turn and grab so every tab is captured.
    for dock in getattr(win, "_docks", []):
        label = dock.windowTitle().replace("dock:", "").strip()
        dock.raise_()
        _paint(app, 400)
        path = os.path.join(outdir, f"{role}_tab_{label.replace(' ', '_')}.png")
        win.grab().save(path)
        saved.append(path)

    # print() before the hard exit; flush so a piped shell still sees it.
    print(f"[driver] role={role} wrote {len(saved)} screenshots to {outdir}")
    for p in saved:
        print("  ", p)
    sys.stdout.flush()

    # bin.gui installs non-daemon ZMQ threads via embedded panels; os._exit
    # skips waiting on them (mirrors bin/gui.py's own _run()).
    os._exit(0)


if __name__ == "__main__":
    main()
