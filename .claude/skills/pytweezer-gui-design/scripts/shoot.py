"""Screenshot one pytweezer widget, themed, for design review.

    poetry run python .claude/skills/pytweezer-gui-design/scripts/shoot.py \\
        pytweezer.GUI.experiments.results:ResultsPanel out.png [--size 1500x950]
        [--wait 1500] [--path DIR] [--crop X,Y,W,H]

TARGET is ``module:callable`` returning a QWidget, called with no arguments.
For a panel that needs fakes or data (a fake client, a temporary data root),
write a tiny factory function in a scratch module and pass its directory with
``--path``. Then Read the PNG and critique it.

Pumps a *local* event loop: ``app.quit()`` would close every window in Qt 6,
and closing a server GUI stops all the servers it started.
"""

import argparse
import importlib
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("out")
    parser.add_argument("--size", default="1500x950")
    parser.add_argument("--wait", type=int, default=1500, help="ms to paint/settle")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--crop", help="X,Y,W,H of the region to keep")
    args = parser.parse_args()
    sys.path[:0] = args.path

    from PyQt6.QtCore import QEventLoop, QRect, QTimer
    from PyQt6.QtWidgets import QApplication

    from pytweezer.GUI.theme import apply_theme

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)

    module_name, _, attr = args.target.partition(":")
    widget = getattr(importlib.import_module(module_name), attr)()
    width, height = (int(v) for v in args.size.split("x"))
    widget.resize(width, height)
    widget.show()

    loop = QEventLoop()
    QTimer.singleShot(args.wait, loop.quit)
    loop.exec()

    image = widget.grab()
    if args.crop:
        image = image.copy(QRect(*(int(v) for v in args.crop.split(","))))
    image.save(args.out)
    print(args.out, flush=True)
    widget.close()
    os._exit(0)  # skip lingering non-daemon ZMQ threads, as bin/gui.py does


if __name__ == "__main__":
    main()
