"""Console entry points for the GUIs.

Imports nothing from ``pytweezer`` at module level: ``server_main`` has to mark
the process as a server session before anything reads the configuration, which
then simulates if this is not the server PC (see ``SERVER_ROLE_ENV`` in
``pytweezer/configuration/config.py``). Every process the GUI starts inherits
the mark, so the whole session simulates together.
"""

import os
import sys


def server_main():
    os.environ["PYTWEEZER_ROLE"] = "server"
    from bin.gui import server_main as run

    run()


def client_main():
    from bin.gui import client_main as run

    run()


if __name__ == "__main__":
    client_main() if sys.argv[1:] == ["client"] else server_main()
