"""The Properties server: owner of the shared properties tree.

Holds the tree in a :class:`~sipyco.sync_struct.Notifier` published as
``"properties"``, so every :class:`~pytweezer.servers.properties.Properties`
client keeps a live mirror of it, and accepts edits over ``pc_rpc`` (target
``"properties"``). The tree is loaded from, and saved back to,
``configuration/properties/properties.json``: shortly after each burst of
edits, and on shutdown. Saving writes a temporary file and renames it over the
old one, so a kill mid-save cannot truncate the file.
"""

import argparse
import json
import os
import signal
import threading
import time

from sipyco.sync_struct import Notifier

from pytweezer.configuration.config import get_config
from pytweezer.configuration.paths import propertyfilename
from pytweezer.logging_utils import get_logger
from pytweezer.servers.property_tree import apply_ops, plan_delete, plan_set
from pytweezer.servers.sync import SyncServer

logger = get_logger("Properties")

SERVER_NAME = "Properties"
NOTIFIER_NAME = "properties"
RPC_TARGET = "properties"
#: A save waits for edits to pause this long, but never longer than SAVE_MAX_S.
SAVE_QUIET_S = 1.0
SAVE_MAX_S = 5.0


class PropertyStore:
    """RPC target applying edits to the published tree.

    Runs on the server's event loop, so it may mutate the notifier directly.
    """

    def __init__(self, notifier: Notifier, on_change):
        self._notifier = notifier
        self._on_change = on_change

    def set(self, keys: list[str], value) -> None:
        """Set ``keys`` to ``value``, which must be JSON-serialisable."""
        json.dumps(value)
        ops = plan_set(self._notifier.raw_view, keys, value)
        if ops:
            apply_ops(self._notifier, ops)
            self._on_change()

    def delete(self, keys: list[str]) -> None:
        ops = plan_delete(self._notifier.raw_view, keys)
        if ops:
            apply_ops(self._notifier, ops)
            self._on_change()

    def ping(self) -> bool:
        return True


class PropertyServer:
    """Serves the tree in ``path`` and saves it back there.

    Args:
        host: Address to bind.
        port: Port for the published tree.
        rpc_port: Port for edits.
        path: The JSON file the tree is loaded from and saved to.
    """

    def __init__(self, host: str, port: int, rpc_port: int, path: str):
        self.path = path
        self.notifier = Notifier(_load(path))
        self._changed = threading.Event()
        self._save_lock = threading.Lock()
        self._last_change = 0.0
        self.store = PropertyStore(self.notifier, self._note_change)
        self.sync = SyncServer(
            {NOTIFIER_NAME: self.notifier},
            host,
            port,
            {RPC_TARGET: self.store},
            rpc_port,
        )
        self._running = True
        self._saver = threading.Thread(target=self._save_loop, daemon=True)
        self._saver.start()

    def _note_change(self) -> None:
        self._last_change = time.monotonic()
        self._changed.set()

    def _save_loop(self) -> None:
        while self._running:
            if not self._changed.wait(0.5):
                continue
            first = time.monotonic()
            while (
                self._running
                and time.monotonic() - self._last_change < SAVE_QUIET_S
                and time.monotonic() - first < SAVE_MAX_S
            ):
                time.sleep(0.05)
            if self._running:
                self.save()

    def save(self) -> None:
        """Write the current tree to :attr:`path` atomically."""
        with self._save_lock:
            self._changed.clear()
            try:
                text = self.sync.call(
                    lambda: json.dumps(self.notifier.raw_view, indent=4)
                ).result(timeout=5)
            except Exception:
                logger.exception("Could not snapshot the properties tree")
                return
            temporary = self.path + ".tmp"
            try:
                with open(temporary, "w") as outfile:
                    outfile.write(text)
                    outfile.flush()
                    os.fsync(outfile.fileno())
                os.replace(temporary, self.path)
            except OSError:
                logger.exception("Could not save %s", self.path)

    def close(self) -> None:
        """Stop serving, saving any unsaved edits first."""
        self._running = False
        self._saver.join(timeout=5)
        if self._changed.is_set():
            self.save()
        self.sync.close()


def _load(path: str) -> dict:
    try:
        with open(path) as inputfile:
            tree = json.load(inputfile)
    except (OSError, ValueError):
        tree = {}
    if not tree and os.path.exists(path):
        logger.warning("%s is unreadable or empty; starting from an empty tree", path)
    return tree


def _terminate_stale_instances(grace_s: float = 1.0) -> None:
    """Stop older property servers on this PC, which would hold the ports."""
    if os.name != "posix":
        return
    this_pid = os.getpid()
    stale = []
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit() or int(pid_str) == this_pid:
            continue
        try:
            with open(f"/proc/{pid_str}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "property_server.py" in cmdline:
            stale.append(int(pid_str))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in stale:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        deadline = time.time() + grace_s
        while stale and time.time() < deadline:
            stale = [pid for pid in stale if _alive(pid)]
            time.sleep(0.1)
        if not stale:
            return


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default=SERVER_NAME)
    args, _unknown = parser.parse_known_args()
    conf = get_config()["Servers"][args.name]

    _terminate_stale_instances()
    server = PropertyServer(
        conf["host"], conf["port"], conf["rpc_port"], propertyfilename
    )
    logger.info(
        "Properties serving on %s:%s (edits on %s)",
        conf["host"],
        conf["port"],
        conf["rpc_port"],
    )

    def _stop(_signo, _frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _stop)
    except (ValueError, OSError):
        pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Properties server shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    main()
