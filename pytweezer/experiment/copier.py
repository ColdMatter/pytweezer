"""Copies finished measurement files from the manager's disk to a second root.

The Experiment Manager writes measurements locally, so a run never depends on
the network, and hands each finished file to a :class:`FileCopier`, which
copies it to ``copy_root`` (the lab network drive) at the same relative path.
Copies run on a thread, one at a time, and a failed copy is retried every
``retry_s`` until it lands. The paths still to copy are saved next to the
data, so a restart carries on where it stopped.
"""

import os
import shutil
import threading
from pathlib import Path

from pydantic import BaseModel

from pytweezer.experiment.queue import ModelStore
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.experiment.copier")

RETRY_S = 60.0


class PendingCopies(BaseModel):
    paths: list[str] = []


class PendingCopyStore(ModelStore):
    model = PendingCopies


class FileCopier:
    """Copies files under ``source_root`` to the same paths under ``target_root``.

    Args:
        source_root: Where the files are written.
        target_root: Where they are copied to.
        retry_s: Seconds to wait after a failed copy before trying again.
    """

    def __init__(
        self, source_root: Path | str, target_root: Path | str, retry_s: float = RETRY_S
    ):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.retry_s = retry_s
        self._store = PendingCopyStore(self.source_root / "pending_copies.json")
        self._pending = self._store.load()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="file-copier", daemon=True
        )
        self._thread.start()
        if self._pending.paths:
            logger.info(
                "%d file(s) still to copy to %s", len(self._pending.paths), target_root
            )
            self._wake.set()

    @property
    def pending(self) -> list[str]:
        """Relative paths not yet copied, oldest first."""
        with self._lock:
            return list(self._pending.paths)

    def add(self, relpath: str) -> None:
        """Copy ``relpath`` (relative to ``source_root``) as soon as possible."""
        with self._lock:
            if relpath not in self._pending.paths:
                self._pending.paths.append(relpath)
                self._store.save(self._pending)
        self._wake.set()

    def close(self, timeout: float = 5.0) -> None:
        """Stop after the copy in progress; what is left is copied after a restart."""
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            while not self._stop.is_set() and (relpath := self._next()):
                try:
                    self._copy(relpath)
                except OSError:
                    logger.warning(
                        "Could not copy %s to %s; retrying in %.0f s",
                        relpath,
                        self.target_root,
                        self.retry_s,
                        exc_info=True,
                    )
                    self._stop.wait(self.retry_s)
                    continue
                with self._lock:
                    self._pending.paths.remove(relpath)
                    self._store.save(self._pending)

    def _next(self) -> str | None:
        with self._lock:
            return self._pending.paths[0] if self._pending.paths else None

    def _copy(self, relpath: str) -> None:
        source = self.source_root / relpath
        if not source.exists():
            logger.warning("%s no longer exists, so it was not copied", source)
            return
        target = self.target_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy under another name first, so an interrupted copy never looks complete.
        partial = target.with_name(target.name + ".partial")
        shutil.copy2(source, partial)
        os.replace(partial, target)
        logger.info("Copied %s to %s", relpath, self.target_root)
