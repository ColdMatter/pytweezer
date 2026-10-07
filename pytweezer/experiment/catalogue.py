"""The list of available experiments, kept up to date as files change.

Each module under the experiments package is described by running
:mod:`pytweezer.experiment.introspect` in a subprocess; results are cached by
file modification time and size, and :meth:`Catalogue.poll` never blocks.
"""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from pytweezer.configuration.paths import tweezerpath

INTROSPECT_TIMEOUT_S = 60.0


@dataclass
class _Pending:
    process: subprocess.Popen
    output: IO[bytes]
    stamp: tuple[int, int]
    started: float


class Catalogue:
    def __init__(
        self,
        package: str = "pytweezer.experiments",
        directory: Path | str | None = None,
        max_parallel: int = 4,
    ) -> None:
        self.package = package
        self.directory = Path(directory or Path(tweezerpath, *package.split(".")))
        self.max_parallel = max_parallel
        self._entries: dict[str, dict[str, Any]] = {}
        self._stamps: dict[str, tuple[int, int]] = {}
        self._pending: dict[str, _Pending] = {}

    def modules(self) -> dict[str, tuple[Path, tuple[int, int]]]:
        """``{module name: (file, (mtime_ns, size))}`` for every experiment module."""
        found = {}
        for path in sorted(self.directory.rglob("*.py")):
            relative = path.relative_to(self.directory).with_suffix("")
            if any(part.startswith("_") for part in relative.parts):
                continue
            stat = path.stat()
            module = ".".join((self.package, *relative.parts))
            found[module] = (path, (stat.st_mtime_ns, stat.st_size))
        return found

    def refresh(self) -> bool:
        """Start describing new or changed modules; forget deleted ones. Returns whether anything was dropped."""
        modules = self.modules()
        dropped = False
        for module in set(self._entries) - set(modules):
            del self._entries[module]
            self._stamps.pop(module, None)
            dropped = True
        for module, (_path, stamp) in modules.items():
            if self._stamps.get(module) == stamp or module in self._pending:
                continue
            if len(self._pending) >= self.max_parallel:
                break
            self._start(module, stamp)
        return dropped

    def _start(self, module: str, stamp: tuple[int, int]) -> None:
        # A file rather than a pipe: a pipe fills up and stalls the child
        # unless someone reads it continuously.
        output = tempfile.TemporaryFile()  # noqa: SIM115 - closed in poll()
        process = subprocess.Popen(
            [sys.executable, "-m", "pytweezer.experiment.introspect", module],
            stdout=output,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=tweezerpath,
        )
        self._pending[module] = _Pending(process, output, stamp, time.monotonic())

    def poll(self) -> bool:
        """Collect finished descriptions. Returns whether the catalogue changed."""
        changed = False
        for module, pending in list(self._pending.items()):
            if pending.process.poll() is None:
                if time.monotonic() - pending.started < INTROSPECT_TIMEOUT_S:
                    continue
                pending.process.kill()
                pending.process.wait()
                entry = _error_entry(module, "timed out while importing")
            else:
                pending.output.seek(0)
                try:
                    entry = json.loads(pending.output.read())
                except ValueError:
                    entry = _error_entry(
                        module,
                        f"introspection exited with code {pending.process.returncode}",
                    )
            pending.output.close()
            del self._pending[module]
            entry["stamp"] = list(pending.stamp)
            self._entries[module] = entry
            self._stamps[module] = pending.stamp
            changed = True
        return changed

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    def entries(self) -> list[dict[str, Any]]:
        return [self._entries[name] for name in sorted(self._entries)]

    def close(self) -> None:
        for pending in self._pending.values():
            pending.process.kill()
            pending.output.close()
        self._pending.clear()


def _error_entry(module: str, error: str) -> dict[str, Any]:
    return {"module": module, "classes": [], "warnings": [], "error": error}
