"""Python side of ``lab_setup/lab-setup.ps1``: check, report and apply machine changes."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pytweezer.configuration.config import DATABASE
from pytweezer.configuration.paths import tweezerpath
from pytweezer.database.writer import DBWriter
from pytweezer.GUI.lab_setup_dialog import LabSetupDialog
from pytweezer.logging_utils import get_logger

logger = get_logger("lab_setup")

SKIP_ENV = "PYTWEEZER_SKIP_LAB_SETUP"
REPO = Path(tweezerpath)

_POWERSHELL = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]


@dataclass(frozen=True)
class ChangeStatus:
    id: str
    title: str
    description: str
    admin: bool
    state: str
    error: str | None

    @classmethod
    def from_json(cls, change: dict) -> "ChangeStatus":
        """Build from one -Check entry, ignoring fields main's tool added later."""
        id_ = str(change.get("id", ""))
        return cls(
            id=id_,
            title=change.get("title") or id_,
            description=change.get("description") or "",
            admin=bool(change.get("admin", False)),
            state=change.get("state", "error"),
            error=change.get("error"),
        )


@dataclass(frozen=True)
class CheckResult:
    commit: str
    host: str
    changes: list[ChangeStatus]

    @property
    def pending(self) -> list[ChangeStatus]:
        return [c for c in self.changes if c.state == "pending"]


def _tool(repo: Path) -> Path:
    return repo / "lab_setup" / "lab-setup.ps1"


def _run_bounded(command: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    """Run ``command``, killing its whole process tree if it outlives ``timeout``.

    Killing only the child would leave ``communicate`` blocked for as long as a
    grandchild (a hung ``git fetch``) holds the output pipes.

    Raises:
        subprocess.TimeoutExpired: The command did not finish in time.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        raise
    return process.returncode, stdout, stderr


def check(repo: Path = REPO, timeout: float = 10.0) -> CheckResult | None:
    """Ask the lab setup tool which changes apply here and whether they are in place.

    Never raises: returns None, after logging why, whenever the check cannot be
    completed, so that pytweezer starts normally.
    """
    if os.environ.get(SKIP_ENV):
        logger.info("Lab setup check skipped (%s set)", SKIP_ENV)
        return None
    if sys.platform != "win32":
        logger.info("Lab setup check skipped: not Windows")
        return None
    tool = _tool(repo)
    if not tool.is_file():
        logger.warning("Lab setup check skipped: %s is missing", tool)
        return None
    command = [*_POWERSHELL, str(tool), "-Check", "-Repo", str(repo)]
    try:
        returncode, stdout, stderr = _run_bounded(command, timeout)
        if returncode != 0:
            logger.warning(
                "Lab setup check exited with %d: %s",
                returncode,
                stderr.decode("utf-8", "replace").strip(),
            )
            return None
        data = json.loads(stdout.decode("utf-8-sig"))
        return CheckResult(
            commit=data["commit"],
            host=data["host"],
            changes=[ChangeStatus.from_json(c) for c in data["changes"]],
        )
    except subprocess.TimeoutExpired:
        logger.warning("Lab setup check timed out after %s s", timeout)
    except Exception:
        logger.exception("Lab setup check failed")
    return None


def _records(repo: Path):
    """Yield the record lines in file order, skipping any that do not parse."""
    try:
        text = (repo / "logs" / "lab_setup.jsonl").read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "change" in record:
            yield record


def last_attempts(repo: Path = REPO) -> dict[str, dict]:
    """Return each change id's latest record line."""
    return {record["change"]: record for record in _records(repo)}


def report(result: CheckResult, writer, repo: Path = REPO) -> None:
    """Queue one ``machine_changes`` row per change; does not wait for the database."""
    latest = {}
    last_done = {}
    for record in _records(repo):
        latest[record["change"]] = record
        if record.get("result") == "done":
            last_done[record["change"]] = record
    checked_at = datetime.now(UTC)
    for change in result.changes:
        attempt = latest.get(change.id)
        state, error = change.state, change.error
        if state == "pending" and attempt and attempt.get("result") == "failed":
            state, error = "failed", attempt.get("error")
        applied_at = None
        if change.id in last_done:
            applied_at = datetime.fromisoformat(last_done[change.id]["time"])
        writer.record_machine_change(
            result.host, change.id, state, checked_at, applied_at, result.commit, error
        )


def launch_apply(role: str, commit: str, repo: Path = REPO) -> None:
    """Start the tool in its own console to apply pending changes, then relaunch pytweezer.

    Args:
        role: ``server`` or ``client``, the pytweezer to relaunch afterwards.
        commit: The commit whose changes the person was shown, so a change
            merged since is not applied unseen.
        repo: The pytweezer working copy.
    """
    command = [
        *_POWERSHELL,
        str(_tool(repo)),
        "-Yes",
        "-Relaunch",
        role,
        "-Ref",
        commit,
        "-Repo",
        str(repo),
    ]
    subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)


_writer: DBWriter | None = None


def _report_writer() -> DBWriter:
    global _writer
    if _writer is None:
        _writer = DBWriter(DATABASE["server_dsn"])
    return _writer


def offer_lab_setup(role: str, *, timeout_s: float = 120) -> bool:
    """Report this PC's lab setup state and offer any pending changes.

    The writer is never flushed or closed here, so an unreachable database cannot
    delay startup; its daemon thread sends during the session.

    Returns:
        True when the apply tool was launched and pytweezer should exit.
    """
    try:
        result = check()
        if result is None:
            return False
        try:
            report(result, _report_writer())
        except Exception:
            logger.exception("Lab setup report failed; offering the changes anyway")
        pending = result.pending
        if not pending:
            return False
        dialog = LabSetupDialog(pending, timeout_s)
        if dialog.exec() != LabSetupDialog.DialogCode.Accepted:
            return False
        launch_apply(role, result.commit)
        return True
    except Exception:
        logger.exception("Lab setup offer failed; starting normally")
        return False
