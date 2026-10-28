import json
import subprocess
from datetime import UTC, datetime

import pytest

from pytweezer import lab_setup
from pytweezer.lab_setup import ChangeStatus, CheckResult


@pytest.fixture(autouse=True)
def _enable_lab_setup(monkeypatch):
    monkeypatch.delenv(lab_setup.SKIP_ENV, raising=False)


class FakeTool:
    """Stands in for ``subprocess.run`` of the PowerShell tool."""

    def __init__(self, repo):
        self.repo = repo
        self.calls = 0
        self.failure = None
        self.stdout = ""

    def prints(self, payload):
        self.stdout = json.dumps(payload)

    def fails(self, failure):
        self.failure = failure
        if failure == "missing_tool":
            (self.repo / "lab_setup" / "lab-setup.ps1").unlink()

    def run(self, command, timeout, **kwargs):
        self.calls += 1
        if self.failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if self.failure == "exit_code":
            return subprocess.CompletedProcess(command, 1, "", "no origin/main")
        stdout = "not json" if self.failure == "bad_json" else self.stdout
        return subprocess.CompletedProcess(command, 0, stdout, "")


@pytest.fixture
def fake_tool(tmp_path, monkeypatch):
    (tmp_path / "lab_setup").mkdir()
    (tmp_path / "lab_setup" / "lab-setup.ps1").write_text("")
    tool = FakeTool(tmp_path)
    monkeypatch.setattr(lab_setup.subprocess, "run", tool.run)
    return tool


def write_records(repo, records):
    (repo / "logs").mkdir(exist_ok=True)
    (repo / "logs" / "lab_setup.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


def result_with(*states):
    return CheckResult(
        commit="abc",
        host="PH-BEAST",
        changes=[ChangeStatus(i, i, "d", False, s, None) for i, s in states],
    )


def test_check_parses_the_tool_output(fake_tool):
    fake_tool.prints(
        {
            "commit": "abc",
            "host": "PH-BEAST",
            "changes": [
                {
                    "id": "a",
                    "title": "A",
                    "description": "d",
                    "admin": True,
                    "state": "pending",
                    "error": None,
                },
                {
                    "id": "b",
                    "title": "B",
                    "description": "d",
                    "admin": False,
                    "state": "done",
                    "error": None,
                },
            ],
        }
    )
    result = lab_setup.check(fake_tool.repo)
    assert [c.id for c in result.pending] == ["a"] and result.changes[0].admin


@pytest.mark.parametrize(
    "failure", ["timeout", "exit_code", "bad_json", "missing_tool"]
)
def test_check_returns_none_on_failure(fake_tool, failure):
    fake_tool.fails(failure)
    assert lab_setup.check(fake_tool.repo, timeout=1) is None


def test_check_is_skipped_by_the_environment(fake_tool, monkeypatch):
    monkeypatch.setenv(lab_setup.SKIP_ENV, "1")
    assert lab_setup.check(fake_tool.repo) is None
    assert fake_tool.calls == 0


def test_report_marks_failed_attempts_and_last_applied(tmp_path, recording_db):
    write_records(
        tmp_path,
        [
            {
                "time": "2026-10-28T09:00:00+00:00",
                "change": "a",
                "result": "done",
                "error": None,
            },
            {
                "time": "2026-10-28T10:00:00+00:00",
                "change": "b",
                "result": "failed",
                "error": "boom",
            },
        ],
    )
    lab_setup.report(
        result_with(("a", "done"), ("b", "pending"), ("c", "error")),
        recording_db,
        tmp_path,
    )
    rows = {r[1]: r for r in recording_db.machine_changes}
    assert rows["a"][2] == "done" and rows["a"][4] == datetime(
        2026, 10, 28, 9, tzinfo=UTC
    )
    assert rows["b"][2] == "failed" and rows["b"][6] == "boom"
    assert rows["c"][2] == "error"


def test_last_attempts_keeps_the_latest_line_and_skips_garbage(tmp_path):
    write_records(
        tmp_path,
        [
            {"time": "t1", "change": "a", "result": "failed", "error": "x"},
            {"time": "t2", "change": "a", "result": "done", "error": None},
        ],
    )
    with open(tmp_path / "logs" / "lab_setup.jsonl", "a") as f:
        f.write("{truncated\n")
    assert lab_setup.last_attempts(tmp_path)["a"]["result"] == "done"
    assert lab_setup.last_attempts(tmp_path / "nowhere") == {}


def test_launch_apply_opens_its_own_console(tmp_path, monkeypatch):
    launched = {}
    monkeypatch.setattr(
        lab_setup.subprocess,
        "Popen",
        lambda command, **kwargs: launched.update(command=command, **kwargs),
    )
    lab_setup.launch_apply("client", tmp_path)
    assert launched["command"][-5:] == [
        "-Yes",
        "-Relaunch",
        "client",
        "-Repo",
        str(tmp_path),
    ]
    assert launched["creationflags"] == subprocess.CREATE_NEW_CONSOLE
