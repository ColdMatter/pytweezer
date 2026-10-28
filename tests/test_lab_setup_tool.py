"""lab_setup/lab-setup.ps1 against real git repos: a bare ``origin`` and a clone of it."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="lab-setup.ps1 needs Windows PowerShell"
)

LAB_SETUP = Path(__file__).resolve().parents[1] / "lab_setup"
APPLIED_ORDER = "_applied_order.txt"


def _git(cwd, *args):
    done = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _new_clone(origin, path):
    _git(origin.parent, "clone", "-q", str(origin), str(path))
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.com")


def _write_change(work, change_id, check, apply, systems, hosts, secrets=()):
    folder = work / "lab_setup" / "changes" / change_id
    folder.mkdir(parents=True)
    manifest = {
        "title": f"Title of {change_id}",
        "description": f"Description of {change_id}",
        "systems": systems,
        "admin": False,
    }
    if hosts is not None:
        manifest["hosts"] = hosts
    if secrets:
        manifest["secrets"] = [
            {"name": name, "prompt": f"Value of {name}", "ask": "Tom"}
            for name in secrets
        ]
    marker = f'(Join-Path $Repo "markers/{change_id}")'
    if check is None:
        check = f"if (Test-Path {marker}) {{ exit 0 }} else {{ exit 1 }}"
    if apply is None:
        apply = (
            f"New-Item -ItemType File -Force {marker} | Out-Null\n"
            f"Add-Content (Join-Path $Repo 'markers/{APPLIED_ORDER}') '{change_id}'"
        )
    (folder / "change.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "check.ps1").write_text(
        f"param([string]$Repo, [hashtable]$Secrets)\n{check}\n", encoding="utf-8"
    )
    (folder / "apply.ps1").write_text(
        f"param([string]$Repo, [hashtable]$Secrets)\n{apply}\n", encoding="utf-8"
    )


def _commit_and_push(work, message):
    _git(work, "add", "-A", "lab_setup")
    _git(work, "commit", "-q", "-m", message)
    _git(work, "push", "-q", "origin", "HEAD:main")


class LabRepo:
    """A pytweezer clone whose ``origin/main`` holds the tool under test.

    ``work`` is the clone the tool is run against; fake changes check for and
    create marker files ``<work>/markers/<id>``, and their applies append
    their id to ``<work>/markers/_applied_order.txt``.
    """

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.origin = tmp_path / "origin.git"
        self.work = tmp_path / "work"
        self.temp = tmp_path / "temp"
        self.temp.mkdir()
        subprocess.run(
            ["git", "init", "-q", "--bare", "-b", "main", str(self.origin)],
            check=True,
        )
        _new_clone(self.origin, self.work)
        (self.work / "lab_setup").mkdir()
        for script in LAB_SETUP.glob("*.ps1"):
            shutil.copy(script, self.work / "lab_setup" / script.name)
        _commit_and_push(self.work, "Add the tool")

    def git(self, *args):
        return _git(self.work, *args)

    def add_change(
        self,
        change_id,
        check=None,
        apply=None,
        systems=("windows",),
        hosts=None,
        secrets=(),
    ):
        self.add_change_uncommitted(change_id, check, apply, systems, hosts, secrets)
        _commit_and_push(self.work, f"Add {change_id}")

    def add_change_uncommitted(
        self,
        change_id,
        check=None,
        apply=None,
        systems=("windows",),
        hosts=None,
        secrets=(),
    ):
        _write_change(self.work, change_id, check, apply, list(systems), hosts, secrets)

    def push_change_from_elsewhere(self, change_id):
        elsewhere = self.tmp_path / "elsewhere"
        _new_clone(self.origin, elsewhere)
        _write_change(elsewhere, change_id, None, None, ["windows"], None)
        _commit_and_push(elsewhere, f"Add {change_id}")

    def break_origin_url(self):
        self.git("remote", "set-url", "origin", str(self.tmp_path / "missing.git"))

    def mark_done(self, change_id):
        markers = self.work / "markers"
        markers.mkdir(exist_ok=True)
        (markers / change_id).touch()

    def origin_main_sha(self):
        return _git(self.origin, "rev-parse", "main")

    def run(self, *args, env=None):
        """Run the tool from the clone's working copy, as pytweezer does."""
        return subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.work / "lab_setup" / "lab-setup.ps1"),
                "-Repo",
                str(self.work),
                *args,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={
                **os.environ,
                "TEMP": str(self.temp),
                "TMP": str(self.temp),
                **(env or {}),
            },
        )

    def check(self, host="PH-BEAST", fetch=False):
        done = self.run("-Check", "-HostName", host, *([] if fetch else ["-NoFetch"]))
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)

    def apply(self, *args, yes=False, host="PH-BEAST", secret=None):
        """``secret`` is given as the value of every secret, instead of a prompt."""
        return self.run(
            *args,
            *(["-Yes"] if yes else []),
            "-NoFetch",
            "-HostName",
            host,
            env={"PYTWEEZER_LAB_SETUP_TEST_SECRET": secret} if secret else None,
        )

    def records(self):
        path = self.work / "logs" / "lab_setup.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text("utf-8").splitlines()]

    def add_launcher(self, name):
        """A launcher that records the TEMP it was started with, then exits."""
        (self.work / "markers").mkdir(exist_ok=True)
        (self.work / name).write_text(
            '@echo %TEMP%> "%~dp0markers\\relaunched.txt"\n', encoding="ascii"
        )
        return self.work / "markers" / "relaunched.txt"

    def marker_order(self):
        path = self.work / "markers" / APPLIED_ORDER
        return path.read_text("utf-8").split() if path.exists() else []


@pytest.fixture
def lab_repo(tmp_path):
    return LabRepo(tmp_path)


def test_check_lists_pending_and_done(lab_repo):
    lab_repo.add_change("2026-01-01-a")
    lab_repo.add_change("2026-01-02-b")
    lab_repo.mark_done("2026-01-01-a")
    out = lab_repo.check()
    assert [(c["id"], c["state"]) for c in out["changes"]] == [
        ("2026-01-01-a", "done"),
        ("2026-01-02-b", "pending"),
    ]
    assert out["commit"] == lab_repo.origin_main_sha()


def test_host_filters_ignore_case(lab_repo):
    lab_repo.add_change("only-beast", hosts={"only": ["PH-BEAST"]})
    lab_repo.add_change("not-beast", hosts={"except": ["ph-beast"]})
    ids = [c["id"] for c in lab_repo.check(host="ph-beast")["changes"]]
    assert ids == ["only-beast"]


def test_template_and_other_systems_are_skipped(lab_repo):
    lab_repo.add_change("_template")
    lab_repo.add_change("mac-only", systems=["macos"])
    assert lab_repo.check()["changes"] == []


def test_check_reads_origin_main_not_the_working_copy(lab_repo):
    lab_repo.add_change("on-main")
    lab_repo.git("switch", "-c", "elsewhere")
    lab_repo.add_change_uncommitted("only-local")
    status = lab_repo.git("status", "--porcelain")
    assert [c["id"] for c in lab_repo.check()["changes"]] == ["on-main"]
    assert lab_repo.git("status", "--porcelain") == status
    assert lab_repo.git("branch", "--show-current") == "elsewhere"


def test_check_sees_changes_pushed_after_the_clone(lab_repo):
    lab_repo.push_change_from_elsewhere("new")
    assert "new" in [c["id"] for c in lab_repo.check(fetch=True)["changes"]]


def test_check_uses_existing_origin_main_when_fetch_fails(lab_repo):
    lab_repo.add_change("a")
    lab_repo.break_origin_url()
    assert [c["id"] for c in lab_repo.check(fetch=True)["changes"]] == ["a"]


def test_a_throwing_check_is_reported_as_error(lab_repo):
    lab_repo.add_change("bad", check='throw "boom"')
    lab_repo.add_change("good")
    states = {c["id"]: c["state"] for c in lab_repo.check()["changes"]}
    assert states == {"bad": "error", "good": "pending"}


def test_apply_applies_pending_in_order_and_records(lab_repo):
    lab_repo.add_change("2026-01-01-a")
    lab_repo.add_change("2026-01-02-b")
    lab_repo.apply(yes=True)
    assert lab_repo.marker_order() == ["2026-01-01-a", "2026-01-02-b"]
    records = lab_repo.records()
    assert [(r["change"], r["result"]) for r in records] == [
        ("2026-01-01-a", "done"),
        ("2026-01-02-b", "done"),
    ]
    assert {c["state"] for c in lab_repo.check()["changes"]} == {"done"}


def test_apply_stops_at_the_first_failure(lab_repo):
    lab_repo.add_change("2026-01-01-a", apply='throw "disk full"')
    lab_repo.add_change("2026-01-02-b")
    lab_repo.apply(yes=True)
    (record,) = lab_repo.records()
    assert record["result"] == "failed" and "disk full" in record["error"]
    assert lab_repo.marker_order() == []


def test_apply_that_does_not_satisfy_its_check_is_failed(lab_repo):
    lab_repo.add_change("a", apply="Write-Host 'did nothing'")
    lab_repo.apply(yes=True)
    assert lab_repo.records()[0]["result"] == "failed"


def test_apply_skips_changes_already_done(lab_repo):
    lab_repo.add_change("a")
    lab_repo.mark_done("a")
    lab_repo.apply(yes=True)
    assert lab_repo.records() == []


def test_secret_values_are_masked_in_the_record(lab_repo):
    lab_repo.add_change(
        "a", apply='throw "login with $($Secrets.PW) failed"', secrets=["PW"]
    )
    lab_repo.apply(yes=True, secret="hunter2")
    (record,) = lab_repo.records()
    assert record["error"] == "login with *** failed"


def test_a_throwing_check_reports_its_message_as_plain_text(lab_repo):
    lab_repo.add_change("bad", check='throw "boom"')
    (change,) = lab_repo.check()["changes"]
    assert change["error"] == "boom"


def test_check_at_a_commit_reads_that_commit_without_fetching(lab_repo):
    lab_repo.add_change("seen")
    seen = lab_repo.origin_main_sha()
    lab_repo.push_change_from_elsewhere("merged-later")
    lab_repo.git("fetch", "-q", "origin")
    done = lab_repo.run("-Check", "-Ref", seen, "-HostName", "PH-BEAST")
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["commit"] == seen
    assert [c["id"] for c in out["changes"]] == ["seen"]


def test_a_truncated_record_line_does_not_swallow_the_next_record(lab_repo):
    lab_repo.add_change("a")
    (lab_repo.work / "logs").mkdir()
    (lab_repo.work / "logs" / "lab_setup.jsonl").write_text('{"trunc', "utf-8")
    lab_repo.apply(yes=True)
    lines = (lab_repo.work / "logs" / "lab_setup.jsonl").read_text("utf-8")
    assert json.loads(lines.splitlines()[-1])["change"] == "a"


def _wait_for(path, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.2)
    return None


def test_relaunch_still_happens_after_an_error_with_a_fresh_environment(lab_repo):
    lab_repo.add_change("a")
    (lab_repo.work / "logs" / "lab_setup.jsonl").mkdir(parents=True)
    relaunched = lab_repo.add_launcher("start_client.bat")
    done = lab_repo.apply("-Relaunch", "client", yes=True)
    assert "lab_setup.jsonl" in done.stdout
    temp = _wait_for(relaunched)
    assert temp is not None, done.stdout + done.stderr
    assert temp != str(lab_repo.temp)


def test_relaunch_without_its_launcher_says_so(lab_repo):
    done = lab_repo.apply("-Relaunch", "server", yes=True)
    assert "start_servers.bat" in done.stdout
    assert done.returncode == 0
