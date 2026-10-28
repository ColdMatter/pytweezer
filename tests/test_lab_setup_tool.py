"""lab_setup/lab-setup.ps1 against real git repos: a bare ``origin`` and a clone of it."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="lab-setup.ps1 needs Windows PowerShell"
)

TOOL = Path(__file__).resolve().parents[1] / "lab_setup" / "lab-setup.ps1"


def _git(cwd, *args):
    done = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _new_clone(origin, path):
    _git(origin.parent, "clone", "-q", str(origin), str(path))
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.com")


def _write_change(work, change_id, check, apply, systems, hosts):
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
    marker = f'(Join-Path $Repo "markers/{change_id}")'
    if check is None:
        check = f"if (Test-Path {marker}) {{ exit 0 }} else {{ exit 1 }}"
    if apply is None:
        apply = f"New-Item -ItemType File -Force {marker} | Out-Null"
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
    create marker files ``<work>/markers/<id>``.
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
        shutil.copy(TOOL, self.work / "lab_setup" / "lab-setup.ps1")
        _commit_and_push(self.work, "Add the tool")

    def git(self, *args):
        return _git(self.work, *args)

    def add_change(
        self, change_id, check=None, apply=None, systems=("windows",), hosts=None
    ):
        self.add_change_uncommitted(change_id, check, apply, systems, hosts)
        _commit_and_push(self.work, f"Add {change_id}")

    def add_change_uncommitted(
        self, change_id, check=None, apply=None, systems=("windows",), hosts=None
    ):
        _write_change(self.work, change_id, check, apply, list(systems), hosts)

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

    def run(self, *args):
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
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={**os.environ, "TEMP": str(self.temp), "TMP": str(self.temp)},
        )

    def check(self, host="PH-BEAST", fetch=False):
        done = self.run("-Check", "-HostName", host, *([] if fetch else ["-NoFetch"]))
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)


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
