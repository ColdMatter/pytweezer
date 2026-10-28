"""Static checks on every folder under lab_setup/changes/."""

import json
import re
from pathlib import Path

import pytest

CHANGES = Path(__file__).resolve().parents[1] / "lab_setup" / "changes"


@pytest.mark.parametrize(
    "folder", sorted(p for p in CHANGES.iterdir() if p.is_dir()), ids=lambda p: p.name
)
def test_every_change_is_well_formed(folder):
    manifest = json.loads((folder / "change.json").read_text())
    assert manifest["title"] and manifest["description"]
    assert set(manifest["systems"]) <= {"windows", "macos", "linux"}
    assert len(manifest.get("hosts", {})) <= 1 and set(manifest.get("hosts", {})) <= {
        "only",
        "except",
    }
    assert isinstance(manifest["admin"], bool)
    for secret in manifest.get("secrets", []):
        assert set(secret) == {"name", "prompt", "ask"}
    if "windows" in manifest["systems"]:
        assert (folder / "check.ps1").is_file() and (folder / "apply.ps1").is_file()
    if not folder.name.startswith("_"):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+", folder.name)


def test_changes_do_not_use_psscriptroot_for_repo_paths():
    for script in CHANGES.glob("*/*.ps1"):
        assert "$PSScriptRoot" not in script.read_text(), script


def test_changes_are_ascii_only():
    for path in CHANGES.glob("*/*"):
        path.read_bytes().decode("ascii")


def test_the_first_changes_exist():
    names = {p.name for p in CHANGES.iterdir()}
    assert {
        "2026-10-28-install-uv",
        "2026-10-28-db-dsn",
        "2026-10-28-launch-tasks",
        "2026-10-28-pytweezer-data-share",
        "2026-10-28-data-share-login",
    } <= names
