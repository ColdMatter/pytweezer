"""Finished measurement files copied to the lab network drive."""

from pytweezer.experiment.copier import FileCopier
from tests.helpers import until

RELPATH = "2026/10/07/000001_Sweep.h5"


def write_source(root, data=b"measurement bytes"):
    path = root / RELPATH
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def test_a_file_lands_at_the_same_path_unchanged(tmp_path):
    write_source(tmp_path / "local")
    copier = FileCopier(tmp_path / "local", tmp_path / "drive")
    try:
        copier.add(RELPATH)
        assert until(lambda: not copier.pending)
    finally:
        copier.close()
    assert (tmp_path / "drive" / RELPATH).read_bytes() == b"measurement bytes"
    assert list((tmp_path / "drive" / RELPATH).parent.iterdir()) == [
        tmp_path / "drive" / RELPATH
    ]


def test_a_drive_that_refuses_permission_changes_still_gets_the_file(
    tmp_path, monkeypatch
):
    def refuse(*args, **kwargs):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr("os.chmod", refuse)
    write_source(tmp_path / "local")
    copier = FileCopier(tmp_path / "local", tmp_path / "drive", retry_s=60)
    try:
        copier.add(RELPATH)
        assert until(lambda: not copier.pending)
    finally:
        copier.close()
    assert (tmp_path / "drive" / RELPATH).read_bytes() == b"measurement bytes"


def test_an_unreachable_drive_is_retried_until_the_copy_lands(tmp_path):
    write_source(tmp_path / "local")
    drive = tmp_path / "drive"
    drive.write_text("a file where the drive's folder should be")
    copier = FileCopier(tmp_path / "local", drive, retry_s=0.05)
    try:
        copier.add(RELPATH)
        assert not until(lambda: not copier.pending, timeout=0.3)
        drive.unlink()
        assert until(lambda: not copier.pending)
    finally:
        copier.close()
    assert (drive / RELPATH).exists()


def test_copies_still_pending_resume_after_a_restart(tmp_path):
    write_source(tmp_path / "local")
    drive = tmp_path / "drive"
    drive.write_text("unreachable")
    copier = FileCopier(tmp_path / "local", drive, retry_s=60)
    copier.add(RELPATH)
    copier.close()
    drive.unlink()

    restarted = FileCopier(tmp_path / "local", drive)
    try:
        assert until(lambda: not restarted.pending)
    finally:
        restarted.close()
    assert (drive / RELPATH).exists()
