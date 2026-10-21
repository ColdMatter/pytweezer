import contextlib
from typing import ClassVar

import pytest

from pytweezer.drivers import hamamatsu

CAMERAS = [
    {"index": 0, "model": "C10600-10B (ORCA-R2)", "serial_number": "280187"},
    {"index": 1, "model": "C9100-23B", "serial_number": "000407"},
]


@pytest.fixture(autouse=True)
def connected_cameras(monkeypatch):
    monkeypatch.setattr(hamamatsu, "list_dcam_cameras", lambda: CAMERAS)


@pytest.mark.parametrize("serial", ["000407", "S/N: 000407"])
def test_selects_by_serial_regardless_of_model(serial):
    assert hamamatsu.find_dcam_index(serial, "C10600") == 1


def test_falls_back_to_model_without_serial():
    assert hamamatsu.find_dcam_index(None, "C9100") == 1
    assert hamamatsu.find_dcam_index(None, "C10600") == 0


def test_missing_serial_raises():
    with pytest.raises(RuntimeError, match="serial number 123456"):
        hamamatsu.find_dcam_index("123456", "C9100")


def test_ambiguous_model_raises(monkeypatch):
    monkeypatch.setattr(
        hamamatsu,
        "list_dcam_cameras",
        lambda: CAMERAS + [{"index": 2, "model": "C9100-23B", "serial_number": "1"}],
    )
    with pytest.raises(RuntimeError, match="found 2"):
        hamamatsu.find_dcam_index(None, "C9100")


class _FakeDCAMCamera:
    serials: ClassVar[dict[int, str]] = {0: "S/N: 280187", 1: "S/N: 000407"}

    def __init__(self, index):
        self.serial_number = self.serials[index]
        self.closed = False

    def get_device_info(self):
        return type("Info", (), {"serial_number": self.serial_number})

    def close(self):
        self.closed = True


@pytest.fixture
def fake_dcam(monkeypatch):
    monkeypatch.setattr(hamamatsu, "DCAMCamera", _FakeDCAMCamera)
    monkeypatch.setattr(hamamatsu.libctl, "temp_open", contextlib.nullcontext)


def test_open_returns_the_requested_camera(fake_dcam):
    assert hamamatsu.open_dcam_camera("000407", "C9100").serial_number.endswith(
        "000407"
    )


def test_open_rejects_a_camera_with_the_wrong_serial(fake_dcam, monkeypatch):
    monkeypatch.setattr(_FakeDCAMCamera, "serials", {1: "S/N: 280187"})
    with pytest.raises(RuntimeError, match="expected 000407"):
        hamamatsu.open_dcam_camera("000407", "C9100")
