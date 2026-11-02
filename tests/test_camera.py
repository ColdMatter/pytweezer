"""What the camera drivers do with frames and settings on their way through."""

import numpy as np
import pytest
import tifffile

from pytweezer.drivers import thorcam, tweezermonitorcam
from pytweezer.drivers.camera_base import Camera
from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.drivers.orca import ORCACamera

FRAMES = np.arange(5 * 2 * 3, dtype=np.uint16).reshape(5, 2, 3)


class BufferCamera(Camera):
    """A camera whose buffer holds ``FRAMES``; the hardware hooks do nothing else."""

    image_prefix = "test"

    def _connect(self):
        self._backend = FRAMES

    def _disconnect(self):
        pass

    def _read_frames(self, nframes, start_frame):
        return self._backend[start_frame : start_frame + nframes]

    set_roi = set_trigger_source = set_exposure_time = setup_acquisition = (
        lambda self, *args: None
    )
    start_acquisition = stop_acquisition = lambda self: None


class RecordingImageClient:
    def __init__(self):
        self.sent = []

    def send(self, info, image):
        self.sent.append((info, image))


def test_broadcast_frames_go_out_unchanged_in_order_with_their_frame_number():
    camera = BufferCamera()
    camera.image_client = RecordingImageClient()
    camera.acquire_n_frames(3, start_frame=2, broadcast=True)
    assert [info["index"] for info, _image in camera.image_client.sent] == [2, 3, 4]
    for (_info, image), frame in zip(camera.image_client.sent, FRAMES[2:], strict=True):
        np.testing.assert_array_equal(image, frame)


def test_autosave_writes_each_frame_to_its_own_tiff_in_frame_order(tmp_path):
    BufferCamera(image_dir=str(tmp_path)).acquire_n_frames(3, autosave=True)
    for i, frame in enumerate(FRAMES[:3], start=1):
        saved = tifffile.imread(tmp_path / f"test0000_{i}.tif")
        assert saved.dtype == np.uint16
        np.testing.assert_array_equal(saved, frame)
    assert len(list(tmp_path.iterdir())) == 3


def test_save_tiff_never_overwrites_an_earlier_image(tmp_path):
    BufferCamera.save_tiff(FRAMES[0], str(tmp_path))
    BufferCamera.save_tiff(FRAMES[1], str(tmp_path))
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "test0000_1.tif"), FRAMES[0]
    )
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "test0000_2.tif"), FRAMES[1]
    )


class FakeVendorCamera:
    """The pylablib camera at the SDK edge; records the frame range and ROI asked for."""

    def __init__(self):
        self.calls = {}

    def wait_for_frame(self, nframes, timeout):
        pass

    def read_multiple_images(self, frame_range, return_info):
        self.calls["frames"] = frame_range
        return FRAMES[slice(*frame_range)], None

    def set_roi(self, hstart, hend, vstart, vend):
        self.calls["roi"] = (hstart, hend, vstart, vend)


DRIVERS = [ImagEMX2Camera, ORCACamera, thorcam.ThorCam, tweezermonitorcam.ThorCam]
DRIVER_IDS = ["imagemX2", "orca", "thorcam", "tweezermonitorcam"]


def _driver(cls):
    driver = cls.__new__(cls)
    driver._backend, driver.timeout = FakeVendorCamera(), 1.0
    return driver


@pytest.mark.parametrize("cls", DRIVERS, ids=DRIVER_IDS)
def test_drivers_read_exactly_the_requested_frames(cls):
    driver = _driver(cls)
    np.testing.assert_array_equal(driver._read_frames(2, 3), FRAMES[3:5])
    assert driver._backend.calls["frames"] == (3, 5)


@pytest.mark.parametrize("cls", DRIVERS, ids=DRIVER_IDS)
def test_drivers_ask_pylablib_for_the_requested_roi(cls):
    # pylablib ROI ends are exclusive: hend - hstart is the width.
    driver = _driver(cls)
    driver.set_roi(110, 256, 220, 64)
    assert driver._backend.calls["roi"] == (110, 366, 220, 284)
