"""What the camera drivers do with frames and settings on their way through."""

import contextlib
import threading

import numpy as np
import pytest
import tifffile

from pytweezer.drivers import thorcam, tweezermonitorcam
from pytweezer.drivers.camera_base import Camera, SimulatedCamera
from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.drivers.orca_fusion import ORCAFusionCamera
from pytweezer.drivers.orca_r2 import ORCAR2Camera

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


def test_broadcast_frames_carry_the_header_the_caller_adds():
    camera = BufferCamera()
    camera.image_client = RecordingImageClient()
    camera.acquire_n_frames(2, broadcast=True, header={"rid": 57})
    assert [info["rid"] for info, _image in camera.image_client.sent] == [57, 57]


def test_frames_broadcast_on_another_stream_leave_the_camera_s_own_alone(monkeypatch):
    camera = BufferCamera()
    camera.image_client = RecordingImageClient()
    other = RecordingImageClient()
    monkeypatch.setattr(camera, "_client_for", lambda stream: other)
    camera.acquire_n_frames(2, broadcast=True, stream="elsewhere")
    assert camera.image_client.sent == []
    assert len(other.sent) == 2


def _scaled_broadcast_header(**camera_kwargs):
    camera = SimulatedCamera(**camera_kwargs)
    camera.image_client = RecordingImageClient()
    camera.set_binning(2)
    camera.set_roi(100, 64, 40, 64)
    camera.broadcast_image(FRAMES[0], {"index": 0})
    return camera.image_client.sent[0][0]


def test_broadcast_header_carries_pixel_scale():
    header = _scaled_broadcast_header(object_pixel_m=2e-6)
    assert header["_imgresolution"] == pytest.approx([4e-6, 4e-6])
    assert header["_offset"] == pytest.approx([50, 20])


def test_broadcast_header_without_scale_is_unchanged():
    header = _scaled_broadcast_header()
    assert "_imgresolution" not in header
    assert "_offset" not in header


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
        self.binning = (1, 1)

    def wait_for_frame(self, nframes, timeout):
        pass

    def read_multiple_images(self, frame_range, return_info):
        self.calls["frames"] = frame_range
        return FRAMES[slice(*frame_range)], None

    def setup_acquisition(self, *args):
        self.calls["setup"] = args

    def start_acquisition(self):
        self.calls["started"] = True

    def acquisition_in_progress(self):
        return "started" in self.calls

    def get_roi(self):
        return (0, 1024, 0, 1024, *self.binning)

    def set_roi(self, hstart, hend, vstart, vend, hbin=1, vbin=1):
        # pylablib writes the binning on every call, 1 unless it is passed.
        self.calls["roi"] = (hstart, hend, vstart, vend)
        self.binning = (hbin, vbin)
        # Like DCAM, which moves the ROI to its subarray step; returns get_roi().
        return (hstart // 4 * 4, hend, vstart, vend, hbin, vbin)


DCAM_DRIVERS = [ImagEMX2Camera, ORCAR2Camera, ORCAFusionCamera]
DCAM_DRIVER_IDS = ["imagemX2", "orca_r2", "orca_fusion"]
DRIVERS = [*DCAM_DRIVERS, thorcam.ThorCam, tweezermonitorcam.ThorCam]
DRIVER_IDS = [*DCAM_DRIVER_IDS, "thorcam", "tweezermonitorcam"]


def _driver(cls):
    driver = cls.__new__(cls)
    driver._backend, driver.timeout = FakeVendorCamera(), 1.0
    driver._lock = threading.RLock()
    driver._kept = {}
    driver._shot_tags = []
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


@pytest.mark.parametrize("cls", DRIVERS, ids=DRIVER_IDS)
def test_setting_the_roi_keeps_the_binning(cls):
    driver = _driver(cls)
    driver._backend.binning = (4, 4)
    driver.set_roi(110, 256, 220, 64)
    assert driver._backend.binning == (4, 4)


def _refuse_bare_binning_write(*args, **kwargs):
    raise AssertionError("DCAMERR_NOTSTABLE: frame buffer attached")


@pytest.mark.parametrize("cls", DCAM_DRIVERS, ids=DCAM_DRIVER_IDS)
def test_dcam_binning_goes_through_set_roi_and_keeps_the_roi(cls):
    # DCAM refuses a bare BINNING write while the frame buffer is attached, which
    # it is whenever the camera has been live; pylablib's set_roi frees it.
    driver = _driver(cls)
    driver._backend.set_attribute_value = _refuse_bare_binning_write
    driver.set_binning(8)
    assert driver._backend.binning == (8, 8)
    assert driver._backend.calls["roi"] == (0, 1024, 0, 1024)
    assert driver.binning == 8


@pytest.mark.parametrize("cls", DRIVERS, ids=DRIVER_IDS)
def test_drivers_keep_the_roi_origin_the_camera_used(cls):
    driver = _driver(cls)
    assert driver.set_roi(110, 256, 220, 64) == (108, 220)
    assert driver.roi_origin == (108, 220)


@pytest.mark.parametrize("cls", DRIVERS, ids=DRIVER_IDS)
def test_drivers_keep_the_binning_the_camera_reads_with(cls):
    driver = _driver(cls)
    driver._backend.binning = (4, 4)
    driver.set_roi(110, 256, 220, 64)
    assert driver.binning == 4


# pylablib's DCAM cameras take (mode, nframes); its Thorlabs cameras have no
# modes and take only the ring-buffer size.
SETUPS = {
    ImagEMX2Camera: (("sequence", 7), ("snap", 2)),
    ORCAR2Camera: (("sequence", 7), ("snap", 2)),
    ORCAFusionCamera: (("sequence", 7), ("snap", 2)),
    thorcam.ThorCam: ((7,), (2,)),
    tweezermonitorcam.ThorCam: ((7,), (2,)),
}


@pytest.mark.parametrize("cls", DRIVERS, ids=DRIVER_IDS)
def test_arm_starts_a_ring_buffer_of_the_requested_size(cls):
    driver = _driver(cls)
    driver.arm(buffer_frames=7)
    assert driver._backend.calls["setup"] == SETUPS[cls][0]
    assert driver._backend.calls["started"]


@pytest.mark.parametrize("cls", DRIVERS, ids=DRIVER_IDS)
def test_setup_acquisition_passes_the_mode_only_to_cameras_that_have_one(cls):
    driver = _driver(cls)
    driver.setup_acquisition(2, mode="snap")
    assert driver._backend.calls["setup"] == SETUPS[cls][1]


class FakeDCAMAttributes(FakeVendorCamera):
    """DCAM attributes as values by name, written only while acquisition is paused."""

    def __init__(self, **attributes):
        super().__init__()
        self.attributes = attributes
        self.paused = False

    @contextlib.contextmanager
    def pausing_acquisition(self, clear=None):
        self.paused = True
        yield
        self.paused = False

    def set_attribute_value(self, name, value):
        assert self.paused, f"{name} written while acquiring"
        self.attributes[name] = value

    def get_attribute_value(self, name, **kwargs):
        return self.attributes[name]


# DCAM's TRIGGER GLOBAL EXPOSURE: 3 is DELAYED, 5 is GLOBAL RESET.
@pytest.mark.parametrize(("enable", "dcam_value"), [(True, 5), (False, 3)])
def test_fusion_global_reset_reaches_dcam_and_is_kept(enable, dcam_value):
    driver = _driver(ORCAFusionCamera)
    driver._backend = FakeDCAMAttributes(trigger_global_exposure=0)
    driver.set_global_reset(enable)
    assert driver._backend.attributes["trigger_global_exposure"] == dcam_value
    assert driver._kept["global_reset"] is enable
