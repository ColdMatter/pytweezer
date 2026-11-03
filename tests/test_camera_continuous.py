import pytest

from pytweezer.experiment import run_local
from pytweezer.experiments import camera_continuous as cc
from pytweezer.servers import device_server


class RecordingCamera:
    def __init__(self):
        self.calls = []

    def close_rpc(self):
        pass

    def set_roi(self, *roi):
        self.calls.append(("set_roi", *roi))
        return roi[0], roi[2]

    def settings(self):
        return {"exposure_s": 0.01}

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))

        return call


def test_dropdown_lists_only_active_cameras(monkeypatch):
    devices = {
        "Cam A": {"class": "pytweezer.drivers.thorcam:ThorCam"},
        "Cam Off": {
            "active": False,
            "class": "pytweezer.drivers.thorcam:ThorCam",
        },
        "Dac": {"class": "pytweezer.drivers.ni_cards:NICards"},
        "Rig": {
            "devices": {"Rig Cam": {"class": "pytweezer.drivers.orca:ORCACamera"}},
        },
    }
    monkeypatch.setattr(device_server, "get_config", lambda: {"Devices": devices})
    assert cc.camera_names() == ["Cam A", "Rig Cam"]


def test_task_reserves_every_camera_in_the_dropdown():
    assert set(cc.CAMERAS) <= set(cc.ContinuousAcquisition.reserved_devices())


@pytest.mark.parametrize("broadcast", [True, False])
@pytest.mark.parametrize(
    "chosen", [(), ("gain",), ("binning", "roi"), ("gain", "binning", "roi")]
)
def test_camera_is_reconfigured_while_stopped_then_left_acquiring(
    monkeypatch, broadcast, chosen
):
    camera = RecordingCamera()
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device", lambda name, timeout: camera
    )
    measurement = run_local(
        cc.ContinuousAcquisition,
        camera_name=cc.CAMERAS[0],
        exposure=2e-3,
        buffer_frames=50,
        broadcast=broadcast,
        set_gain="gain" in chosen,
        gain=6.0,
        set_binning="binning" in chosen,
        binning=4,
        set_roi="roi" in chosen,
        roi_x0=8,
        roi_width=64,
        roi_y0=4,
        roi_height=32,
    )
    assert measurement.status == "completed", measurement.attrs["error"]
    expected = [
        ("stop_live",),
        ("stop_acquisition",),
        ("set_trigger_source", "int"),
        ("set_exposure_time", 2e-3),
        *([("set_gain", 6.0)] if "gain" in chosen else []),
        # binning can reset the ROI, so it goes first
        *([("set_binning", 4)] if "binning" in chosen else []),
        *([("set_roi", 8, 64, 4, 32)] if "roi" in chosen else []),
        ("arm", 50),
        *([("start_live",)] if broadcast else []),
    ]
    assert camera.calls == expected
