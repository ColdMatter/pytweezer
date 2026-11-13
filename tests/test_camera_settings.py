import contextlib

import pytest

from pytweezer.drivers.camera_catalogue import camera_names
from pytweezer.drivers.hamamatsu import set_dcam_exposure_mode
from pytweezer.experiment import run_local
from pytweezer.experiment.introspect import runnable_experiments
from pytweezer.experiments import camera_settings as cs


class RecordingCamera:
    def __init__(self, live=False):
        self.live = live
        self.calls = []

    def close_rpc(self):
        pass

    def is_live(self):
        return self.live

    def set_roi(self, *roi):
        self.calls.append(("set_roi", *roi))
        return roi[0] + 2, roi[2]

    def settings(self):
        return {"exposure_s": 0.01}

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))

        return call


@pytest.fixture
def camera(monkeypatch):
    camera = RecordingCamera()
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device", lambda name, timeout: camera
    )
    return camera


def run(experiment, **arguments):
    measurement = run_local(experiment, **arguments)
    assert measurement.status == "completed", measurement.attrs["error"]
    return measurement


def test_every_settings_experiment_names_a_configured_camera():
    listed = runnable_experiments(cs)
    cameras = [cls.devices()["camera"].device_name for cls in listed]
    assert set(cameras) <= set(camera_names())


def test_a_form_left_alone_changes_nothing(camera):
    measurement = run(cs.RbHamCamSettings)
    assert camera.calls == [("stop_live",)]
    assert measurement.results["camera_settings"][0] == {"exposure_s": 0.01}


@pytest.mark.parametrize("live", [True, False])
def test_ticked_settings_are_sent_in_order_while_stopped(camera, live):
    camera.live = live
    measurement = run(
        cs.RbThorCamSettings,
        set_exposure=True,
        exposure=2e-3,
        set_gain=True,
        gain=6.0,
        set_binning=True,
        binning=4,
        set_roi=True,
        roi_x0=8,
        roi_width=64,
        roi_y0=4,
        roi_height=32,
        trigger_mode="bulb",
    )
    assert camera.calls == [
        ("stop_live",),
        # binning can reset the ROI, so it goes first
        ("set_binning", 4),
        ("set_roi", 8, 64, 4, 32),
        ("set_exposure_time", 2e-3),
        ("set_trigger_source", "bulb"),
        ("set_gain", 6.0),
        *([("start_live",)] if live else []),
    ]
    assert list(measurement.results["roi_origin"][0]) == [10, 4]


@pytest.mark.parametrize(
    ("experiment", "arguments", "calls"),
    [
        (
            cs.RbHamCamSettings,
            {"exposure_mode": "edge"},
            [("set_exposure_mode", "edge")],
        ),
        (
            cs.RbHamCamSettings,
            {"em_gain": "on", "direct_em_gain": "off"},
            [("enable_em_gain", True), ("enable_direct_em_gain", False)],
        ),
        (
            cs.CaFOrcaSettings,
            {
                "exposure_mode": "internal",
                "set_readout_speed": True,
                "readout_speed": 2,
            },
            [("set_exposure_mode", "internal"), ("set_readout_speed", 2)],
        ),
        (
            cs.CaFThorCamSettings,
            {"trigger_polarity": "fall"},
            [("set_trigger_polarity", "fall")],
        ),
    ],
)
def test_driver_specific_choices_reach_the_camera(camera, experiment, arguments, calls):
    run(experiment, **arguments)
    assert camera.calls == [("stop_live",), *calls]


class RecordingDCAM:
    def __init__(self):
        self.calls = []

    def pausing_acquisition(self, **kwargs):
        return contextlib.nullcontext()

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, *args, *kwargs.items()))

        return call


@pytest.mark.parametrize(
    ("mode", "calls"),
    [
        ("internal", [("set_trigger_mode", "int")]),
        (
            "edge",
            [
                ("set_trigger_mode", "ext"),
                ("set_attribute_value", "trigger_active", 1),
                ("setup_ext_trigger", ("invert", True)),
            ],
        ),
        (
            "level",
            [
                ("set_trigger_mode", "ext"),
                ("set_attribute_value", "trigger_active", 2),
                ("setup_ext_trigger", ("invert", True)),
            ],
        ),
    ],
)
def test_dcam_exposure_mode_sets_the_trigger_activation(mode, calls):
    backend = RecordingDCAM()
    set_dcam_exposure_mode(backend, mode)
    assert backend.calls == calls
