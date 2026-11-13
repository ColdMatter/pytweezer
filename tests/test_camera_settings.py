import contextlib

import pytest

from pytweezer.drivers.camera_catalogue import camera_names
from pytweezer.drivers.hamamatsu import set_dcam_exposure_mode
from pytweezer.experiment import run_local
from pytweezer.experiment.introspect import runnable_experiments
from pytweezer.experiments import camera_settings as cs


class RecordingCamera:
    def __init__(self):
        self.applied = []

    def close_rpc(self):
        pass

    def apply_controls(self, values):
        self.applied.append(values)
        origin = {
            "roi_x0": values.get("roi_x0", 0) + 2,
            "roi_y0": values.get("roi_y0", 0),
        }
        return [
            {"name": name, "value": origin.get(name, value)}
            for name, value in values.items()
        ] + [{"name": name, "value": value} for name, value in origin.items()]

    def settings(self):
        return {"exposure_s": 0.01}


@pytest.fixture
def camera(monkeypatch):
    camera = RecordingCamera()
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device", lambda name, timeout: camera
    )
    return camera


@pytest.fixture
def stored(monkeypatch):
    stored = []
    monkeypatch.setattr(
        "pytweezer.experiment.camera_settings.store_camera_settings",
        lambda *call: stored.append(call),
    )
    return stored


def run(experiment, **arguments):
    measurement = run_local(experiment, **arguments)
    assert measurement.status == "completed", measurement.attrs["error"]
    return measurement


def test_every_settings_experiment_names_a_configured_camera():
    listed = runnable_experiments(cs)
    cameras = [cls.devices()["camera"].device_name for cls in listed]
    assert set(cameras) <= set(camera_names())


def test_a_form_left_alone_changes_nothing(camera, stored):
    measurement = run(cs.RbHamCamSettings)
    assert camera.applied == [{}]
    assert stored == []
    assert measurement.results["camera_settings"][0] == {"exposure_s": 0.01}


def test_ticked_settings_are_sent_together_and_stored(camera, stored):
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
        trigger_source="bulb",
    )
    sent = {
        "binning": 4,
        "roi_x0": 8,
        "roi_width": 64,
        "roi_y0": 4,
        "roi_height": 32,
        "exposure": 2e-3,
        "trigger_source": "bulb",
        "gain": 6.0,
    }
    assert camera.applied == [sent]
    assert list(measurement.results["roi_origin"][0]) == [10, 4]
    [(name, values, reason)] = stored
    assert name == "Rb ThorCam"
    assert values == {**sent, "roi_x0": 10}
    assert reason.startswith("RbThorCamSettings, rid ")


@pytest.mark.parametrize(
    ("experiment", "arguments", "values"),
    [
        (cs.RbHamCamSettings, {"exposure_mode": "edge"}, {"exposure_mode": "edge"}),
        (
            cs.RbHamCamSettings,
            {"em_gain": "on", "direct_em_gain": "off"},
            {"em_gain": True, "direct_em_gain": False},
        ),
        (
            cs.CaFOrcaSettings,
            {
                "exposure_mode": "internal",
                "set_readout_speed": True,
                "readout_speed": 2,
            },
            {"exposure_mode": "internal", "readout_speed": 2},
        ),
        (
            cs.CaFThorCamSettings,
            {"trigger_polarity": "fall"},
            {"trigger_polarity": "fall"},
        ),
    ],
)
def test_driver_specific_choices_reach_the_camera(
    camera, stored, experiment, arguments, values
):
    run(experiment, **arguments)
    assert camera.applied == [values]


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
