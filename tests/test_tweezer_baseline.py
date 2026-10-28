"""TweezerBaseline: the saved MOTMaster settings and camera setup it runs with."""

import inspect
import json

import numpy as np
import pytest

from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.experiment import Scan, run_local
from pytweezer.experiments.tweezer_baseline import (
    SAVED_MOTMASTER_PARAMETERS,
    TweezerBaseline,
)

SAVED = {
    name: 100.0 + i
    for i, name in enumerate(dict.fromkeys(SAVED_MOTMASTER_PARAMETERS.values()))
}


class FakeSequencer:
    """The Rb MOTMaster: knows the script's parameters and records each run's."""

    def __init__(self):
        self.runs = []

    def get_params(self):
        script = {"tTweezerSetVVA0": 0.0, "tTweezerSetVVA1": 0.0, "BackgroundDrop": 1}
        return script | dict.fromkeys(SAVED_MOTMASTER_PARAMETERS, 0.0)

    def start_motmaster_experiment(self, parameters=None):
        self.runs.append(parameters)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeCamera:
    """The Rb HamCam: records the settings it is given."""

    def __init__(self):
        self.settings = {}

    def set_roi(self, *args):
        bound = inspect.signature(ImagEMX2Camera.set_roi).bind(None, *args)
        self.settings["roi"] = dict(list(bound.arguments.items())[1:])

    def set_sensitivity(self, sensitivity):
        self.settings["sensitivity"] = sensitivity

    def get_conversion_factor(self):
        return 1.0, 0.0

    def acquire_n_frames(self, n):
        return [np.zeros((2, 2)) for _ in range(n)]

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def rig(monkeypatch, tmp_path):
    devices = {"Rb MotMaster": FakeSequencer(), "Rb HamCam": FakeCamera()}
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: devices[name],
    )
    parameters_file = tmp_path / "parameters.json"
    parameters_file.write_text(json.dumps({"cooling": SAVED}))
    devices["parameters_file"] = str(parameters_file)
    return devices


def run(rig, **args):
    measurement = run_local(
        TweezerBaseline, Scan(), parameters_file=rig["parameters_file"], **args
    )
    assert measurement.status == "completed", measurement.attrs["error"]
    return measurement


def test_saved_settings_reach_the_rb_motmaster_under_their_script_names(rig):
    run(rig)
    (sent,) = rig["Rb MotMaster"].runs
    for parameter, saved_name in SAVED_MOTMASTER_PARAMETERS.items():
        assert sent[parameter] == SAVED[saved_name], parameter
    assert sent["tImgCoolVCO1"] == sent["tImgCoolVCO2"] == SAVED["tImgCoolVCO"]


def test_a_dotted_override_beats_the_saved_value(rig):
    run(rig, **{"rb.tMolCoolVCO1": 7.5})
    (sent,) = rig["Rb MotMaster"].runs
    assert sent["tMolCoolVCO1"] == 7.5
    assert sent["tMolCoolVCO2"] == SAVED["tMolCoolVCO2"]


def test_the_recorded_motmaster_parameters_are_those_sent(rig):
    measurement = run(rig, **{"rb.tMolCoolVCO1": 7.5}, tweezer_vva_0=0.3)
    (sent,) = rig["Rb MotMaster"].runs
    assert measurement.constants["motmaster_parameters"] == sent


def test_camera_gets_the_roi_and_sensitivity_arguments(rig):
    run(rig, roi_x0=1, roi_width=2, roi_y0=3, roi_height=4, sensitivity=500)
    settings = rig["Rb HamCam"].settings
    assert settings["roi"] == {"x0": 1, "width": 2, "y0": 3, "height": 4}
    assert settings["sensitivity"] == 500
