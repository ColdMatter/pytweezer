"""MOT resonance and loading scans, against a simulated MOT whose brightness
peaks at known settings."""

import json

import numpy as np
import pytest

from pytweezer import optimisation
from pytweezer.experiment.experiment_parameter_manager import ExpParameterManager

COOL_VCO_GRID = np.linspace(3.0, 5.0, 20)
REP_VCO_GRID = np.linspace(0.0, 5.0, 20)
MOT_SCANS = {
    "tMOTLoadCoolVCO": np.linspace(3.0, 5.0, 20),
    "tMOTLoadCoolVVA": np.linspace(0.4, 1.5, 20),
    "tMOTLoadRepVCO": np.linspace(0.0, 5.0, 20),
    "tMOTLoadRepVVA": np.linspace(0.4, 1.0, 20),
}
BRIGHTEST = {
    "tImgCoolVCO1": COOL_VCO_GRID[12],
    "tImgRepVCO1": REP_VCO_GRID[7],
    "tMOTLoadCoolVCO": MOT_SCANS["tMOTLoadCoolVCO"][5],
    "tMOTLoadCoolVVA": MOT_SCANS["tMOTLoadCoolVVA"][14],
    "tMOTLoadRepVCO": MOT_SCANS["tMOTLoadRepVCO"][9],
    "tMOTLoadRepVVA": MOT_SCANS["tMOTLoadRepVVA"][3],
}
#: Saved before the scans: the first value of each scan, away from every peak.
STARTING_PARAMETERS = {
    "motmaster_parameters": {
        name: float(values[0]) for name, values in MOT_SCANS.items()
    },
    "misc_parameters": {"cool_vco_resonance": 3.0, "rep_vco_resonance": 0.0},
}
MOT_CENTRE = (70, 60)


class FakeMotMaster:
    """Records the parameters of every shot."""

    def __init__(self):
        self.shots = []

    def start_motmaster_experiment(self, parameters):
        self.shots.append(dict(parameters))

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeMOTCamera:
    """Images the last shot: stray light that grows with every setting, plus,
    unless the MOT coils were off (``tMOTccValue`` 0), a Gaussian MOT that each
    setting brightens most at its ``BRIGHTEST`` value. The stray light outgrows
    any MOT peak, so only background-subtracted counts find the peaks."""

    def __init__(self, motmaster):
        self.motmaster = motmaster
        y, x = np.mgrid[0:128, 0:128]
        self.mot = np.exp(
            -((x - MOT_CENTRE[0]) ** 2 + (y - MOT_CENTRE[1]) ** 2) / (2 * 8.0**2)
        )

    def acquire_n_frames(self, nframes):
        shot = self.motmaster.shots[-1]
        stray = 1000.0 * sum(shot.get(name, 0.0) for name in BRIGHTEST)
        brightness = 1 + sum(
            9 / (1 + ((shot[name] - peak) / 0.05) ** 2)
            for name, peak in BRIGHTEST.items()
            if name in shot
        )
        mot_on = shot.get("tMOTccValue", 1.0) != 0
        image = stray + mot_on * 100.0 * brightness * self.mot
        return np.array([image] * nframes)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A MOTMaster and camera, with the saved parameters in a scratch file."""
    parameters_file = tmp_path / "parameters.json"
    parameters_file.write_text(json.dumps(STARTING_PARAMETERS))
    monkeypatch.setattr(optimisation, "ExpParams", ExpParameterManager(parameters_file))
    motmaster = FakeMotMaster()
    return motmaster, FakeMOTCamera(motmaster), parameters_file


def saved(parameters_file, name):
    return ExpParameterManager(parameters_file).get_parameter(name)


@pytest.mark.parametrize(
    "parameter, stored_as",
    [("tImgCoolVCO1", "cool_vco_resonance"), ("tImgRepVCO1", "rep_vco_resonance")],
    ids=["cooling", "repump"],
)
def test_detect_resonances_saves_the_vco_with_the_most_mot_counts(
    rig, parameter, stored_as
):
    motmaster, camera, parameters_file = rig
    optimisation.DetectResonances(camera, motmaster, img_it=1, bg_it=1)
    assert saved(parameters_file, stored_as) == pytest.approx(BRIGHTEST[parameter])


@pytest.mark.parametrize("parameter", MOT_SCANS)
def test_mot_number_scan_saves_each_parameters_brightest_value(rig, parameter):
    motmaster, camera, parameters_file = rig
    optimisation.OptimiseMOTNumber(camera, motmaster, img_it=1, bg_it=1)
    assert saved(parameters_file, parameter) == pytest.approx(BRIGHTEST[parameter])


def test_each_mot_number_scan_runs_at_the_earlier_scans_optima(rig):
    motmaster, camera, parameters_file = rig
    before = STARTING_PARAMETERS["motmaster_parameters"]
    optimisation.OptimiseMOTNumber(camera, motmaster, img_it=1, bg_it=1)

    # Two shots locate the MOT, then each scan takes a background and an image per value.
    scans = np.split(np.array(motmaster.shots[2:]), len(MOT_SCANS))
    for index, (scanned, values) in enumerate(MOT_SCANS.items()):
        shots = scans[index]
        np.testing.assert_allclose([shot[scanned] for shot in shots[::2]], values)
        for k, other in enumerate(MOT_SCANS):
            expected = BRIGHTEST[other] if k < index else before[other]
            if other != scanned:
                sent = [shot[other] for shot in shots]
                assert sent == pytest.approx([expected] * len(shots)), (scanned, other)
