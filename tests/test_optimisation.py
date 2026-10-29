"""MOT resonance, loading and shim optimisation, against a simulated MOT and
tweezer array that respond best to known settings."""

import json

import numpy as np
import pytest

from pytweezer.experiment.experiment_parameter_manager import ExpParameterManager
from pytweezer.helper_scripts import optimisation

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


SHIMS = {
    "MOT": ("tMOTShimXccValue", "tMOTShimYccValue", "tMOTShimZccValue"),
    "molasses": (
        "tMolassesShimXccValue",
        "tMolassesShimYccValue",
        "tMolassesShimZccValue",
    ),
}
#: The shim currents that load (MOT) or keep (molasses) the most atoms.
BEST_SHIMS = {"MOT": (6.0, -2.0, 0.3), "molasses": (4.4, -3.6, 0.2)}
SHIM_RANGES = ((0.0, 10.0), (-5.0, 5.0), (-1.0, 1.0))
SITES = {i: (10 + 10 * (i // 8), 10 + 10 * (i % 8)) for i in range(64)}
SHOTS_PER_POINT = 20


def best_shim_fraction(kind, shot):
    """1 at ``BEST_SHIMS[kind]``, falling off over half of each shim's range."""
    offsets = [
        (shot[name] - best) / (high - low)
        for name, best, (low, high) in zip(SHIMS[kind], BEST_SHIMS[kind], SHIM_RANGES)
    ]
    return float(np.exp(-sum(offset**2 for offset in offsets) / (2 * 0.5**2)))


def atoms_per_image(fraction, n_images, n_sites):
    total = round(fraction * n_images * n_sites)
    return [total // n_images + (k < total % n_images) for k in range(n_images)]


def loaded_fraction(shot):
    """Of all sites in all first images: the MOT shims set the loading, up to 80 %."""
    return 0.8 * best_shim_fraction("MOT", shot)


def surviving_fraction(shot):
    """Of the loaded atoms: the molasses shims set how many survive to the second image."""
    return best_shim_fraction("molasses", shot)


def loading_imaged(shot):
    """The loading probability the camera's images show for this shot."""
    loaded = atoms_per_image(loaded_fraction(shot), SHOTS_PER_POINT, len(SITES))
    return sum(loaded) / (SHOTS_PER_POINT * len(SITES))


def survival_imaged(shot):
    """The survival probability the camera's image pairs show for this shot."""
    loaded = atoms_per_image(loaded_fraction(shot), SHOTS_PER_POINT, len(SITES))
    return np.mean([round(surviving_fraction(shot) * n) / n for n in loaded])


class FakeTweezerCamera:
    """Images the last shot as pairs of tweezer-array frames: the first with
    ``loaded_fraction`` of the sites filled, the second with ``surviving_fraction``
    of those atoms still there, on a flat background with a little fixed noise."""

    def __init__(self, motmaster):
        self.motmaster = motmaster
        y, x = np.mgrid[0:90, 0:90]
        self.spots = {
            i: 200.0 * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / 2.0)
            for i, (sy, sx) in SITES.items()
        }
        self.noise = np.random.default_rng(0).normal(
            0.0, 1.0, (2 * SHOTS_PER_POINT, 90, 90)
        )

    def acquire_n_frames(self, nframes):
        shot = self.motmaster.shots[-1]
        loaded = atoms_per_image(loaded_fraction(shot), nframes // 2, len(SITES))
        frames = []
        for k, n_loaded in enumerate(loaded):
            atoms = [(i + 5 * k) % len(SITES) for i in range(n_loaded)]
            kept = atoms[: round(surviving_fraction(shot) * n_loaded)]
            for sites in (atoms, kept):
                frames.append(
                    50.0 + sum((self.spots[i] for i in sites), np.zeros((90, 90)))
                )
        return np.array(frames) + self.noise[:nframes]

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def tweezer_rig(tmp_path, monkeypatch):
    """A MOTMaster and tweezer camera, the saved parameters in a scratch file, and
    the array's analysis settings set in this session as after a calibration."""
    shims = {name: 0.0 for names in SHIMS.values() for name in names}
    saved_settings = {
        "motmaster_parameters": {
            **STARTING_PARAMETERS["motmaster_parameters"],
            **dict.fromkeys(
                ["tMolCoolVCO1", "tMolRepVCO1", "tMolCoolVVA1", "tMolRepVVA1"], 1.0
            ),
            **dict.fromkeys(
                ["tMolCoolVCO2", "tMolRepVCO2", "tMolCoolVVA2", "tMolRepVVA2"], 1.0
            ),
            "tImgCoolVCO": 4.0,
            "tImgCoolVVA": 1.0,
            **shims,
        },
        "emccd_parameters": {"conversion_factor": 0.01, "conversion_offset": 0.0},
    }
    parameters_file = tmp_path / "parameters.json"
    parameters_file.write_text(json.dumps(saved_settings))
    parameters = ExpParameterManager(parameters_file)
    for name, value in {
        "site_positions": SITES,
        "feature_size": 5,
        "sum_window": 3,
        "threshold": 5.0,
    }.items():
        parameters.add_parameter(name, value)
    monkeypatch.setattr(optimisation, "ExpParams", parameters)
    monkeypatch.setattr(
        "pytweezer.analysis.analysis.ExpParameterManager", lambda: parameters
    )
    motmaster = FakeMotMaster()
    return motmaster, FakeTweezerCamera(motmaster), parameters


@pytest.mark.parametrize(
    "kind, optimise, score",
    [
        ("MOT", optimisation.OptimiseMOTShimParameters, loading_imaged),
        ("molasses", optimisation.OptimiseMolassesShimParameters, survival_imaged),
    ],
    ids=["MOT", "molasses"],
)
def test_shim_optimisation_keeps_the_best_shims_it_tried(
    tweezer_rig, kind, optimise, score
):
    motmaster, camera, parameters = tweezer_rig
    if kind == "molasses":
        # Survival is only scored once loading reaches 50 %.
        for name, best in zip(SHIMS["MOT"], BEST_SHIMS["MOT"]):
            parameters.set_parameter(name, best)
    optimise(camera, motmaster, n_it_per_exp=SHOTS_PER_POINT, n_calls=6, init_calls=4)

    kept = {name: parameters.get_parameter(name) for name in SHIMS[kind]}
    (kept_shot,) = [
        shot
        for shot in motmaster.shots
        if {name: shot[name] for name in SHIMS[kind]} == pytest.approx(kept)
    ]
    assert score(kept_shot) == max(score(shot) for shot in motmaster.shots)
