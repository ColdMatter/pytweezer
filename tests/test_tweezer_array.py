import numpy as np
import pytest

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import photons_at_sites
from pytweezer.analysis.trap_sites import default_geometry, geometry_to_pixels
from pytweezer.drivers.motmaster import SIMULATED_SCRIPT_PARAMETERS
from pytweezer.experiment import Scan, motmaster, optimisers, run_local
from pytweezer.experiment.client import ManagerUnavailable
from pytweezer.experiments.load_tweezers import LoadTweezers, LoadTweezersMOTImage
from pytweezer.experiments.tweezer_array_measurement import TweezerArrayMeasurement
from pytweezer.optimisers.find_trap_sites import FindTrapSites
from pytweezer.optimisers.optimise_molasses_shims import OptimiseMolassesShims
from pytweezer.optimisers.optimise_tweezer_imaging import OptimiseTweezerImaging

#: The default 8x8, 5 µm grid at 2 px/µm, in the pixels of the camera's ROI.
SITES = [
    (round(y), round(x))
    for y, x in geometry_to_pixels(default_geometry(), 2.0, 0.0, 60.0, 60.0)
]
ROI_ORIGIN = (220, 110)
LOADING = 0.6
SURVIVAL = 0.9
FAST = {"shots": 10}


class ArrayBatch(TweezerArrayMeasurement):
    """One scored batch per point at the saved settings, as the optimisers measure."""

    def run_point(self):
        for name, value in self.score_batch(BackgroundDrop=0).items():
            self.record(name, value)


#: The arguments that make a run one point long.
ONE_POINT = {
    OptimiseTweezerImaging: {"n_calls": 1, "n_initial": 1},
    OptimiseMolassesShims: {"n_calls": 1, "n_initial": 1},
}


class FakeSequencer:
    def __init__(self):
        self.goes = []

    def close_rpc(self):
        pass

    def set_linked(self, value): ...
    def set_motmaster_experiment(self, script): ...
    def set_run_until_stopped(self, value): ...
    def set_iterations(self, value): ...
    def set_save_toggle(self, value): ...
    def set_trigger_mode(self, value): ...

    def get_params(self):
        return dict(SIMULATED_SCRIPT_PARAMETERS["RbTweezerBasic2026_2"])

    def start_motmaster_experiment(self, parameters):
        self.goes.append(parameters)


class FakeHamCam:
    """Images the array twice per shot; turning the tweezers off empties the second frame."""

    def __init__(self, sequencer):
        self.sequencer = sequencer
        self.headers = []
        self.rng = np.random.default_rng(1)
        y, x = np.mgrid[:128, :128]
        self.spots = np.array(
            [2000 * np.exp(-((y - sy) ** 2 + (x - sx) ** 2) / 4.5) for sy, sx in SITES]
        )

    def close_rpc(self):
        pass

    def set_roi(self, x0, width, y0, height):
        return self.roi_origin

    roi_origin = (ROI_ORIGIN[1], ROI_ORIGIN[0])

    def set_external_exposure_mode(self): ...
    def enable_em_gain(self, on): ...
    def enable_direct_em_gain(self, on): ...
    def set_trigger_source(self, source): ...
    def set_binning(self, binning): ...
    def set_gain(self, gain): ...
    def setup_acquisition(self, nframes, mode="sequence"): ...
    def start_acquisition(self): ...
    def stop_acquisition(self): ...
    def stop_live(self): ...

    def is_live(self):
        return False

    def _frame(self, occupied):
        atoms = np.tensordot(occupied, self.spots, axes=1)
        return self.rng.poisson(100 + atoms).astype(np.uint16)

    def acquire_n_frames(self, n, broadcast=False, header=None):
        self.headers.append(header)
        loaded = self.rng.random(len(SITES)) < LOADING
        if self.sequencer.goes[-1].get("tTweezerSetVVA0") == 0.0:
            kept = np.zeros(len(SITES), dtype=bool)
        else:
            kept = loaded & (self.rng.random(len(SITES)) < SURVIVAL)
        return self._frame(loaded), self._frame(kept)


class FakeORCA:
    """Images a Gaussian MOT of widths (6, 4) px on every shot; has no EM gain."""

    def __init__(self):
        self.headers = []
        y, x = np.mgrid[:60, :80]
        self.frame = (
            20
            + 200 * np.exp(-((x - 40) ** 2 / (2 * 6**2) + (y - 30) ** 2 / (2 * 4**2)))
        ).astype(np.uint16)

    def close_rpc(self):
        pass

    def set_external_exposure_mode(self): ...
    def set_trigger_source(self, source): ...
    def setup_acquisition(self, nframes, mode="sequence"): ...
    def start_acquisition(self): ...
    def stop_acquisition(self): ...
    def stop_live(self): ...

    def is_live(self):
        return False

    def acquire_n_frames(self, n, broadcast=False, header=None):
        self.headers.append(header)
        return np.stack([self.frame] * n)


class FakeRig:
    """Knows no array unless a test gives it one."""

    def __init__(self):
        self.array = None

    def close_rpc(self):
        pass

    def get_array(self):
        return self.array


def midpoint_threshold():
    """Halfway between the photons of an empty and a full site, in the fake camera's frames."""
    camera = FakeHamCam(None)
    sites = dict(enumerate(SITES))

    def mean_photons(occupied):
        frame = an.morphological_tophat_high_pass(
            camera._frame(occupied).astype(float), feature_size=10
        )
        return photons_at_sites([frame], sites, 5).mean()

    return (mean_photons(np.ones(len(SITES))) + mean_photons(np.zeros(len(SITES)))) / 2


SAVED = {
    key: 1.0 + index for index, key in enumerate(LoadTweezers.rb.saved.values())
} | {
    "trap_sites": np.array(SITES, dtype=float) + ROI_ORIGIN,
    "trap_threshold": midpoint_threshold(),
    "trap_window_size": 5,
    "trap_feature_size": 10,
    "trap_geometry": default_geometry(),
}


@pytest.fixture
def rig(monkeypatch):
    sequencer = FakeSequencer()
    rearrangement_rig = FakeRig()
    sequencer.rearrangement_rig = rearrangement_rig
    sequencer.camera = FakeHamCam(sequencer)
    devices = {
        "Rb MotMaster": sequencer,
        "Rb HamCam": sequencer.camera,
        "ORCA": FakeORCA(),
        "Rb Rearrangement Rig": rearrangement_rig,
    }
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: devices[name],
    )
    saved = dict(SAVED)
    sequencer.saved = saved
    for module in (motmaster, optimisers):
        monkeypatch.setattr(module, "get_saved_parameters", lambda: saved)
    return sequencer


def test_a_point_measures_the_loading_and_survival_put_in(rig):
    # Also checks the calibration: the wrong sites or threshold would miss these.
    measurement = run_local(ArrayBatch, **FAST)
    assert measurement.results["loading"][0] == pytest.approx(LOADING, abs=0.1)
    assert measurement.results["survival"][0] == pytest.approx(SURVIVAL, abs=0.05)
    assert measurement.results["fidelity"][0] > 0.99


def test_find_trap_sites_proposes_the_default_grid_where_the_atoms_are(rig):
    measurement = run_local(FindTrapSites, calibration_shots=30)

    assert measurement.status == "completed", measurement.attrs.get("error")
    proposed = measurement.constants["proposed.trap_sites"]
    assert np.abs(proposed - SAVED["trap_sites"]).max() < 0.5
    np.testing.assert_array_equal(
        measurement.constants["proposed.trap_geometry"], default_geometry()
    )
    assert measurement.constants["sites_from"] == "default"


def test_find_trap_sites_records_the_fitted_transform(rig):
    measurement = run_local(FindTrapSites, calibration_shots=30)
    results = measurement.results
    assert results["scale"][0] == pytest.approx(2.0, rel=0.01)
    assert results["angle"][0] == pytest.approx(0.0, abs=0.3)
    assert (results["x0"][0], results["y0"][0]) == (
        pytest.approx(60.0, abs=0.3),
        pytest.approx(60.0, abs=0.3),
    )


def test_find_trap_sites_keeps_the_rigs_order_of_its_array(rig):
    order = np.random.default_rng(2).permutation(len(SITES))
    x_um, y_um = default_geometry()[order].T
    rig.rearrangement_rig.array = {"sites": len(SITES), "x_um": x_um, "y_um": y_um}

    measurement = run_local(FindTrapSites, calibration_shots=30)

    proposed = measurement.constants["proposed.trap_sites"]
    assert np.abs(proposed - SAVED["trap_sites"][order]).max() < 0.5
    assert measurement.constants["sites_from"] == "rig"


def test_find_trap_sites_takes_its_shots_with_the_tweezers_off_in_the_background_drop(
    rig,
):
    run_local(FindTrapSites, calibration_shots=30)
    assert len(rig.goes) == 30
    assert all(go["BackgroundDrop"] == 1 for go in rig.goes)
    assert all(go["tTweezerSetVVA0"] == go["tTweezerSetVVA1"] == 0.0 for go in rig.goes)


def test_measurements_record_the_saved_sites_in_their_roi(rig):
    measurement = run_local(ArrayBatch, **FAST)

    assert measurement.status == "completed", measurement.attrs.get("error")
    np.testing.assert_array_equal(measurement.constants["trap_sites"], SITES)
    assert measurement.constants["threshold"] == SAVED["trap_threshold"]


def test_a_measurement_refuses_without_saved_sites(rig):
    del rig.saved["trap_sites"]
    measurement = run_local(ArrayBatch, **FAST)
    assert measurement.status == "failed"
    assert "FindTrapSites" in measurement.attrs["error"]


def test_a_measurement_refuses_sites_fitted_to_another_array(rig):
    x_um, y_um = default_geometry().T
    rig.rearrangement_rig.array = {
        "sites": len(SITES),
        "x_um": x_um + 0.5,
        "y_um": y_um,
    }
    measurement = run_local(ArrayBatch, **FAST)
    assert measurement.status == "failed"
    assert "another array" in measurement.attrs["error"]


def test_a_measurement_refuses_a_site_whose_window_leaves_the_roi(rig):
    sites = rig.saved["trap_sites"].copy()
    sites[5] = [ROI_ORIGIN[0] + 1, ROI_ORIGIN[1] + 100]
    rig.saved["trap_sites"] = sites
    measurement = run_local(ArrayBatch, **FAST)
    assert measurement.status == "failed"
    assert "[5]" in measurement.attrs["error"]


def test_every_shot_broadcasts_its_frames_tagged_with_the_task(rig):
    run_local(ArrayBatch, **FAST)
    assert rig.camera.headers
    assert all(header == {"rid": -1} for header in rig.camera.headers)


@pytest.mark.parametrize(
    ("experiment", "args"), [(LoadTweezers, {}), (ArrayBatch, FAST)]
)
def test_every_shot_runs_at_the_saved_parameters_unless_the_form_sets_one(
    rig, experiment, args
):
    run_local(experiment, **args, **{"rb.tMolCoolVCO1": 7.5})
    expected = {
        parameter: SAVED[key] for parameter, key in LoadTweezers.rb.saved.items()
    } | {"tMolCoolVCO1": 7.5}
    for go in rig.goes:
        assert {name: go[name] for name in expected} == expected


@pytest.mark.parametrize(
    ("experiment", "point_drop"),
    [(OptimiseTweezerImaging, (0, None)), (OptimiseMolassesShims, (1, 3))],
)
def test_each_point_keeps_the_tweezers_on_and_drops_as_its_measurement_needs(
    rig, experiment, point_drop
):
    run_local(experiment, **FAST, **ONE_POINT[experiment])
    assert len(rig.goes) == FAST["shots"]
    for go in rig.goes:
        assert "tTweezerSetVVA0" not in go
        assert (go["BackgroundDrop"], go.get("tDropDuration")) == point_drop


def test_each_evaluation_images_both_times_at_its_settings(rig):
    measurement = run_local(OptimiseTweezerImaging, **FAST, n_calls=2, n_initial=2)
    assert measurement.status == "completed", measurement.attrs["error"]
    evaluations = rig.goes
    for point in range(2):
        vco = measurement.results["imaging_vco"][point]
        vva = measurement.results["imaging_vva"][point]
        for go in evaluations[point * FAST["shots"] : (point + 1) * FAST["shots"]]:
            assert (go["tImgCoolVCO1"], go["tImgCoolVCO2"]) == (vco, vco)
            assert (go["tImgCoolVVA1"], go["tImgCoolVVA2"]) == (vva, vva)


@pytest.mark.parametrize(
    ("fidelity_floor", "expected"),
    [(0.9, "survival"), (1.0, "0.8 fidelity")],
)
def test_the_score_is_survival_only_above_the_fidelity_floor(
    rig, fidelity_floor, expected
):
    measurement = run_local(
        OptimiseTweezerImaging,
        **FAST,
        n_calls=1,
        n_initial=1,
        fidelity_floor=fidelity_floor,
    )
    results = measurement.results
    score = {
        "survival": results["survival"][0],
        "0.8 fidelity": 0.8 * results["fidelity"][0],
    }
    assert results["objective"][0] == pytest.approx(score[expected])


def test_each_molasses_evaluation_runs_at_its_shims(rig):
    measurement = run_local(OptimiseMolassesShims, **FAST, n_calls=2, n_initial=2)
    assert measurement.status == "completed", measurement.attrs["error"]
    evaluations = rig.goes
    for point in range(2):
        shims = tuple(
            measurement.results[setting][point]
            for setting in ("shim_x", "shim_y", "shim_z")
        )
        for go in evaluations[point * FAST["shots"] : (point + 1) * FAST["shots"]]:
            sent = (
                go["tMolassesShimXccValue"],
                go["tMolassesShimYccValue"],
                go["tMolassesShimZccValue"],
            )
            assert sent == shims


@pytest.mark.parametrize(
    ("loading_floor", "expected"),
    [(0.5, "floor plus recapture"), (0.9, "loading")],
)
def test_the_molasses_score_counts_recapture_only_once_the_array_loads(
    rig, loading_floor, expected
):
    measurement = run_local(
        OptimiseMolassesShims,
        **FAST,
        n_calls=1,
        n_initial=1,
        loading_floor=loading_floor,
    )
    results = measurement.results
    score = {
        "floor plus recapture": loading_floor + results["survival"][0],
        "loading": results["loading"][0],
    }
    assert results["objective"][0] == pytest.approx(score[expected])


def window_sums(frames, window_size=5):
    """Each frame's summed pixel values in a window at every site, ``(frames, sites)``."""
    half = window_size // 2
    return np.array(
        [
            [
                frame[y - half : y + half + 1, x - half : x + half + 1].sum()
                for y, x in SITES
            ]
            for frame in frames
        ]
    )


@pytest.mark.parametrize("cls", [OptimiseTweezerImaging, OptimiseMolassesShims])
def test_each_point_records_every_shots_counts_and_occupancy_at_every_site(rig, cls):
    measurement = run_local(cls, **FAST, **ONE_POINT[cls])

    counts = measurement.results["first_counts"]
    assert counts.shape == (1, FAST["shots"], len(SITES))
    assert measurement.results["second_counts"].shape == counts.shape
    occupied = measurement.results["first_occupied"]
    assert occupied.shape == counts.shape
    assert occupied.mean() == pytest.approx(LOADING, abs=0.1)


def test_a_measurements_counts_are_the_raw_pixel_sums_of_its_frames(rig):
    measurement = run_local(ArrayBatch, **FAST, save_raw_images=True)

    for i, name in enumerate(("first", "second")):
        np.testing.assert_array_equal(
            measurement.results[f"{name}_counts"][0],
            window_sums(measurement.results["hamcam_images"][0][:, i]),
        )


def test_an_optimiser_records_the_counts_of_each_evaluation(rig):
    measurement = run_local(OptimiseTweezerImaging, **FAST, n_calls=2, n_initial=2)

    assert measurement.results["first_counts"].shape == (2, FAST["shots"], len(SITES))


def test_load_tweezers_records_each_sites_raw_counts_and_occupancy(rig):
    measurement = run_local(LoadTweezers, Scan(repetitions=10))

    assert measurement.status == "completed", measurement.attrs.get("error")
    results = measurement.results
    for i, name in enumerate(("first", "second")):
        np.testing.assert_array_equal(
            results[f"{name}_counts"], window_sums(results["hamcam_images"][:, i])
        )
    assert results["first_occupied"].shape == (10, len(SITES))
    assert results["first_occupied"].mean() == pytest.approx(LOADING, abs=0.1)
    assert results["second_occupied"].mean() == pytest.approx(
        LOADING * SURVIVAL, abs=0.1
    )
    np.testing.assert_array_equal(measurement.constants["trap_sites"], SITES)


def test_load_tweezers_without_raw_images_records_no_frames(rig):
    measurement = run_local(LoadTweezers, save_raw_images=False)

    assert measurement.status == "completed", measurement.attrs.get("error")
    assert "hamcam_images" not in measurement.results
    assert "first_occupied" in measurement.results


def test_load_tweezers_without_saved_sites_records_frames_only(rig):
    del rig.saved["trap_sites"]
    measurement = run_local(LoadTweezers)

    assert measurement.status == "completed", measurement.attrs.get("error")
    assert "hamcam_images" in measurement.results
    assert "first_counts" not in measurement.results
    assert "trap_sites" not in measurement.constants


def test_dropping_raw_images_needs_trap_sites(rig):
    del rig.saved["trap_sites"]
    measurement = run_local(LoadTweezers, save_raw_images=False)
    assert measurement.status == "failed"
    assert "FindTrapSites" in measurement.attrs["error"]


def test_load_tweezers_needs_the_saved_parameters(rig, monkeypatch):
    def unreachable():
        raise ManagerUnavailable("not responding")

    monkeypatch.setattr(motmaster, "get_saved_parameters", unreachable)
    measurement = run_local(LoadTweezers)

    assert measurement.status == "failed"
    assert "saved parameters could not be read" in measurement.attrs["error"]


def test_load_tweezers_needs_every_saved_sequence_parameter(rig):
    del rig.saved["tImgCoolVCO"]
    measurement = run_local(LoadTweezers)

    assert measurement.status == "failed"
    assert "tImgCoolVCO" in measurement.attrs["error"]


@pytest.mark.parametrize("save_raw_images", [False, True])
def test_a_measurement_records_occupancy_and_frames_only_when_asked(
    rig, save_raw_images
):
    measurement = run_local(ArrayBatch, **FAST, save_raw_images=save_raw_images)

    assert measurement.results["first_occupied"].shape == (1, FAST["shots"], len(SITES))
    assert ("hamcam_images" in measurement.results) == save_raw_images
    if save_raw_images:
        assert measurement.results["hamcam_images"].shape == (
            1,
            FAST["shots"],
            2,
            128,
            128,
        )


def test_sites_are_placed_in_the_roi_the_camera_actually_used(rig):
    rig.camera.roi_origin = (ROI_ORIGIN[1] - 2, ROI_ORIGIN[0])
    measurement = run_local(ArrayBatch, **FAST)
    np.testing.assert_array_equal(
        measurement.constants["trap_sites"], np.array(SITES) + [0, 2]
    )


def test_load_tweezers_without_tweezer_counts_records_only_occupancy(rig):
    measurement = run_local(LoadTweezers, save_tweezer_counts=False)

    assert measurement.status == "completed", measurement.attrs.get("error")
    assert "first_counts" not in measurement.results
    assert "second_counts" not in measurement.results
    assert measurement.results["first_occupied"].shape == (1, len(SITES))


def test_a_measurement_without_tweezer_counts_still_scores_its_batches(rig):
    measurement = run_local(ArrayBatch, **FAST, save_tweezer_counts=False)

    assert measurement.status == "completed", measurement.attrs.get("error")
    assert "first_counts" not in measurement.results
    assert measurement.results["first_occupied"].shape == (1, FAST["shots"], len(SITES))
    assert measurement.results["loading"][0] == pytest.approx(LOADING, abs=0.1)


def test_tweezer_experiments_offer_counts_and_photon_histograms(rig):
    for cls in (LoadTweezers, ArrayBatch):
        assert cls.schema()["analyses"] == [
            "tweezer_histogram",
            "tweezer_photon_histogram",
        ]


def test_optimisers_offer_no_default_analyses(rig):
    for cls in (FindTrapSites, OptimiseTweezerImaging, OptimiseMolassesShims):
        assert cls.schema()["analyses"] == []


def test_load_tweezers_reports_the_loading_fraction_of_each_shot(rig):
    measurement = run_local(LoadTweezers, Scan(repetitions=10))

    results = measurement.results
    np.testing.assert_allclose(
        results["loading"], results["first_occupied"].mean(axis=1)
    )
    assert results["loading"].mean() == pytest.approx(LOADING, abs=0.1)


def test_mot_image_variant_records_orca_fit_from_the_same_shot(rig):
    measurement = run_local(LoadTweezersMOTImage, Scan(repetitions=2))

    assert measurement.status == "completed", measurement.attrs.get("error")
    results = measurement.results
    assert len(rig.goes) == 2
    assert results["orca_sigma_x"] == pytest.approx([6, 6], rel=0.03)
    assert results["orca_sigma_y"] == pytest.approx([4, 4], rel=0.03)
    assert results["hamcam_images"].shape[:2] == (2, 2)
    assert "orca_images" not in results
    assert measurement.constants["cloud_mass_amu"] == 86.909
    assert "first_occupied" in results


def test_load_tweezers_never_touches_the_orca(rig):
    measurement = run_local(LoadTweezers)

    assert measurement.status == "completed", measurement.attrs.get("error")
    assert not any(name.startswith("orca") for name in measurement.results)
    assert "cloud_mass_amu" not in measurement.constants


def test_mot_image_variant_offers_the_cloud_analyses_after_the_tweezer_ones(rig):
    assert LoadTweezersMOTImage.schema()["analyses"] == [
        "tweezer_histogram",
        "tweezer_photon_histogram",
        "cloud_fit",
        "cloud_temperature",
    ]
