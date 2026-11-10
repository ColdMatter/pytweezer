import time

import numpy as np
import pytest

from pytweezer.analysis.array_loading import occupied_sites
from pytweezer.analysis.trap_sites import default_geometry, geometry_to_pixels
from pytweezer.experiment import ListAxis, Scan, motmaster, run_local
from pytweezer.experiments.rearrange_tweezers import (
    RearrangeTweezers,
    apply_affine,
    fit_affine,
)
from tests.test_tweezer_array import (
    ROI_ORIGIN,
    SAVED,
    SITES,
    FakeHamCam,
    FakeSequencer,
)

TARGET = default_geometry(3, 3, 5.0)


def camera_pixels(geometry):
    """``(y, x)`` pixels of ``geometry`` on the fake HamCam, under the map that places ``SITES``."""
    return np.rint(geometry_to_pixels(geometry, 2.0, 0.0, 60.0, 60.0)).astype(int)


def lit_frame(pixels):
    """A HamCam frame with an atom at each ``(y, x)`` of ``pixels``."""
    return FakeHamCam(None, sites=pixels)._frame(np.ones(len(pixels)))


class FakeRearrangementRig:
    """Has the default array on the SLM; each shot lights the frames at ``lit``."""

    def __init__(self):
        x_um, y_um = default_geometry().T
        self.array = {"sites": len(SITES), "x_um": x_um, "y_um": y_um}
        self.target = TARGET.copy()
        targets = camera_pixels(TARGET)
        self.lit = (SITES[:10], targets[:5], targets[:4])
        self.prepared = None
        self.rearrange_times = []

    def close_rpc(self):
        pass

    def get_array(self):
        return self.array

    def prepare_rearrangement(self, **kwargs):
        self.prepared = kwargs
        x_um, y_um = self.target.T
        return {
            "x_um": x_um,
            "y_um": y_um,
            "weights": np.ones(len(self.target)),
            "uniformity": 0.99,
            "min_max_ratio": 0.98,
            "mse": 1e-3,
        }

    def rearrange(self, *, d0_um, profile, header):
        self.rearrange_times.append(time.monotonic())
        frames = np.array([lit_frame(pixels) for pixels in self.lit])
        occupied = occupied_sites(
            frames[0],
            self.prepared["occupancy_sites"],
            window_size=self.prepared["window_size"],
            feature_size=self.prepared["feature_size"],
            threshold=self.prepared["threshold_photons"],
        )
        self.destination = np.where(occupied, np.cumsum(occupied) - 1, -1)
        self.destination[self.destination >= len(self.target)] = -1
        return {
            "frames": frames,
            "occupied": occupied,
            "destination": self.destination,
            "n_frames": 12,
            "occupancy_ms": 1.0,
            "calculation_and_upload_ms": 20.0,
            "total_rearrangement_ms": 25.0,
        }


@pytest.fixture
def rig(monkeypatch):
    sequencer = FakeSequencer()
    sequencer.camera = FakeHamCam(sequencer)
    sequencer.rearrangement_rig = FakeRearrangementRig()
    devices = {
        "Rb MotMaster": sequencer,
        "Rb HamCam": sequencer.camera,
        "Rb Rearrangement Rig": sequencer.rearrangement_rig,
    }
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: devices[name],
    )
    saved = dict(SAVED)
    sequencer.saved = saved
    monkeypatch.setattr(motmaster, "get_saved_parameters", lambda: saved)
    return sequencer


def test_fit_affine_recovers_a_known_map():
    angle = np.radians(3)
    rotation = 2 * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    offset = np.array([60.0, 40.0])
    from_xy = np.random.default_rng(0).uniform(-20, 20, (12, 2))

    matrix, residual = fit_affine(from_xy, from_xy @ rotation.T + offset)

    assert residual < 1e-9
    elsewhere = np.array([[100.0, -50.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        apply_affine(matrix, elsewhere), elsewhere @ rotation.T + offset
    )


def test_a_shot_records_occupancy_destination_fill_and_timings(rig):
    measurement = run_local(RearrangeTweezers)

    assert measurement.status == "completed", measurement.attrs.get("error")
    results = measurement.results
    assert results["occupied"][0][:10].all()
    assert results["loading"][0] == 10 / 64
    np.testing.assert_array_equal(
        results["destination"][0], rig.rearrangement_rig.destination
    )
    np.testing.assert_array_equal(
        results["target_filled_1"][0], [True] * 5 + [False] * 4
    )
    assert results["fill_fraction_2"][0] == 4 / 9
    assert results["n_frames"][0] == 12
    assert results["total_rearrangement_ms"][0] == 25.0
    assert results["rearrangement_images"].shape == (1, 3, 128, 128)
    assert results["initial_counts"].shape == (1, 64)
    assert results["target_counts"].shape == (1, 2, 9)


def test_motmaster_fires_arm_delay_after_the_rig_is_asked_to_rearrange(rig):
    measurement = run_local(RearrangeTweezers, arm_delay=0.05)

    assert measurement.status == "completed", measurement.attrs.get("error")
    (asked,) = rig.rearrangement_rig.rearrange_times
    (fired,) = rig.go_times
    assert fired - asked >= 0.04


def test_target_sites_are_placed_by_the_fitted_map(rig):
    measurement = run_local(RearrangeTweezers)

    np.testing.assert_array_equal(
        measurement.constants["target_sites"], camera_pixels(TARGET)
    )
    assert measurement.constants["affine_residual_px"] < 0.5


def test_the_initial_array_and_its_map_to_the_camera_are_recorded(rig):
    measurement = run_local(RearrangeTweezers)

    constants = measurement.constants
    array = rig.rearrangement_rig.array
    np.testing.assert_array_equal(constants["array_x_um"], array["x_um"])
    np.testing.assert_array_equal(constants["array_y_um"], array["y_um"])
    array_um = np.column_stack([constants["array_x_um"], constants["array_y_um"]])
    np.testing.assert_allclose(
        apply_affine(constants["um_to_px"], array_um)[:, ::-1], SITES, atol=0.5
    )


def test_it_refuses_without_an_array_on_the_slm(rig):
    rig.rearrangement_rig.array = None
    measurement = run_local(RearrangeTweezers)

    assert measurement.status == "failed"
    assert "no array on the SLM" in measurement.attrs["error"]
    assert rig.goes == []


def test_it_refuses_saved_sites_in_another_order_than_the_array(rig):
    order = np.random.default_rng(2).permutation(len(SITES))
    rig.saved["trap_sites"] = rig.saved["trap_sites"][order]
    measurement = run_local(RearrangeTweezers)

    assert measurement.status == "failed"
    assert "do not fit the rig's array" in measurement.attrs["error"]


def test_it_refuses_an_array_in_one_row_which_cannot_place_a_2d_target(rig):
    # Sites that fit the row exactly, so only the map's rank can refuse it.
    row_um = np.column_stack([5.0 * np.arange(8), np.zeros(8)])
    rig.rearrangement_rig.array = {
        "sites": 8,
        "x_um": row_um[:, 0],
        "y_um": row_um[:, 1],
    }
    rig.saved["trap_geometry"] = row_um
    row_px = np.column_stack([np.full(8, 60.0), 20.0 + 10.0 * np.arange(8)])
    rig.saved["trap_sites"] = row_px + ROI_ORIGIN
    measurement = run_local(RearrangeTweezers)

    assert measurement.status == "failed"
    assert "collinear" in measurement.attrs["error"]
    assert rig.goes == []


def test_it_refuses_a_target_whose_window_leaves_the_roi(rig):
    rig.rearrangement_rig.target[4, 0] = 200
    measurement = run_local(RearrangeTweezers)

    assert measurement.status == "failed"
    assert "target sites [4] leave the camera's ROI" in measurement.attrs["error"]


@pytest.mark.parametrize(
    ("argument", "values"),
    [("target_spacing", [4e-6, 5e-6]), ("target_kind", ["square", "kagome"])],
)
def test_the_target_array_cannot_be_scanned(rig, argument, values):
    scan = Scan(axes=[ListAxis(argument=argument, values=values)])
    with pytest.raises(ValueError, match=f"{argument} cannot be scanned"):
        run_local(RearrangeTweezers, scan)
    assert rig.goes == []


def test_a_scanned_gain_is_resent_before_each_shot_that_changes_it(rig):
    scan = Scan(axes=[ListAxis(argument="hamcam_gain", values=[600, 600, 300])])
    measurement = run_local(RearrangeTweezers, scan)

    assert measurement.status == "completed", measurement.attrs.get("error")
    gains = [args for name, args in rig.camera.calls if name == "set_gain"]
    assert gains == [(600,), (300,)]


def test_a_failed_go_fails_the_task(rig):
    rig.fail = True
    measurement = run_local(RearrangeTweezers)

    assert measurement.status == "failed"
    assert "MOTMaster refused" in measurement.attrs["error"]


def test_the_rig_is_given_the_saved_sites_threshold_and_target(rig):
    run_local(RearrangeTweezers)

    prepared = rig.rearrangement_rig.prepared
    np.testing.assert_array_equal(prepared["occupancy_sites"], SITES)
    assert np.shape(prepared["occupancy_sites"]) == (64, 2)
    assert prepared["threshold_photons"] == SAVED["trap_threshold"]
    assert prepared["kind"] == "kagome"
