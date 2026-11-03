"""The SLM tweezer-array experiment: trap weighting, target rendering, mask delivery."""

import numpy as np
import pytest

from pytweezer.experiments import slm_tweezer_array
from pytweezer.experiments.slm_tweezer_array import render_target
from pytweezer.helper_scripts.array_hologram import envelope, envelope_weights

#: The lab's SLM saved parameters, in SI.
SAVED = {
    "slm_wavelength": 0.852e-6,
    "slm_focal_length": 17.3e-3,
    "slm_pixel_pitch": 17e-6,
    "slm_resolution": [1024, 1024],
    "slm_input_beam_waist": 16e-3,
    "slm_zernike_coefficients": [
        [5, 1.195],
        [6, 0.725],
        [7, 0.970],
        [8, 0.478],
        [9, -1.091],
        [10, 0.303],
        [11, 0.021],
        [12, 0.072],
        [13, 0.049],
    ],
    "slm_envelope_amplitude": -2.14197604e03,
    "slm_envelope_offset": 2.14239505e03,
    "slm_envelope_centre": [28.8556150e-6, -5.9008040e-6],
    "slm_envelope_sigma": [2.95002910e-3, 2.64281503e-3],
    "slm_blaze_x": 48e-6,
    "slm_blaze_y": 4e-6,
    "slm_fresnel_focal_length": 1.125,
    "slm_pupil_radius": 8.704e-3,
    "slm_pupil_dump_x": 100e-6,
    "slm_pupil_dump_y": 0.0,
}


def test_envelope_weights_match_the_notebook_calibration_in_array_coordinates():
    # 261029_test.ipynb evaluates its fit in array coordinates, with the centre
    # moved against the array shift (dx, dy) on top of a (40, -8) um blaze.
    dx, dy = 8.0, 12.0
    A, x0, y0, sx, sy, offset = (
        -2.14197604e03,
        -1.11443850e01 - dx,
        2.09919600e00 - dy,
        2.95002910e03,
        2.64281503e03,
        2.14239505e03,
    )
    x_n, y_n = np.meshgrid(np.linspace(-35, 35, 15), np.linspace(-35, 35, 15))
    notebook = A * np.exp(
        -((x_n - x0) ** 2) / (2 * sx**2) - (y_n - y0) ** 2 / (2 * sy**2)
    )
    notebook += offset

    weights = envelope_weights(envelope(SAVED), x_n + 40 + dx, y_n - 8 + dy)

    np.testing.assert_allclose(weights, notebook, rtol=1e-9)


@pytest.mark.parametrize("x_um, y_um", [(0.0, 0.0), (12.3, -4.7), (-50.05, 31.9)])
def test_rendered_spot_peaks_where_the_returned_origin_places_the_trap(x_um, y_um):
    pixel_um = 0.25
    image, (x0_um, y0_um) = render_target(
        np.array([x_um]),
        np.array([y_um]),
        np.array([1.0]),
        1.0,
        pixel_um,
        margin_um=5.0,
    )

    row, column = np.unravel_index(np.argmax(image), image.shape)
    assert x0_um + (column + 0.5) * pixel_um == pytest.approx(x_um, abs=pixel_um / 2)
    assert y0_um + (row + 0.5) * pixel_um == pytest.approx(y_um, abs=pixel_um / 2)


def test_each_trap_is_drawn_at_its_weight():
    x_um = np.array([-10.0, 0.0, 10.0])
    weights = np.array([0.2, 0.5, 1.0])
    image, (x0_um, _) = render_target(
        x_um, np.zeros(3), weights, 1.0, 0.1, margin_um=5.0
    )

    columns = np.round((x_um - x0_um) / 0.1 - 0.5).astype(int)
    np.testing.assert_allclose(image.max(axis=0)[columns], weights, rtol=1e-2)


class RecordingRig:
    """Stands in for the rearrangement rig: records what it was asked to make."""

    def __init__(self):
        self.made = []

    def make_array(self, **array):
        self.made.append(array)
        rows, columns = array["rows"], array["columns"]
        y_um, x_um = np.mgrid[0:rows, 0:columns] * array["spacing_um"]
        return {
            "mask": np.full((1024, 1024), len(self.made), dtype=np.uint8),
            "x_um": x_um.ravel(),
            "y_um": y_um.ravel(),
            "weights": np.ones(rows * columns),
            "focal_pixel_um": 0.5,
            "uniformity": 0.9,
            "min_max_ratio": 0.8,
            "mse": 0.01,
        }

    def close_rpc(self):
        pass


@pytest.fixture
def rig_and_streams(monkeypatch):
    saved = dict(SAVED)
    monkeypatch.setattr(slm_tweezer_array, "get_saved_parameters", lambda: saved)
    monkeypatch.setattr(
        "pytweezer.experiment.client.get_saved_parameters", lambda: saved
    )
    rig = RecordingRig()
    rig.saved = saved
    sent = {}
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device", lambda name, timeout: rig
    )
    monkeypatch.setattr(
        "pytweezer.servers.clients.ImageClient.send",
        lambda self, header, data, channel="", **_: sent.__setitem__(
            self.name + channel, data
        ),
    )
    return rig, sent


def test_rig_is_asked_for_the_array_in_the_form(rig_and_streams):
    from pytweezer.experiment import run_local
    from pytweezer.experiments.slm_tweezer_array import SlmTweezerArray

    rig, _ = rig_and_streams

    measurement = run_local(
        SlmTweezerArray, rows=2, columns=3, spacing=4e-6, iterations=5
    )

    assert measurement.status == "completed", measurement.attrs.get("error")
    ((array),) = rig.made
    assert (array["rows"], array["columns"], array["iterations"]) == (2, 3, 5)
    assert array["spacing_um"] == pytest.approx(4.0)


def test_mask_the_rig_made_is_the_one_recorded_and_broadcast(rig_and_streams):
    from pytweezer.experiment import run_local
    from pytweezer.experiments.slm_tweezer_array import SlmTweezerArray

    _, sent = rig_and_streams

    measurement = run_local(SlmTweezerArray, rows=2, columns=3, iterations=5)

    assert measurement.status == "completed", measurement.attrs.get("error")
    mask = np.full((1024, 1024), 1, dtype=np.uint8)
    np.testing.assert_array_equal(measurement.results["phasemask"][0], mask)
    np.testing.assert_array_equal(sent["slm_phasemask"], mask)
    assert sent["slm_target"].ndim == 2


def test_the_optics_come_from_the_saved_parameters_unless_the_form_sets_them(
    rig_and_streams,
):
    from pytweezer.experiment import run_local
    from pytweezer.experiments.slm_tweezer_array import SlmTweezerArray

    rig, _ = rig_and_streams

    measurement = run_local(SlmTweezerArray, iterations=5, blaze_x=50e-6)

    assert measurement.status == "completed", measurement.attrs.get("error")
    ((array),) = rig.made
    assert array["blaze_um"] == pytest.approx((50.0, 4.0))
    assert array["fresnel_focal_length_mm"] == pytest.approx(1125.0)
    assert array["pupil_radius_um"] == pytest.approx(8704.0)
    assert array["generator_settings"]["wavelength_um"] == pytest.approx(0.852)
    assert array["generator_settings"]["zernike_coeff_dict"][9] == pytest.approx(-1.091)
    assert array["envelope"]["centre_um"] == pytest.approx((28.855615, -5.900804))
    assert measurement.arguments["blaze_y"] == pytest.approx(4e-6)


def test_an_uncompensated_array_is_made_without_the_envelope(rig_and_streams):
    from pytweezer.experiment import run_local
    from pytweezer.experiments.slm_tweezer_array import SlmTweezerArray

    rig, _ = rig_and_streams

    run_local(SlmTweezerArray, iterations=5, compensate_envelope=False)

    assert rig.made[0]["envelope"] is None


def test_a_missing_saved_optics_parameter_fails_the_run(rig_and_streams):
    from pytweezer.experiment import run_local
    from pytweezer.experiments.slm_tweezer_array import SlmTweezerArray

    rig, _ = rig_and_streams
    del rig.saved["slm_envelope_sigma"]

    measurement = run_local(SlmTweezerArray, iterations=5)

    assert measurement.status == "failed"
    assert "slm_envelope_sigma" in measurement.attrs["error"]
