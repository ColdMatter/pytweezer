"""The SLM tweezer-array experiment: trap weighting, target rendering, mask delivery."""

import numpy as np
import pytest

from pytweezer.experiments.slm_array import envelope_weights, render_target


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

    weights = envelope_weights(x_n + 40 + dx, y_n - 8 + dy)

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


class RecordingSLM:
    def __init__(self):
        self.shown = []

    def update_mask(self, mask):
        self.shown.append(mask)

    def close_rpc(self):
        pass


def test_mask_on_the_slm_is_the_one_recorded_and_broadcast(monkeypatch):
    pytest.importorskip("cupy")
    from pytweezer.experiment import run_local
    from pytweezer.experiments.slm_array import SlmTweezerArray

    slm = RecordingSLM()
    sent = {}
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device", lambda name, timeout: slm
    )
    monkeypatch.setattr(
        "pytweezer.servers.clients.ImageClient.send",
        lambda self, header, data, channel="", **_: sent.__setitem__(
            self.name + channel, data
        ),
    )

    measurement = run_local(SlmTweezerArray, rows=2, columns=3, iterations=5)

    assert measurement.status == "completed", measurement.attrs.get("error")
    (mask,) = slm.shown
    assert mask.dtype == np.uint8 and mask.shape == (1024, 1024)
    np.testing.assert_array_equal(measurement.results["phasemask"][0], mask)
    np.testing.assert_array_equal(sent["slm_phasemask"], mask)
    assert sent["slm_target"].ndim == 2
