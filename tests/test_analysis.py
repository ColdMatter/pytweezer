"""The analysis code in ``pytweezer/analysis/``."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pytweezer.analysis import analysis as an


class FixedScorer:
    """Stands in for :class:`~pytweezer.analysis.analysis.SiteScorer`: returns given photons."""

    def __init__(self, photons):
        self.photons = np.asarray(photons, dtype=float)

    def score_stack(self, images):
        return self.photons


@pytest.mark.parametrize(
    "loading_statistics",
    [
        an.get_array_loading_statistics,
        lambda *args, **kwargs: an.get_array_loading_statistics_grid(
            *args, grid_shape=(1, 2), **kwargs
        ),
    ],
    ids=["sites", "grid"],
)
def test_well_separated_counts_at_a_fixed_threshold_have_fidelity_near_one(
    loading_statistics,
):
    photons = [[-1.0, 99.0], [1.0, 101.0]]
    *_, fidelity = loading_statistics(
        [None, None],
        {0: (0, 0), 1: (0, 1)},
        threshold=50,
        scorer=FixedScorer(photons),
        threshold_detection=False,
        show_histogram=False,
        verbose=False,
    )
    assert fidelity == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "detect",
    [
        lambda frame: an.detect_trap_sites(frame, atom_number=2),
        lambda frame: an.detect_trap_sites_grid(frame, grid_shape=(1, 2)),
    ],
    ids=["sites", "grid"],
)
def test_trap_site_search_gives_up_on_a_uint16_frame_with_too_few_sites(detect):
    frame = np.array([[0, 50]], dtype=np.uint16)
    thread = threading.Thread(target=detect, args=(frame,), daemon=True)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive(), (
        "the threshold wrapped round and the search never ended"
    )


def test_bright_points_are_placed_at_the_nearest_pixel_to_their_centre_of_mass():
    # Centre of mass x = (1 * 1 + 2 * 99) / 100 = 1.99
    centres = an.detect_bright_points(np.array([[0, 1, 99]]), threshold=0)
    assert centres.tolist() == [[0, 2]]


def _elongated_gaussian_image():
    """A cloud 30 px wide in x and 70 px in y, with seeded noise."""
    y, x = np.mgrid[:300, :300]
    model = an.gaussian2D(x, y, A=100, x0=150, y0=140, sx=30, sy=70, theta=0, offset=10)
    noisy = model + np.random.default_rng(0).normal(0, 1, model.shape)
    return x, y, model, noisy


def test_fitted_gaussian_parameters_redraw_the_image():
    x, y, model, image = _elongated_gaussian_image()
    fit = an.fit_gaussian(image)
    names = ("A", "x0", "y0", "sx", "sy", "theta", "offset")
    redrawn = an.gaussian2D(x, y, *(fit[name] for name in names))
    np.testing.assert_allclose(redrawn, model, atol=1.0)


def test_fitted_gaussian_width_errors_follow_their_widths():
    *_, image = _elongated_gaussian_image()
    fit, fit_transposed = an.fit_gaussian(image), an.fit_gaussian(image.T)
    for name in ("sx", "sy", "sx_err", "sy_err"):
        assert fit[name] == pytest.approx(fit_transposed[name], rel=1e-3), name


class _StreamEnded(Exception):
    pass


class OneMessageStream:
    """Stands in for an ``ImageClient``: delivers one message, then ends ``run()``."""

    def __init__(self, head, image):
        self.inbox = [("cam", head, image)]
        self.sent = []

    def recv(self):
        if not self.inbox:
            raise _StreamEnded
        return self.inbox.pop()

    def send(self, head, data, channel=""):
        self.sent.append((head, data))


def _run_filter(filter_class, stream, **properties):
    """Run a streaming filter over one message; returns what it sent."""
    process = filter_class.__new__(filter_class)
    process._props = SimpleNamespace(
        get=lambda name, default: properties.get(name, default)
    )
    process.imageq = stream
    with pytest.raises(_StreamEnded):
        process.run()
    return stream.sent


@pytest.mark.parametrize(
    "angle, offset, rotated",
    [(0, [10, 0], [10, 0]), (90, [10, 0], [0, -10]), (90, [0, 10], [10, 0])],
)
def test_rotating_an_image_rotates_its_offset_the_same_way(angle, offset, rotated):
    from pytweezer.analysis.rotate_image import ImageRotate

    stream = OneMessageStream({"_offset": offset}, np.zeros((4, 4)))
    [(head, _image)] = _run_filter(ImageRotate, stream, angle=angle)
    np.testing.assert_allclose(head["_offset"], rotated, atol=1e-12)
