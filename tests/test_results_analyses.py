import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pytweezer.analysis.cloud_fit import fit_frames
from pytweezer.analysis.cloud_results import cloud_sizes
from pytweezer.experiment import ListAxis, Number, Scan, run_local
from pytweezer.experiment.archive import ArchiveUnavailable, ResultsArchive
from pytweezer.experiment.motmaster import MotMasterNumber
from pytweezer.GUI.experiments import analyses
from pytweezer.GUI.experiments.analyses import (
    TweezerHistogramView,
    declared_analyses,
)
from pytweezer.GUI.experiments.cloud_views import (
    CloudFitView,
    CloudTemperatureView,
    holds_seconds,
)
from tests.helpers import until, write_run
from tests.test_analysis import SITE_CENTRES
from tests.test_cloud_results import tof_file
from tests.test_tweezer_histogram import ImagesArray


def frames_file(root, rid=1, label="", scan=None):
    relative = f"2026/11/02/{rid:06d}_ImagesArray.h5"
    run_local(
        ImagesArray, scan or Scan(repetitions=40), path=root / relative, label=label
    )
    return relative


def open_view(archive, relative):
    view = TweezerHistogramView()
    view.sites.setValue(len(SITE_CENTRES))
    view.detection_step.setValue(10)
    view.set_measurement(archive, relative, archive.load(relative, results=[]))
    return view


def test_view_draws_the_histogram_of_a_frames_file(qapp, tmp_path):
    archive = ResultsArchive(tmp_path)
    view = open_view(archive, frames_file(tmp_path))

    assert until(lambda: "loading" in view.plots.readout.text(), qapp=qapp)
    assert not view.sites.isHidden()
    assert "imaging survival 0.0 %" in view.plots.readout.text()
    assert view.message.text() == ""


def test_choosing_a_scan_value_analyses_only_its_shots(qapp, tmp_path):
    archive = ResultsArchive(tmp_path)
    scan = Scan(repetitions=40)
    view = open_view(archive, frames_file(tmp_path, scan=scan))
    until(lambda: "40 shots" in view.plots.readout.text(), qapp=qapp)

    view.shots_from.setCurrentIndex(1)

    assert until(lambda: "1 shots" in view.plots.readout.text(), qapp=qapp)


def test_a_stale_result_is_discarded(qapp, tmp_path, monkeypatch):
    archive = ResultsArchive(tmp_path)
    first, second = (
        frames_file(tmp_path, 1, "first"),
        frames_file(tmp_path, 2, "second"),
    )
    release = threading.Event()
    compute = analyses.tweezer_histogram

    def held_for_the_first(measurement, settings):
        if measurement.attrs["label"] == "first":
            release.wait(5)
            return {"shots": 0, "sites": 0, "error": "from the first file"}, None
        return compute(measurement, settings)

    monkeypatch.setattr(analyses, "tweezer_histogram", held_for_the_first)
    view = open_view(archive, first)
    drawn = []
    show_result = view.plots.show_result
    view.plots.show_result = lambda head, photons: (
        drawn.append(head),
        show_result(head, photons),
    )
    view.set_measurement(archive, second, archive.load(second, results=[]))

    release.set()

    assert until(lambda: "loading" in view.plots.readout.text(), qapp=qapp)
    assert all(head.get("error") != "from the first file" for head in drawn)


@pytest.mark.parametrize(
    ("failure", "shown"),
    [
        (ArchiveUnavailable("down"), "Experiment Manager not reachable"),
        (ValueError("bad file"), "Could not analyse: bad file"),
    ],
)
def test_view_shows_the_compute_error_instead_of_raising(
    qapp, tmp_path, monkeypatch, failure, shown
):
    def fails(measurement, settings):
        raise failure

    monkeypatch.setattr(analyses, "tweezer_histogram", fails)
    view = open_view(ResultsArchive(tmp_path), frames_file(tmp_path))

    assert until(lambda: shown in view.message.text(), qapp=qapp)


def test_a_file_without_the_attribute_declares_no_analyses(tmp_path):
    measurement = ResultsArchive(tmp_path).load(frames_file(tmp_path), results=[])
    del measurement.attrs["analyses"]

    assert declared_analyses(measurement) == []


def test_only_one_analysis_runs_at_a_time_and_the_latest_request_wins(
    qapp, tmp_path, monkeypatch
):
    archive = ResultsArchive(tmp_path)
    relative = frames_file(tmp_path, scan=Scan(repetitions=6))
    release = threading.Event()
    running, overlaps, groups = [0], [], []
    compute = analyses.tweezer_histogram

    def counted(measurement, settings):
        running[0] += 1
        overlaps.append(running[0])
        release.wait(5)
        groups.append(settings.group)
        running[0] -= 1
        return compute(measurement, settings)

    monkeypatch.setattr(analyses, "tweezer_histogram", counted)
    view = open_view(archive, relative)
    for index in range(1, 6):
        view.shots_from.setCurrentIndex(index)
    release.set()

    assert until(lambda: groups and groups[-1] == 4, qapp=qapp)
    assert max(overlaps) == 1
    assert len(groups) <= 2


def test_the_previous_files_histogram_is_cleared_while_the_next_loads(
    qapp, tmp_path, monkeypatch
):
    archive = ResultsArchive(tmp_path)
    first, second = frames_file(tmp_path, 1), frames_file(tmp_path, 2, "second")
    view = open_view(archive, first)
    assert until(lambda: "loading" in view.plots.readout.text(), qapp=qapp)
    release = threading.Event()
    compute = analyses.tweezer_histogram

    def held(measurement, settings):
        release.wait(5)
        return compute(measurement, settings)

    monkeypatch.setattr(analyses, "tweezer_histogram", held)
    view.set_measurement(archive, second, archive.load(second, results=[]))

    assert view.plots.readout.text() == ""
    release.set()


def sizes_file(path, values, repetitions, sigma_x, sigma_y, error):
    """``thorcam`` sizes per point, in point order, each with the recorded ``error``."""

    def record(point):
        return {
            "thorcam_sigma_x": sigma_x[point.index],
            "thorcam_sigma_y": sigma_y[point.index],
            "thorcam_sigma_x_error": error[point.index],
            "thorcam_sigma_y_error": error[point.index],
        }

    scan = Scan(axes=[ListAxis(argument="x", values=values)], repetitions=repetitions)
    return ResultsArchive(path.parent).load(
        write_run(path, scan=scan, record=record).name
    )


def test_cloud_sizes_pool_repeated_scan_values(tmp_path):
    measurement = sizes_file(
        tmp_path / "m.h5",
        values=[1.0, 2.0],
        repetitions=3,
        sigma_x=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        sigma_y=[5.0, 5.0, 5.0, 1.0, 1.0, 4.0],
        error=[0.5] * 6,
    )

    sizes = cloud_sizes(measurement, "thorcam", "x")

    x, mean, error = sizes["sigma_x"]
    np.testing.assert_allclose(x, [1.0, 2.0])
    np.testing.assert_allclose(mean, [2.0, 6.0])
    np.testing.assert_allclose(error, [1 / np.sqrt(3), 2 / np.sqrt(3)])
    _, mean, error = sizes["sigma_y"]
    np.testing.assert_allclose(mean, [5.0, 2.0])
    np.testing.assert_allclose(error, [0.0, 1.0])


def test_single_point_value_keeps_recorded_error(tmp_path):
    measurement = sizes_file(
        tmp_path / "m.h5",
        values=[1.0, 2.0],
        repetitions=1,
        sigma_x=[3.0, 4.0],
        sigma_y=[5.0, 6.0],
        error=[0.25, np.nan],
    )

    x, mean, error = cloud_sizes(measurement, "thorcam", "x")["sigma_x"]

    np.testing.assert_allclose(mean, [3.0, 4.0])
    np.testing.assert_allclose(error, [0.25, np.nan])


def cloud_frames(centre_rc, shape=(40, 50), sigma=3.0, frames=2, seed=0):
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    cloud = 200 * np.exp(
        -((rows - centre_rc[0]) ** 2 + (cols - centre_rc[1]) ** 2) / (2 * sigma**2)
    )
    noise = np.random.default_rng(seed).normal(0, 2, (frames, *shape))
    return cloud + noise


def cloud_fit_file(root, centre_rc=(14, 30)):
    """A ``thorcam`` cloud fit per point, recorded in metres from a cropped frame.

    As from a camera with ``object_pixel_m`` and an ROI; the mean image itself
    carries no geometry. 2 values of ``x`` x 2 repetitions.
    """

    def record(point):
        values, mean_image = fit_frames(
            cloud_frames(centre_rc, sigma=2.0 + point.values["x"], seed=point.index),
            None,
            (4e-6, 4e-6),
            (100.0, 60.0),
        )
        recorded = {f"thorcam_{name}": value for name, value in values.items()}
        return recorded | {"thorcam_mean_image": mean_image}

    relative = "2026/11/04/000001_Sweep.h5"
    scan = Scan(axes=[ListAxis(argument="x", values=[1.0, 2.0])], repetitions=2)
    write_run(root / relative, scan=scan, record=record)
    return relative


class RecordingArchive(ResultsArchive):
    """Notes every load and point read, its clones' included."""

    def __init__(self, root, reads=None):
        super().__init__(root)
        self.reads = [] if reads is None else reads

    def clone(self):
        return RecordingArchive(self.root, self.reads)

    def load(self, path, results=None):
        self.reads.append(("load", results))
        return super().load(path, results)

    def point_result(self, path, name, index):
        self.reads.append(("point", name, index))
        return super().point_result(path, name, index)


def open_cloud_view(qapp, archive, relative):
    view = CloudFitView()
    view.set_measurement(archive, relative, archive.load(relative, results=[]))
    until(lambda: view.outline_curves, qapp=qapp)
    return view


def test_cloud_fit_view_draws_a_fit_file(qapp, tmp_path):
    centre_rc = (14, 30)
    relative = cloud_fit_file(tmp_path, centre_rc)

    view = open_cloud_view(qapp, ResultsArchive(tmp_path), relative)

    assert len(view.outline_curves) == 8
    assert view.image_item.image.shape == (40, 50)
    outline_x, outline_y = view.outline_curves[0].getData()
    assert np.mean(outline_x) == pytest.approx(centre_rc[1] + 0.5, abs=0.2)
    assert np.mean(outline_y) == pytest.approx(centre_rc[0] + 0.5, abs=0.2)
    for field in ("sigma_x", "sigma_y"):
        x, sigma = view.size_curves[field].getData()
        np.testing.assert_allclose(x, [1.0, 2.0])
        np.testing.assert_allclose(sigma, [12e-6, 16e-6], rtol=0.1)


def test_cloud_fit_view_loads_no_mean_images_for_the_plot(qapp, tmp_path):
    archive = RecordingArchive(tmp_path)
    relative = cloud_fit_file(tmp_path)

    open_cloud_view(qapp, archive, relative)

    loaded = [name for read in archive.reads if read[0] == "load" for name in read[1]]
    assert "thorcam_sigma_x" in loaded
    assert not [name for name in loaded if name.endswith("_mean_image")]


def test_choosing_a_point_reads_only_its_mean_image(qapp, tmp_path):
    archive = RecordingArchive(tmp_path)
    view = open_cloud_view(qapp, archive, cloud_fit_file(tmp_path))
    del archive.reads[:]

    view.point.setCurrentIndex(3)

    assert until(lambda: ("point", "thorcam_mean_image", 3) in archive.reads, qapp=qapp)
    assert archive.reads == [("point", "thorcam_mean_image", 3)]


def open_temperature_view(qapp, measurement, tmp_path):
    view = CloudTemperatureView()
    view.set_measurement(ResultsArchive(tmp_path), "tof.h5", measurement)
    until(lambda: view.readout.text() or view.message.text() != "Analysing…", qapp=qapp)
    return view


def test_cloud_temperature_view_shows_the_temperature_and_g(qapp, tmp_path):
    view = open_temperature_view(qapp, tof_file(tmp_path), tmp_path)

    assert "Tx = " in view.readout.text()
    assert "Ty = " in view.readout.text()
    assert "g = 9.8" in view.readout.text()
    assert len(view.sigma_curves["sigma_x"].getData()[0]) == 5
    assert len(view.gravity_line.getData()[0]) == 2


def test_cloud_temperature_view_says_when_no_species_was_recorded(qapp, tmp_path):
    view = open_temperature_view(qapp, tof_file(tmp_path, species=False), tmp_path)

    assert "recorded no species" in view.message.text()
    assert view.readout.text() == ""


@pytest.mark.parametrize(
    ("argument", "seconds"),
    [
        (Number(1e-3, unit="s"), True),
        (Number(1e-3, unit="ms", scale=1e-3), True),
        (Number(1e-5, unit="us", scale=1e-6), True),
        (Number(1e-5, unit="µs", scale=1e-6), True),
        (MotMasterNumber(100.0, unit="ms", parameter="FreeExpTime"), False),
        (MotMasterNumber(100.0, unit="ms/100", parameter="FreeExpTime"), False),
        (Number(100.0), False),
    ],
)
def test_only_native_time_arguments_hold_seconds(argument, seconds):
    measurement = SimpleNamespace(argument_schema={"t": argument.to_schema()})

    assert holds_seconds(measurement, "t") is seconds
