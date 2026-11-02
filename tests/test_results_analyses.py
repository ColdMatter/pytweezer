import threading

import pytest

from pytweezer.experiment import Scan, run_local
from pytweezer.experiment.archive import ArchiveUnavailable, ResultsArchive
from pytweezer.GUI.experiments import analyses
from pytweezer.GUI.experiments.analyses import (
    TweezerHistogramView,
    declared_analyses,
)
from tests.helpers import until
from tests.test_analysis import SITE_CENTRES
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
    view.set_measurement(archive, second, archive.load(second, results=[]))
    assert until(lambda: "loading" in view.plots.readout.text(), qapp=qapp)

    release.set()

    assert not until(
        lambda: "from the first file" in view.plots.readout.text(),
        timeout=0.5,
        qapp=qapp,
    )


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
