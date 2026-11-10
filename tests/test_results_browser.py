import threading
import time

import h5py
import numpy as np
import pytest

from pytweezer.experiment import Choice, Experiment, ListAxis, Number, Scan, run_local
from pytweezer.experiment.archive import ResultsArchive
from pytweezer.GUI.experiments.analyses import AnalysisView
from pytweezer.GUI.experiments.results import ResultsPanel, aggregate, resubmission
from tests.helpers import FakeFeed, until
from tests.helpers import settle as settle_threads


class Sweep(Experiment):
    x = Number(0.0, unit="us", scale=1e-6)
    gain = Number(2.0)
    mode = Choice(["a", "b"])

    def run_point(self):
        self.record(
            "y", self.gain * self.x * 1e6 + self.point.repetition, unit="counts"
        )
        self.record("image", np.zeros((4, 4)))
        self.record("note", "text")


SCAN = Scan(
    axes=[
        ListAxis(argument="x", values=[1e-6, 2e-6, 3e-6]),
        ListAxis(argument="mode", values=["a", "b"]),
    ],
    repetitions=2,
)


def days(panel):
    tree = panel.tree
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


def children(day):
    return [day.child(i).text(0) for i in range(day.childCount())]


def make_file(root, rid=1, day=("2026", "10", "07")):
    path = root.joinpath(*day, f"{rid:06d}_Sweep.h5")
    run_local(Sweep, SCAN, path=path, gain=3)
    return path


def relpath(root, path):
    return path.relative_to(root).as_posix()


def settle(qapp):
    settle_threads(qapp, "Results ")


def refreshed(panel, qapp):
    panel.refresh()
    settle(qapp)


def select(panel, qapp, path):
    panel.select_path(path)
    settle(qapp)
    assert panel.path == path


class CountingArchive(ResultsArchive):
    """Local files, counting the measurement loads; one connection serves every read."""

    def __init__(self, root):
        super().__init__(root)
        self.loads = []

    def clone(self):
        return self

    def load(self, path, results=None):
        self.loads.append(path)
        return super().load(path, results)


def test_aggregate_means_and_standard_errors():
    curves = aggregate([1, 1, 2, 2, 2], [1.0, 3.0, 4.0, np.nan, 4.0])
    xs, means, sems, counts = curves[0.0]
    assert list(xs) == [1, 2]
    assert list(means) == [2.0, 4.0]
    assert sems[0] == pytest.approx(1.0)
    assert list(counts) == [2, 2]
    by_series = aggregate([1, 1], [1.0, 2.0], series=["a", "b"])
    assert set(by_series) == {"a", "b"}
    assert np.isnan(by_series["a"][2][0])


def test_resubmission_reuses_arguments_and_scan():
    measurement = run_local(Sweep, SCAN, gain=3, label="original")
    request = resubmission(measurement)
    assert request.class_name == "Sweep"
    assert request.args == {"gain": 3.0}
    assert request.scan == SCAN
    assert request.label == "original"


def test_panel_lists_days_and_hides_running_files_until_they_finish(qapp, tmp_path):
    make_file(tmp_path, 1)
    make_file(tmp_path, 2, day=("2026", "10", "06"))
    running = make_file(tmp_path, 3)
    with h5py.File(running, "r+") as f:
        f.attrs["status"] = "running"
    panel = ResultsPanel(root=tmp_path)
    refreshed(panel, qapp)
    assert days(panel) == ["2026-10-07", "2026-10-06"]
    newest = panel.tree.topLevelItem(0)
    assert children(newest) == ["000001  Sweep"]

    panel.show_unfinished.setChecked(True)
    settle(qapp)
    newest = panel.tree.topLevelItem(0)
    assert [newest.child(i).text(1) for i in range(2)] == ["running", "completed"]
    panel.show_unfinished.setChecked(False)
    settle(qapp)
    assert newest.childCount() == 1

    with h5py.File(running, "r+") as f:
        f.attrs["status"] = "completed"
    refreshed(panel, qapp)
    assert children(newest) == ["000003  Sweep", "000001  Sweep"]


def test_panel_shows_metadata_plots_and_resubmits(qapp, tmp_path):
    path = make_file(tmp_path)
    panel = ResultsPanel(root=tmp_path)
    refreshed(panel, qapp)
    select(panel, qapp, relpath(tmp_path, path))
    text = panel.metadata.toPlainText()
    assert (
        "Status: completed" in text and "gain = 3" in text and "x = (scanned)" in text
    )
    assert [panel.y_choice.itemText(i) for i in range(panel.y_choice.count())] == ["y"]
    assert panel.x_choice.currentText() == "x"

    panel.series_choice.setCurrentText("mode")
    settle(qapp)
    curves = [item for item in panel.plot.listDataItems()]
    assert len(curves) == 2
    xs, ys = curves[0].getData()
    assert list(xs) == pytest.approx([1, 2, 3])  # display units
    assert list(ys) == pytest.approx([3.5, 6.5, 9.5])  # mean over two repetitions

    panel.x_choice.setCurrentText("mode")  # a categorical axis
    panel.series_choice.setCurrentText("(none)")
    settle(qapp)
    assert len(panel.plot.listDataItems()) == 1

    requests = []
    panel.resubmit_requested.connect(requests.append)
    panel.resubmit_button.click()
    assert requests[0].args == {"gain": 3.0}


def test_tree_updates_in_place_without_a_click(qapp, tmp_path):
    make_file(tmp_path, 1)
    panel = ResultsPanel(root=tmp_path)
    refreshed(panel, qapp)
    first = panel.tree.topLevelItem(0)
    select(panel, qapp, "2026/10/07/000001_Sweep.h5")

    make_file(tmp_path, 2)
    make_file(tmp_path, 3, day=("2026", "10", "08"))
    refreshed(panel, qapp)  # what the timer does
    assert days(panel) == ["2026-10-08", "2026-10-07"]
    assert panel.tree.topLevelItem(1) is first  # not rebuilt
    assert children(first) == ["000002  Sweep", "000001  Sweep"]
    assert panel.tree.topLevelItem(0).isExpanded()  # newest day opens
    assert panel.tree.currentItem().text(0) == "000001  Sweep"  # selection kept


def test_selected_running_measurement_reloads_and_keeps_choices(qapp, tmp_path):
    path = make_file(tmp_path, 1)
    with h5py.File(path, "r+") as f:
        f.attrs["status"] = "running"
        f.attrs["n_done"] = 2
    panel = ResultsPanel(root=tmp_path)
    panel.show_unfinished.setChecked(True)
    refreshed(panel, qapp)
    select(panel, qapp, relpath(tmp_path, path))
    panel.x_choice.setCurrentText("mode")
    settle(qapp)
    assert "points 2/12" in panel.metadata.toPlainText()

    with h5py.File(path, "r+") as f:
        f.attrs["status"] = "completed"
        f.attrs["n_done"] = 12
    refreshed(panel, qapp)
    assert "points 12/12" in panel.metadata.toPlainText()
    assert panel.x_choice.currentText() == "mode"


def running_file(root, rid=1):
    path = make_file(root, rid)
    with h5py.File(path, "r+") as f:
        f.attrs.update(status="running", rid=rid, n_done=0)
    return path


def test_live_running_measurement_plots_from_the_feed(qapp, tmp_path):
    path = running_file(tmp_path)
    feed = FakeFeed()
    archive = CountingArchive(tmp_path)
    panel = ResultsPanel(archive=archive, feed=feed)
    panel.show_unfinished.setChecked(True)
    refreshed(panel, qapp)
    select(panel, qapp, relpath(tmp_path, path))
    panel.x_choice.setCurrentText("point index")
    settle(qapp)

    reads = archive.loads
    reads.clear()
    feed.rows[1] = [
        {"index": 0, "values": {}, "scalars": {"y": 10.0}},
        {"index": 1, "values": {}, "scalars": {"y": 20.0, "extra": 1.0}},
    ]
    feed.point_received.emit({"rid": 1, **feed.rows[1][-1]})
    assert reads == [], "a live measurement is not re-read"
    (curve,) = panel.plot.listDataItems()
    assert list(curve.yData) == [10.0, 20.0]
    assert "extra" in [
        panel.y_choice.itemText(i) for i in range(panel.y_choice.count())
    ]
    assert "2/12 points" in panel.detail_region.hint.text()

    refreshed(panel, qapp)
    assert reads == [], "nor by the tree refresh"

    with h5py.File(path, "r+") as f:
        f.attrs.update(status="completed", n_done=12)
    feed.queue_changed.emit(
        {"running": [], "queue": [], "history": [{"rid": 1, "status": "completed"}]}
    )
    settle(qapp)
    assert panel.measurement.status == "completed"
    assert "12/12 points" in panel.detail_region.hint.text()


def test_queue_changes_drive_refreshes_and_polling_slows(qapp, tmp_path, monkeypatch):
    feed = FakeFeed()
    panel = ResultsPanel(root=tmp_path, feed=feed)
    panel.show()
    calls = []
    real_refresh = panel.refresh
    monkeypatch.setattr(panel, "refresh", lambda: calls.append(1) or real_refresh())
    snapshot = {
        "running": [{"rid": 2, "status": "running"}],
        "queue": [],
        "history": [],
    }
    feed.queue_changed.emit(snapshot)
    feed.queue_changed.emit(snapshot)
    assert calls == [1], "only a change of task status refreshes"

    panel._poll_tick()
    assert calls == [1], "live, the poll waits for the slow fallback"
    feed.connected = False
    panel._poll_tick()
    assert calls == [1, 1]
    panel.hide()
    panel._poll_tick()
    assert calls == [1, 1], "a hidden tab is not polled"


def test_running_file_is_listed_once_its_worker_creates_it(qapp, tmp_path):
    feed = FakeFeed()
    panel = ResultsPanel(root=tmp_path, feed=feed)
    panel.show_unfinished.setChecked(True)
    panel.show()
    settle(qapp)
    # The queue reports the task running before its worker has created the file.
    feed.queue_changed.emit(
        {
            "running": [
                {"rid": 1, "status": "running", "h5_path": "2026/10/07/000001_Sweep.h5"}
            ],
            "queue": [],
            "history": [],
        }
    )
    settle(qapp)
    assert days(panel) == []

    running_file(tmp_path)
    panel._poll_tick()
    settle(qapp)
    newest = panel.tree.topLevelItem(0)
    assert [newest.child(0).text(i) for i in range(2)] == ["000001  Sweep", "running"]

    calls = []
    panel.refresh = lambda: calls.append(1)
    panel._poll_tick()
    assert calls == [], "once listed, the poll waits for the slow fallback again"
    panel.hide()


def test_queue_snapshot_without_running_list_is_tolerated(qapp, tmp_path):
    feed = FakeFeed()
    panel = ResultsPanel(root=tmp_path, feed=feed)
    feed.queue_changed.emit(
        {"running": None, "queue": [{"rid": 3, "status": "queued"}], "history": []}
    )
    assert panel._task_statuses == {3: "queued"}


class HistogramSweep(Sweep):
    analyses = ("tweezer_histogram",)


def make_histogram_file(root, rid, status=None, analyses=None):
    path = root.joinpath("2026", "10", "07", f"{rid:06d}_HistogramSweep.h5")
    run_local(HistogramSweep, SCAN, path=path)
    with h5py.File(path, "r+") as f:
        if status is not None:
            f.attrs["status"] = status
        if analyses is not None:
            f.attrs["analyses"] = analyses
    return relpath(root, path)


def analysis_items(panel):
    combo = panel.analysis_choice
    return [
        (combo.itemText(i), combo.model().item(i).isEnabled())
        for i in range(combo.count())
    ]


def test_dropdown_offers_plot_against_then_the_files_analyses(qapp, tmp_path):
    histogram = make_histogram_file(tmp_path, 1)
    plain = relpath(tmp_path, make_file(tmp_path, 2))
    panel = ResultsPanel(root=tmp_path)
    refreshed(panel, qapp)

    select(panel, qapp, histogram)
    assert analysis_items(panel) == [
        ("Plot against", True),
        ("Tweezer loading histogram (counts)", True),
    ]
    select(panel, qapp, plain)
    assert analysis_items(panel) == [("Plot against", True)]


def test_unknown_analysis_is_listed_disabled(qapp, tmp_path):
    path = make_histogram_file(tmp_path, 1, analyses='["from_the_future"]')
    panel = ResultsPanel(root=tmp_path)
    refreshed(panel, qapp)

    select(panel, qapp, path)

    assert analysis_items(panel) == [("Plot against", True), ("from_the_future", False)]


def test_analyses_are_disabled_while_the_measurement_runs(qapp, tmp_path):
    path = make_histogram_file(tmp_path, 1, status="running")
    panel = ResultsPanel(root=tmp_path)
    panel.show_unfinished.setChecked(True)
    refreshed(panel, qapp)

    select(panel, qapp, path)

    assert analysis_items(panel)[1] == ("Tweezer loading histogram (counts)", False)


def test_choosing_an_analysis_shows_its_view(qapp, tmp_path):
    path = make_histogram_file(tmp_path, 1)
    panel = ResultsPanel(root=tmp_path)
    refreshed(panel, qapp)
    select(panel, qapp, path)

    panel.analysis_choice.setCurrentIndex(1)

    assert isinstance(panel.analysis_stack.currentWidget(), AnalysisView)
    panel.analysis_choice.setCurrentIndex(0)
    assert panel.analysis_stack.currentIndex() == 0


def test_the_choice_is_kept_for_the_next_file_of_the_same_class(qapp, tmp_path):
    first, second = make_histogram_file(tmp_path, 1), make_histogram_file(tmp_path, 2)
    plain = relpath(tmp_path, make_file(tmp_path, 3))
    panel = ResultsPanel(root=tmp_path)
    refreshed(panel, qapp)
    select(panel, qapp, first)
    panel.analysis_choice.setCurrentIndex(1)

    select(panel, qapp, plain)
    assert panel.analysis_choice.currentText() == "Plot against"
    select(panel, qapp, second)
    assert panel.analysis_choice.currentText() == "Tweezer loading histogram (counts)"


class StalledArchive(ResultsArchive):
    """Local files behind a link that answers only once released, like an unreachable manager."""

    def __init__(self, root):
        super().__init__(root)
        self.released = threading.Event()

    def clone(self):
        return self

    def days(self):
        self.released.wait(5)
        return super().days()

    def load(self, path, results=None):
        self.released.wait(5)
        return super().load(path, results)


def test_a_slow_archive_never_holds_up_the_gui_thread(qapp, tmp_path):
    path = make_file(tmp_path)
    archive = StalledArchive(tmp_path)
    panel = ResultsPanel(archive=archive)

    started = time.monotonic()
    panel.refresh()
    panel.load(relpath(tmp_path, path))
    qapp.processEvents()
    assert time.monotonic() - started < 1
    assert days(panel) == []

    archive.released.set()
    assert until(lambda: days(panel) == ["2026-10-07"], qapp=qapp)
    assert until(lambda: "Status: completed" in panel.metadata.toPlainText(), qapp=qapp)


def test_resubmission_leaves_arguments_taken_from_saved_parameters_unset(monkeypatch):
    from tests.test_experiment_arguments import Calibrated

    monkeypatch.setattr(
        "pytweezer.experiment.client.get_saved_parameters", lambda: {"blaze": 48e-6}
    )
    request = resubmission(run_local(Calibrated, shots=4))

    assert request.args == {"blaze": None, "shots": 4}
