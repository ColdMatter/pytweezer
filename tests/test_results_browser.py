import h5py
import numpy as np
import pytest

from pytweezer.experiment import Choice, Experiment, ListAxis, Number, Scan, run_local
from pytweezer.GUI.experiments.results import ResultsPanel, aggregate, resubmission


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


def make_file(root, rid=1, day=("2026", "10", "07")):
    path = root.joinpath(*day, f"{rid:06d}_Sweep.h5")
    run_local(Sweep, SCAN, path=path, gain=3)
    return path


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


def test_resubmission_reuses_arguments_and_scan(tmp_path):
    measurement = run_local(Sweep, SCAN, gain=3, label="original")
    request = resubmission(measurement)
    assert request.class_name == "Sweep"
    assert request.args == {"gain": 3.0}
    assert request.scan == SCAN
    assert request.label == "original"


def test_panel_lists_days_and_hides_running_files(qapp, tmp_path):
    make_file(tmp_path, 1)
    make_file(tmp_path, 2, day=("2026", "10", "06"))
    running = make_file(tmp_path, 3)
    with h5py.File(running, "r+") as f:
        f.attrs["status"] = "running"
    panel = ResultsPanel(root=tmp_path)
    panel.refresh()
    days = [
        panel.tree.topLevelItem(i).text(0)
        for i in range(panel.tree.topLevelItemCount())
    ]
    assert days == ["2026-10-07", "2026-10-06"]
    newest = panel.tree.topLevelItem(0)
    assert [newest.child(i).text(0) for i in range(newest.childCount())] == [
        "000001  Sweep"
    ]
    panel.show_unfinished.setChecked(True)
    newest = panel.tree.topLevelItem(0)
    assert newest.childCount() == 2


def test_panel_shows_metadata_plots_and_resubmits(qapp, tmp_path):
    path = make_file(tmp_path)
    panel = ResultsPanel(root=tmp_path)
    panel.refresh()
    assert panel.select_path(path)
    text = panel.metadata.toPlainText()
    assert (
        "Status: completed" in text and "gain = 3" in text and "x = (scanned)" in text
    )
    assert [panel.y_choice.itemText(i) for i in range(panel.y_choice.count())] == ["y"]
    assert panel.x_choice.currentText() == "x"

    panel.series_choice.setCurrentText("mode")
    curves = [item for item in panel.plot.listDataItems()]
    assert len(curves) == 2
    xs, ys = curves[0].getData()
    assert list(xs) == pytest.approx([1, 2, 3])  # display units
    assert list(ys) == pytest.approx([3.5, 6.5, 9.5])  # mean over two repetitions

    panel.x_choice.setCurrentText("mode")  # a categorical axis
    panel.series_choice.setCurrentText("(none)")
    assert len(panel.plot.listDataItems()) == 1

    requests = []
    panel.resubmit_requested.connect(requests.append)
    panel.resubmit_button.click()
    assert requests[0].args == {"gain": 3.0}


def test_open_in_grafana_only_for_queued_runs(qapp, tmp_path, monkeypatch):
    from pytweezer.GUI.experiments import results

    opened = []
    monkeypatch.setattr(results, "open_in_browser", opened.append)
    local = make_file(tmp_path, rid=1)
    queued = make_file(tmp_path, rid=2)
    with h5py.File(queued, "a") as f:
        f.attrs["rid"] = 12
    panel = ResultsPanel(root=tmp_path)
    panel.refresh()

    panel.load(local)
    assert not panel.grafana_button.isEnabled()
    panel.load(queued)
    panel.grafana_button.click()
    assert len(opened) == 1 and "var-rid=12" in opened[0]


def test_panel_reports_an_unreadable_file(qapp, tmp_path):
    bad = tmp_path / "2026" / "10" / "07" / "000009_Bad.h5"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not hdf5")
    panel = ResultsPanel(root=tmp_path)
    panel.refresh()
    day = panel.tree.topLevelItem(0)
    assert day.child(0).text(1) == "unreadable"
    panel.load(bad)
    assert "Could not read" in panel.metadata.toPlainText()


def _children(day):
    return [day.child(i).text(0) for i in range(day.childCount())]


def test_tree_updates_in_place_without_a_click(qapp, tmp_path):
    make_file(tmp_path, 1)
    panel = ResultsPanel(root=tmp_path)
    panel.refresh()
    first = panel.tree.topLevelItem(0)
    assert panel.select_path(tmp_path / "2026" / "10" / "07" / "000001_Sweep.h5")

    make_file(tmp_path, 2)
    make_file(tmp_path, 3, day=("2026", "10", "08"))
    panel.refresh()  # what the timer does
    days = [
        panel.tree.topLevelItem(i).text(0)
        for i in range(panel.tree.topLevelItemCount())
    ]
    assert days == ["2026-10-08", "2026-10-07"]
    assert panel.tree.topLevelItem(1) is first  # not rebuilt
    assert _children(first) == ["000002  Sweep", "000001  Sweep"]
    assert panel.tree.topLevelItem(0).isExpanded()  # newest day opens
    assert panel.tree.currentItem().text(0) == "000001  Sweep"  # selection kept


def test_running_file_appears_when_it_finishes(qapp, tmp_path):
    path = make_file(tmp_path, 1)
    with h5py.File(path, "r+") as f:
        f.attrs["status"] = "running"
    panel = ResultsPanel(root=tmp_path)
    panel.refresh()
    day = panel.tree.topLevelItem(0)
    assert day.childCount() == 0

    panel.show_unfinished.setChecked(True)
    assert [day.child(0).text(1)] == ["running"]
    panel.show_unfinished.setChecked(False)
    assert day.childCount() == 0

    with h5py.File(path, "r+") as f:
        f.attrs["status"] = "completed"
    panel.refresh()
    assert [day.child(i).text(1) for i in range(day.childCount())] == ["completed"]


def test_selected_running_measurement_reloads_and_keeps_choices(qapp, tmp_path):
    path = make_file(tmp_path, 1)
    with h5py.File(path, "r+") as f:
        f.attrs["status"] = "running"
        f.attrs["n_done"] = 2
    panel = ResultsPanel(root=tmp_path)
    panel.show_unfinished.setChecked(True)
    panel.refresh()
    assert panel.select_path(path)
    panel.x_choice.setCurrentText("mode")
    assert "points 2/12" in panel.metadata.toPlainText()

    with h5py.File(path, "r+") as f:
        f.attrs["status"] = "completed"
        f.attrs["n_done"] = 12
    panel.refresh()
    assert "points 12/12" in panel.metadata.toPlainText()
    assert panel.x_choice.currentText() == "mode"


def test_polling_skips_a_hidden_tab(qapp, tmp_path, monkeypatch):
    panel = ResultsPanel(root=tmp_path)
    calls = []
    monkeypatch.setattr(panel, "refresh", lambda: calls.append(1))
    panel._poll_tick()
    assert calls == []  # never shown
    panel.show()
    calls.clear()
    panel._poll_tick()
    assert calls == [1]
    panel.hide()
