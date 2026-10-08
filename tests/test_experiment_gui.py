"""Experiments tab widgets, offscreen, with a fake manager client and feed."""

import pytest
from PyQt6 import QtCore

from pytweezer.experiment import (
    Choice,
    Experiment,
    Integer,
    LinearAxis,
    ListAxis,
    Number,
    Scan,
)
from pytweezer.experiment.task import TaskRequest
from pytweezer.GUI.experiments import queue_view
from pytweezer.GUI.experiments.arg_editor import ArgumentEditor, parse_list
from pytweezer.GUI.experiments.panel import ExperimentsPanel
from pytweezer.GUI.experiments.queue_view import QueueView


class Demo(Experiment):
    """A demo."""

    detuning = Number(-5e6, unit="MHz", scale=1e6, min=-20e6, max=0)
    shots = Integer(3, min=1)
    mode = Choice(["a", "b"])

    def run_point(self):
        pass


SCHEMA = Demo.schema()
MODULES = [
    {"module": SCHEMA["module"], "classes": [SCHEMA], "warnings": [], "error": None},
    {
        "module": "pytweezer.experiments.broken",
        "classes": [],
        "warnings": [],
        "error": "boom",
    },
]


class FakeClient:
    def __init__(self):
        self.calls = []
        self.last = None

    def call(self, command, **fields):
        self.calls.append((command, fields))
        if command == "catalogue":
            return {"ok": True, "modules": MODULES, "version": 1}
        return {"ok": True}

    def submit(self, request):
        self.calls.append(("submit", request))
        return 12

    def last_request(self, module, class_name):
        return self.last

    def close(self):
        pass


class FakeFeed(QtCore.QObject):
    queue_changed = QtCore.pyqtSignal(dict)
    point_received = QtCore.pyqtSignal(dict)
    connection_changed = QtCore.pyqtSignal(bool)

    def close(self):
        pass


def task(rid, status, **kwargs):
    return {
        "rid": rid,
        "status": status,
        "experiment": "pytweezer.experiments.demo",
        "class_name": "Demo",
        "args": {},
        "scan": {"axes": []},
        "priority": 0,
        "label": "",
        "submitter": "me",
        "t_submit": "2026-10-07T12:00:00+01:00",
        "points_done": 1,
        "points_total": 4,
        "requested": None,
        "error": None,
        **kwargs,
    }


@pytest.fixture
def editor(qapp):
    editor = ArgumentEditor()
    editor.set_experiment(SCHEMA)
    return editor


def test_form_shows_display_units_and_returns_si(editor):
    row = editor.rows["detuning"]
    assert row.value.widget.value() == pytest.approx(-5.0)
    assert row.unit.text() == "MHz"
    assert row.value.widget.minimum() == pytest.approx(-20)
    row.value.widget.setValue(-7.5)
    request = editor.request()
    assert request.args == {"detuning": -7.5e6, "shots": 3, "mode": "a"}
    assert request.experiment == SCHEMA["module"]
    assert request.class_name == "Demo"


def test_non_default_and_scanned_arguments_are_flagged(editor):
    label = editor.rows["shots"].label
    assert not label.property("state")
    editor.rows["shots"].value.widget.setValue(4)
    assert label.property("state") == "modified"
    editor.rows["shots"].value.widget.setValue(3)
    assert not label.property("state")
    editor.rows["shots"].scan_button.setChecked(True)
    assert label.property("state") == "scanned"


def test_scan_axes_and_point_count(editor):
    detuning = editor.rows["detuning"]
    detuning.scan_button.setChecked(True)
    detuning.scan.start.widget.setValue(-10)
    detuning.scan.stop.widget.setValue(0)
    detuning.scan.steps.setValue(5)
    mode = editor.rows["mode"]
    mode.scan_button.setChecked(True)
    mode.scan.values.setText("a, b")
    editor.repetitions.setValue(3)
    editor.order.setCurrentText("snake")
    assert editor.point_count.text() == "30 points"

    request = editor.request()
    assert "detuning" not in request.args and "mode" not in request.args
    assert request.scan.repetitions == 3 and request.scan.order == "snake"
    [linear, listed] = request.scan.axes
    assert (linear.start, linear.stop, linear.n) == (pytest.approx(-10e6), 0, 5)
    assert listed.values == ["a", "b"]
    assert len(request.scan.points(Demo)) == 30


def test_invalid_list_is_reported_not_submitted(editor):
    submitted = []
    editor.submit_requested.connect(submitted.append)
    editor.rows["shots"].scan_button.setChecked(True)
    editor.rows["shots"].scan.mode.setCurrentText("list")
    editor.rows["shots"].scan.values.setText("1, 0")
    editor.submit_button.click()
    assert submitted == []
    assert "outside the allowed range" in editor.error.text()


def test_parse_list_handles_units_and_kinds():
    assert parse_list("-1, -2.5", SCHEMA["arguments"]["detuning"]) == [-1e6, -2.5e6]
    assert parse_list("true,0", {"kind": "bool"}) == [True, False]
    with pytest.raises(ValueError, match="not one of"):
        parse_list("c", SCHEMA["arguments"]["mode"])
    with pytest.raises(ValueError, match="empty"):
        parse_list(" , ", SCHEMA["arguments"]["shots"])


def test_load_request_round_trips(editor):
    request = TaskRequest(
        experiment=SCHEMA["module"],
        class_name="Demo",
        args={"shots": 5, "mode": "b", "removed_argument": 1},
        scan=Scan(
            axes=[LinearAxis(argument="detuning", start=-2e6, stop=-1e6, n=3)],
            repetitions=2,
            order="shuffle",
            seed=4,
        ),
        priority=3,
        label="again",
    )
    editor.load_request(request)
    again = editor.request()
    assert again.args == {"shots": 5, "mode": "b"}
    assert again.scan.axes == request.scan.axes
    assert (again.priority, again.label, again.scan.repetitions) == (3, "again", 2)
    editor.load_request(
        request.model_copy(
            update={"scan": Scan(axes=[ListAxis(argument="shots", values=[1, 2])])}
        )
    )
    assert editor.rows["shots"].scan.values.text() == "1, 2"


def test_queue_view_rows_and_buttons(qapp):
    view = QueueView()
    view.set_snapshot(
        {
            "running": task(3, "running"),
            "queue": [task(4, "queued"), task(5, "held")],
            "history": [task(2, "failed", error="Traceback\nValueError: bad")],
        }
    )
    assert [view.table.item(r, 0).text() for r in range(4)] == ["3", "4", "5", "2"]
    assert view.table.item(3, 3).toolTip() == "ValueError: bad"

    actions = []
    view.action_requested.connect(
        lambda command, fields: actions.append((command, fields))
    )
    view.table.selectRow(0)
    assert view.buttons["pause"].isEnabled() and not view.buttons["delete"].isEnabled()
    view.buttons["pause"].click()
    view.confirm = lambda question: True
    view.buttons["abort"].click()
    view.table.selectRow(1)
    assert view.buttons["hold"].isEnabled() and not view.buttons["pause"].isEnabled()
    view.buttons["raise"].click()
    assert actions == [
        ("pause", {"rid": 3}),
        ("abort", {"rid": 3}),
        ("set_priority", {"rid": 4, "priority": 1}),
    ]

    # selection survives a refresh
    view.set_snapshot({"running": None, "queue": [task(4, "queued")], "history": []})
    assert view.selected_task()["rid"] == 4


def test_open_in_grafana_needs_a_started_queued_task(qapp, monkeypatch):
    opened = []
    monkeypatch.setattr(queue_view, "open_in_browser", opened.append)
    view = QueueView()
    view.set_snapshot(
        {
            "running": task(3, "running", t_start="2026-10-07T12:01:00+01:00"),
            "queue": [task(4, "queued")],
            "history": [],
        }
    )
    view.table.selectRow(1)
    assert not view.buttons["grafana"].isEnabled()
    view.table.selectRow(0)
    view.buttons["grafana"].click()
    assert len(opened) == 1 and "var-rid=3" in opened[0]


def test_abort_needs_confirmation(qapp):
    view = QueueView()
    view.set_snapshot({"running": task(3, "running"), "queue": [], "history": []})
    actions = []
    view.action_requested.connect(lambda *args: actions.append(args))
    view.confirm = lambda question: False
    view.table.selectRow(0)
    view.buttons["abort"].click()
    assert actions == []


def test_panel_submits_and_forwards_actions(qapp):
    client, feed = FakeClient(), FakeFeed()
    panel = ExperimentsPanel(client=client, feed=feed)
    panel.refresh_catalogue()
    assert panel.catalogue.select(SCHEMA["module"], "Demo")
    assert panel.editor.schema["class_name"] == "Demo"

    panel.editor.rows["shots"].value.widget.setValue(9)
    panel.editor.submit_button.click()
    [(command, request)] = [c for c in client.calls if c[0] == "submit"]
    assert request.args["shots"] == 9 and "@" in request.submitter
    assert "12" in panel.status.text()

    feed.queue_changed.emit(
        {
            "running": task(12, "running"),
            "queue": [],
            "history": [],
            "catalogue_version": 1,
        }
    )
    panel.queue_view.table.selectRow(0)
    panel.queue_view.buttons["terminate"].click()
    assert ("terminate", {"rid": 12}) in client.calls
    catalogue_calls = [c for c in client.calls if c[0] == "catalogue"]
    feed.queue_changed.emit(
        {"running": None, "queue": [], "history": [], "catalogue_version": 2}
    )
    assert (
        len([c for c in client.calls if c[0] == "catalogue"])
        == len(catalogue_calls) + 1
    )


def test_panel_prefills_last_request_and_keeps_drafts(qapp):
    client, feed = FakeClient(), FakeFeed()
    client.last = TaskRequest(
        experiment=SCHEMA["module"], class_name="Demo", args={"shots": 7}
    )
    panel = ExperimentsPanel(client=client, feed=feed)
    panel.refresh_catalogue()
    panel.catalogue.select(SCHEMA["module"], "Demo")
    assert panel.editor.rows["shots"].value.value() == 7

    panel.editor.rows["shots"].value.widget.setValue(8)
    panel._save_draft()
    panel._current_key = None
    panel._experiment_selected(SCHEMA)
    assert panel.editor.rows["shots"].value.value() == 8


def test_panel_resubmit_selects_the_experiment(qapp):
    panel = ExperimentsPanel(client=FakeClient(), feed=FakeFeed())
    panel.refresh_catalogue()
    panel.load_request(
        TaskRequest(experiment=SCHEMA["module"], class_name="Demo", args={"mode": "b"})
    )
    assert panel.catalogue.selected_key() == (SCHEMA["module"], "Demo")
    assert panel.editor.rows["mode"].value.value() == "b"
    panel.load_request(TaskRequest(experiment="gone", class_name="X"))
    assert "not in the catalogue" in panel.status.text()
