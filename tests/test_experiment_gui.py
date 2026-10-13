"""Experiments tab widgets, offscreen, with a fake manager client and feed."""

import threading
import time

import pytest
from PyQt6 import QtCore
from PyQt6.QtTest import QTest

from pytweezer.experiment import (
    Choice,
    Experiment,
    Integer,
    LinearAxis,
    ListAxis,
    Number,
    Scan,
)
from pytweezer.experiment.client import ManagerError
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
    MotMasterNumber,
)
from pytweezer.experiment.recipes import Recipe
from pytweezer.experiment.task import TaskRequest
from pytweezer.GUI.experiments import queue_view
from pytweezer.GUI.experiments.arg_editor import ArgumentEditor, parse_list
from pytweezer.GUI.experiments.catalogue_view import CatalogueView
from pytweezer.GUI.experiments.motmaster_params import ParameterFetcher
from pytweezer.GUI.experiments.panel import ExperimentsPanel
from pytweezer.GUI.experiments.queue_view import QueueView

Left, Right = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton


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
    {"module": "pytweezer.experiments.broken", "classes": [], "error": "boom"},
]


class FakeClient:
    def __init__(self):
        self.calls = []
        self.last = None
        self.recipes = []
        self.recipes_version = 0
        self.errors = {}

    def call(self, command, **fields):
        self.calls.append((command, fields))
        if command in self.errors:
            raise ManagerError(self.errors[command])
        if command == "catalogue":
            return {"ok": True, "modules": MODULES, "version": 1}
        if command == "recipes":
            return {
                "ok": True,
                "recipes": self.recipes,
                "version": self.recipes_version,
            }
        if command == "submit_recipe":
            return {"ok": True, "rid": 13}
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


def commands(client, name):
    return [fields for command, fields in client.calls if command == name]


def pump_until(qapp, condition):
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    return condition()


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
    assert row.label.property("state") == "modified"
    request = editor.request()
    assert request.args == {"detuning": -7.5e6, "shots": 3, "mode": "a"}
    assert request.experiment == SCHEMA["module"]
    assert request.class_name == "Demo"


def test_scan_axes_and_point_count(editor):
    detuning = editor.rows["detuning"]
    detuning.scan_button.setChecked(True)
    assert detuning.label.property("state") == "scanned"
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


def test_an_invalid_form_is_reported_not_submitted_or_saved(editor):
    emitted = []
    editor.submit_requested.connect(emitted.append)
    editor.save_recipe_requested.connect(emitted.append)
    shots = editor.rows["shots"]
    shots.scan_button.setChecked(True)
    shots.scan.mode.setCurrentText("list")
    shots.scan.values.setText("1, 0")
    editor.submit_button.click()
    editor.save_recipe_button.click()
    assert emitted == []
    assert "outside the allowed range" in editor.error.text()

    # Only MOTMaster script parameters are held to whole linear steps.
    shots.scan.mode.setCurrentText("linear")
    shots.scan.start.set_value(1)
    shots.scan.stop.set_value(2)
    shots.scan.steps.setValue(3)
    assert editor.request().scan.axes[0].argument == "shots"


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


def test_the_recipe_name_is_cleared_by_anything_that_replaces_the_form(editor):
    plain = TaskRequest(experiment=SCHEMA["module"], class_name="Demo")
    for replace_form in (
        editor.defaults_button.click,
        lambda: editor.set_experiment(SCHEMA),
        lambda: editor.load_request(plain),
    ):
        editor.set_recipe_name("MOT check")
        assert not editor.recipe_label.isHidden()
        replace_form()
        assert editor.recipe_name == "" and editor.recipe_label.isHidden()


def test_queue_view_rows_and_buttons(qapp, monkeypatch):
    opened = []
    monkeypatch.setattr(queue_view, "open_in_browser", opened.append)
    view = QueueView()
    view.set_snapshot(
        {
            "running": task(3, "running", t_start="2026-10-07T12:01:00+01:00"),
            "queue": [task(4, "queued"), task(5, "held")],
            "history": [task(2, "failed", error="Traceback\nValueError: bad")],
        }
    )
    assert [view.table.item(r, 0).text() for r in range(4)] == ["3", "4", "5", "2"]
    assert "ValueError: bad" in view.table.item(3, 3).toolTip()

    actions = []
    view.action_requested.connect(
        lambda command, fields: actions.append((command, fields))
    )
    view.table.selectRow(0)
    assert view.buttons["pause"].isEnabled() and not view.buttons["delete"].isEnabled()
    view.buttons["pause"].click()
    view.confirm = lambda question: False
    view.buttons["abort"].click()
    view.confirm = lambda question: True
    view.buttons["abort"].click()
    view.buttons["grafana"].click()
    assert len(opened) == 1 and "var-rid=3" in opened[0]

    view.table.selectRow(1)
    assert view.buttons["hold"].isEnabled() and not view.buttons["pause"].isEnabled()
    assert not view.buttons["grafana"].isEnabled(), "not started yet"
    view.buttons["raise"].click()
    assert actions == [
        ("pause", {"rid": 3}),
        ("abort", {"rid": 3}),
        ("set_priority", {"rid": 4, "priority": 1}),
    ]

    # selection survives a refresh
    view.set_snapshot({"running": None, "queue": [task(4, "queued")], "history": []})
    assert view.selected_task()["rid"] == 4


def test_panel_submits_and_forwards_actions(qapp):
    client, feed = FakeClient(), FakeFeed()
    panel = ExperimentsPanel(client=client, feed=feed)
    panel.refresh_catalogue()
    assert panel.catalogue.select(SCHEMA["module"], "Demo")
    assert panel.editor.schema["class_name"] == "Demo"

    panel.editor.rows["shots"].value.widget.setValue(9)
    panel.editor.submit_button.click()
    [request] = commands(client, "submit")
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
    catalogue_fetches = len(commands(client, "catalogue"))
    feed.queue_changed.emit(
        {"running": None, "queue": [], "history": [], "catalogue_version": 2}
    )
    assert len(commands(client, "catalogue")) == catalogue_fetches + 1


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


def test_real_feed_signals_queue_changes_and_points(qapp):
    from sipyco.sync_struct import Notifier

    from pytweezer.GUI.experiments.feed import ExperimentFeed
    from pytweezer.servers.sync import SyncServer

    notifier = Notifier(
        {
            "alive": 0.0,
            "running": None,
            "queue": [],
            "history": [],
            "points": {"rid": 3, "rows": [{"index": 0, "scalars": {"n": 1.0}}]},
        }
    )
    server = SyncServer({"experiment": notifier}, "127.0.0.1", 0)
    feed = ExperimentFeed(address=("127.0.0.1", server.port), poll_interval_ms=20)
    snapshots, points, connections = [], [], []
    feed.queue_changed.connect(snapshots.append)
    feed.point_received.connect(points.append)
    feed.connection_changed.connect(connections.append)

    try:
        assert pump_until(qapp, lambda: snapshots and connections == [True])
        assert "points" not in snapshots[0]
        assert feed.points(3) == [{"index": 0, "scalars": {"n": 1.0}}]
        assert feed.points(4) == []

        snapshots.clear()
        server.call(notifier.__setitem__, "alive", 1.0).result()
        server.call(
            notifier["points"]["rows"].append, {"index": 1, "scalars": {"n": 2.0}}
        ).result()
        assert pump_until(qapp, lambda: points)
        assert points == [{"rid": 3, "index": 1, "scalars": {"n": 2.0}}]
        assert snapshots == [], "an alive bump alone is not a queue change"

        server.call(notifier.__setitem__, "points", {"rid": 4, "rows": []}).result()
        server.call(notifier["points"]["rows"].append, {"index": 0}).result()
        server.call(notifier.__setitem__, "queue", [{"rid": 5}]).result()
        assert pump_until(qapp, lambda: len(points) == 2 and snapshots)
        assert points[1]["rid"] == 4
        assert snapshots[-1]["queue"] == [{"rid": 5}]
    finally:
        feed.close()
        server.close()


class Sequenced(MotMasterExperiment):
    """A MOTMaster experiment."""

    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
    pulse = MotMasterNumber(1e-6, parameter="tPulse")


SEQUENCED = Sequenced.schema()
PARAMETERS = {
    "tDelay1": 5,
    "tPulse": 20e-6,
    "coil_current": 1.5,
    "label": "x",
    "enabled": True,
}


class Source:
    def __init__(self, parameters=None, error=None):
        self.parameters, self.error, self.calls = parameters, error, []

    def __call__(self, device, script):
        self.calls.append((device, script))
        if self.error:
            raise RuntimeError(self.error)
        return dict(self.parameters)


class SlowSource(Source):
    """Answers only once ``release`` is set."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.release = threading.Event()

    def __call__(self, device, script):
        self.release.wait(5)
        return super().__call__(device, script)


def sequenced_editor(source):
    editor = ArgumentEditor()
    editor.fetcher.threaded = False
    editor.set_parameter_source(source)
    editor.set_experiment(SEQUENCED)
    return editor


def sequenced_request(args=None, axes=()):
    return TaskRequest(
        experiment=SEQUENCED["module"],
        class_name="Sequenced",
        args=args or {},
        scan=Scan(axes=list(axes)),
    )


def load_before_fetch(qapp, request):
    """Load ``request`` while the script's parameters are still in flight.

    Returns the editor once the fetch has landed, and each row's kind before it.
    """
    source = SlowSource(PARAMETERS)
    editor = ArgumentEditor()
    editor.set_parameter_source(source)
    try:
        editor.set_experiment(SEQUENCED)
        editor.load_request(request)
        kinds = {name: row.schema["kind"] for name, row in editor.rows.items()}
    finally:
        source.release.set()
    assert pump_until(qapp, lambda: editor.motmaster_boxes["rb"].defaults)
    return editor, kinds


def test_script_parameters_are_fetched_once_and_added_from_the_search(qapp):
    source = Source(PARAMETERS)
    editor = sequenced_editor(source)
    editor.set_experiment(SEQUENCED)
    assert source.calls == [("Rb MotMaster", "RbTweezerBasic")]
    box = editor.motmaster_boxes["rb"]
    assert box.defaults == {"tDelay1": 5, "tPulse": 20e-6, "coil_current": 1.5}
    offered = [label.split(" · ")[0] for label in box.completer.model().stringList()]
    assert offered == ["tDelay1", "coil_current"], "tPulse is declared as pulse"
    box.choose("tPulse")
    assert set(editor.rows) == {"pulse"}

    editor.show()
    QTest.keyClicks(box.search, "del")
    assert box.completer.completionCount() == 1
    QTest.keyClick(box.completer.popup(), QtCore.Qt.Key.Key_Down)
    QTest.keyClick(box.completer.popup(), QtCore.Qt.Key.Key_Return)
    editor.close()
    assert box.search.text() == ""
    row = editor.rows["rb.tDelay1"]
    assert row.schema["kind"] == "integer" and row.value.value() == 5
    assert editor.request().args == {"pulse": 1e-6, "rb.tDelay1": 5}


def test_a_searched_parameter_can_be_scanned_removed_and_reset(qapp):
    editor = sequenced_editor(Source(PARAMETERS))
    box = editor.motmaster_boxes["rb"]
    box.choose("coil_current")
    row = editor.rows["rb.coil_current"]
    assert row.schema["kind"] == "number" and row.value.value() == 1.5
    row.scan_button.setChecked(True)
    assert [axis.argument for axis in editor.request().scan.axes] == ["rb.coil_current"]
    row.remove_button.click()
    assert "rb.coil_current" not in editor.rows
    assert editor.request().scan.axes == []
    assert editor.point_count.text() == "1 point"

    box.choose("tDelay1")
    editor.reset_to_defaults()
    assert set(editor.rows) == {"pulse"}


def test_load_request_restores_searched_rows_and_flags_missing_ones(qapp):
    editor = sequenced_editor(Source(PARAMETERS))
    editor.load_request(
        sequenced_request(
            {"rb.tDelay1": 9, "rb.gone": 1.5},
            [LinearAxis(argument="rb.coil_current", start=0, stop=2, n=3)],
        )
    )
    assert editor.rows["rb.tDelay1"].value.value() == 9
    assert editor.rows["rb.coil_current"].scanning
    assert "not in script" in editor.rows["rb.gone"].label.toolTip()
    assert "not in script" not in editor.rows["rb.tDelay1"].label.toolTip()
    assert editor.request().args == {"pulse": 1e-6, "rb.tDelay1": 9, "rb.gone": 1.5}


def test_an_unreachable_sequencer_shows_the_error_and_a_retry(qapp):
    source = Source(error="no route to host")
    editor = sequenced_editor(source)
    box = editor.motmaster_boxes["rb"]
    assert "no route to host" in box.status.text()
    assert not box.retry.isHidden()
    with pytest.raises(ValueError, match="rb.tDelay1"):
        editor.add_motmaster_row("rb", "tDelay1")
    assert "rb.tDelay1" not in editor.rows

    source.error, source.parameters = None, PARAMETERS
    box.retry.click()
    assert box.retry.isHidden() and "tDelay1" in box.defaults
    assert editor.request().args == {"pulse": 1e-6}


def test_a_threaded_fetch_is_delivered_on_the_gui_thread(qapp):
    fetcher = ParameterFetcher(Source(PARAMETERS))
    delivered = []
    fetcher.fetched.connect(
        lambda *args: delivered.append((args, threading.current_thread()))
    )
    fetcher.request("Rb MotMaster", "RbTweezerBasic")
    assert pump_until(qapp, lambda: delivered)
    assert delivered[0][0] == ("Rb MotMaster", "RbTweezerBasic", PARAMETERS)
    assert delivered[0][1] is threading.main_thread()


def test_the_panel_reads_simulated_sequencers_when_the_manager_simulates(qapp):
    feed = FakeFeed()
    panel = ExperimentsPanel(client=FakeClient(), feed=feed)
    feed.queue_changed.emit(
        {"running": None, "queue": [], "history": [], "simulated": True}
    )
    parameters = panel.editor.fetcher.source("Rb MotMaster", "RbTweezerBasic")
    assert "tDelay1" in parameters


def test_rows_restored_before_the_fetch_are_retyped_from_the_script(qapp):
    axis = LinearAxis(argument="rb.tDelay1", start=0, stop=4, n=5)
    editor, kinds_before = load_before_fetch(
        qapp, sequenced_request({"rb.coil_current": 2}, [axis])
    )
    assert kinds_before["rb.tDelay1"] == "number"
    assert kinds_before["rb.coil_current"] == "integer"

    delay, current = editor.rows["rb.tDelay1"], editor.rows["rb.coil_current"]
    assert (delay.schema["kind"], delay.schema["default"]) == ("integer", 5)
    assert (current.schema["kind"], current.schema["default"]) == ("number", 1.5)
    assert delay.scanning and editor.request().scan.axes == [axis]
    assert current.value.value() == 2.0
    assert editor.request().args == {"pulse": 1e-6, "rb.coil_current": 2.0}


def test_a_non_whole_value_for_an_integer_parameter_is_kept_and_reported(qapp):
    axis = LinearAxis(argument="rb.tDelay1", start=0, stop=2.5, n=2)
    editor, _ = load_before_fetch(qapp, sequenced_request(axes=[axis]))
    row = editor.rows["rb.tDelay1"]
    assert row.schema["default"] == 5 and row.scan.stop.value() == 2.5
    with pytest.raises(ValueError, match="rb.tDelay1 is an integer parameter"):
        editor.request()


def test_an_integer_parameter_scans_in_whole_steps_but_a_draft_keeps_any(qapp):
    client = FakeClient()
    panel = ExperimentsPanel(client=client, feed=FakeFeed())
    editor = panel.editor
    editor.fetcher.threaded = False
    editor.set_parameter_source(Source(PARAMETERS))
    editor.set_experiment(SEQUENCED)
    panel._current_key = (SEQUENCED["module"], "Sequenced")
    editor.motmaster_boxes["rb"].choose("tDelay1")
    row = editor.rows["rb.tDelay1"]
    row.scan_button.setChecked(True)
    row.scan.start.set_value(0)
    row.scan.stop.set_value(5)
    row.scan.steps.setValue(3)
    message = "rb.tDelay1 is an integer parameter: 0 to 5 in 3 steps gives non-whole"
    with pytest.raises(ValueError, match=message):
        editor.request()
    editor.submit_button.click()
    assert commands(client, "submit") == []
    assert "non-whole" in editor.error.text()
    panel._save_draft()
    assert panel._drafts[panel._current_key].scan.axes[0].argument == "rb.tDelay1"

    row.scan.stop.set_value(4)
    row.scan.steps.setValue(5)
    assert editor.request().scan.axes[0].raw_values() == [0, 1, 2, 3, 4]
    row.scan.mode.setCurrentText("list")
    row.scan.values.setText("1, 2, 7")
    assert editor.request().scan.axes[0].values == [1, 2, 7]


def recipe_dict(name="check", **kwargs):
    return Recipe(
        experiment=SCHEMA["module"],
        class_name="Demo",
        name=name,
        submitter="me@pc",
        **kwargs,
    ).model_dump(mode="json")


def demo_item(view):
    for i in range(view.tree.topLevelItemCount()):
        module_item = view.tree.topLevelItem(i)
        for j in range(module_item.childCount()):
            if module_item.child(j).text(0) == "Demo":
                return module_item.child(j)
    raise AssertionError("Demo not in the tree")


def catalogue_with(*recipes):
    view = CatalogueView()
    view.set_modules(MODULES)
    view.set_recipes(list(recipes))
    return view


def recipe_panel(*recipes):
    client, feed = FakeClient(), FakeFeed()
    client.recipes = list(recipes)
    panel = ExperimentsPanel(client=client, feed=feed)
    panel.refresh_catalogue()
    return panel, client, feed


def test_recipes_appear_under_their_experiment_and_filter(qapp):
    view = catalogue_with(recipe_dict("MOT check", args={"shots": 5}))
    demo = demo_item(view)
    assert demo.childCount() == 1
    recipe_item = demo.child(0)
    assert recipe_item.text(0) == "MOT check"

    view.filter.setText("mot ch")
    assert not recipe_item.isHidden() and not demo.isHidden()
    view.filter.setText("Demo")
    assert not recipe_item.isHidden()
    view.filter.setText("nothing like it")
    assert recipe_item.isHidden() and demo.isHidden()


def test_selecting_a_recipe_emits_it_and_a_refresh_keeps_it_quietly(qapp):
    view = catalogue_with(recipe_dict("a"), recipe_dict("b"))
    selected, experiments = [], []
    view.recipe_selected.connect(selected.append)
    view.experiment_selected.connect(experiments.append)
    assert view.select_recipe(SCHEMA["module"], "Demo", "b")
    assert [r["name"] for r in selected] == ["b"] and experiments == []
    assert view.selected_key() == (SCHEMA["module"], "Demo")

    view.set_recipes([recipe_dict("a"), recipe_dict("b"), recipe_dict("c")])
    view.set_modules(MODULES)
    assert len(selected) == 1 and experiments == []
    assert view.tree.currentItem().text(0) == "b"


def test_the_recipe_menu_submits_or_deletes(qapp):
    recipe = recipe_dict()
    view = catalogue_with(recipe)
    actions = []
    view.recipe_action_requested.connect(lambda *args: actions.append(args))
    for action in view.recipe_menu(recipe).actions():
        if not action.isSeparator():
            action.trigger()
    assert [a[0] for a in actions] == ["submit", "delete"]
    assert actions[0][1]["name"] == "check"


def test_panel_lists_recipes_and_refetches_when_they_change(qapp):
    panel, client, feed = recipe_panel(recipe_dict("a"))
    assert [r["name"] for r in panel.catalogue.recipes] == ["a"]
    fetches = len(commands(client, "recipes"))
    snapshot = {"running": None, "queue": [], "history": [], "recipes_version": 0}
    feed.queue_changed.emit(snapshot)
    assert len(commands(client, "recipes")) == fetches
    client.recipes = [recipe_dict("a"), recipe_dict("b")]
    feed.queue_changed.emit({**snapshot, "recipes_version": 1})
    assert [r["name"] for r in panel.catalogue.recipes] == ["a", "b"]


def test_selecting_a_recipe_fills_the_form_and_a_list_refresh_keeps_edits(qapp):
    recipe = recipe_dict("check", args={"shots": 5, "mode": "b"}, label="nightly")
    panel, client, feed = recipe_panel(recipe)
    assert panel.catalogue.select_recipe(SCHEMA["module"], "Demo", "check")
    assert panel.editor.schema["class_name"] == "Demo"
    assert panel.editor.rows["shots"].value.value() == 5
    assert panel.editor.rows["mode"].value.value() == "b"
    assert panel.editor.label.text() == "nightly"
    assert panel.editor.recipe_name == "check"

    panel.editor.rows["shots"].value.widget.setValue(8)
    client.recipes = [recipe, recipe_dict("new")]
    feed.queue_changed.emit(
        {"running": None, "queue": [], "history": [], "recipes_version": 1}
    )
    assert panel.editor.rows["shots"].value.value() == 8
    assert panel.catalogue.tree.currentItem().text(0) == "check"


def test_save_as_recipe_asks_for_a_name_and_confirms_a_replacement(qapp):
    panel, client, _feed = recipe_panel(recipe_dict("check"))
    panel.catalogue.select(SCHEMA["module"], "Demo")
    asked, confirmed = [], []
    panel.ask_recipe_name = lambda default: asked.append(default) or " new one "
    panel.confirm = lambda title, question: confirmed.append(question) or True
    panel.editor.rows["shots"].value.widget.setValue(4)
    panel.editor.save_recipe_button.click()
    [fields] = commands(client, "save_recipe")
    assert fields["recipe"]["name"] == "new one" and not fields["overwrite"]
    assert fields["recipe"]["args"]["shots"] == 4
    assert "@" in fields["recipe"]["submitter"]
    assert confirmed == [] and asked == [""]
    assert panel.editor.recipe_name == "new one"

    panel.ask_recipe_name = lambda default: "check"
    panel.confirm = lambda title, question: False
    panel.editor.save_recipe_button.click()
    assert len(commands(client, "save_recipe")) == 1
    panel.confirm = lambda title, question: True
    panel.editor.save_recipe_button.click()
    assert commands(client, "save_recipe")[-1]["overwrite"] is True

    panel.ask_recipe_name = lambda default: "  "
    panel.editor.save_recipe_button.click()
    assert len(commands(client, "save_recipe")) == 2


def test_submit_now_queues_the_recipe_or_shows_why_not(qapp):
    recipe = recipe_dict("check")
    panel, client, _feed = recipe_panel(recipe)
    panel.catalogue.recipe_action_requested.emit("submit", recipe)
    [fields] = commands(client, "submit_recipe")
    assert (fields["experiment"], fields["class_name"], fields["name"]) == (
        SCHEMA["module"],
        "Demo",
        "check",
    )
    assert "@" in fields["submitter"]
    assert "13" in panel.status.text()

    client.errors["submit_recipe"] = "Demo no longer has argument(s) ['old']"
    panel.catalogue.recipe_action_requested.emit("submit", recipe)
    assert "no longer has argument(s) ['old']" in panel.status.text()


def test_deleting_a_recipe_needs_confirmation(qapp):
    recipe = recipe_dict("check")
    panel, client, _feed = recipe_panel(recipe)
    panel.confirm = lambda title, question: False
    panel.catalogue.recipe_action_requested.emit("delete", recipe)
    assert commands(client, "delete_recipe") == []
    panel.confirm = lambda title, question: True
    panel.catalogue.recipe_action_requested.emit("delete", recipe)
    assert commands(client, "delete_recipe") == [
        {"experiment": SCHEMA["module"], "class_name": "Demo", "name": "check"}
    ]


def test_clicks_on_the_catalogue_only_reload_a_recipe_on_a_left_click(qapp):
    panel, _client, _feed = recipe_panel(
        recipe_dict("check", args={"shots": 5}), recipe_dict("other")
    )
    panel.resize(600, 800)
    panel.show()
    qapp.processEvents()
    tree = panel.catalogue.tree
    demo = demo_item(panel.catalogue)
    check, other = demo.child(0), demo.child(1)

    def click(item, button):
        QTest.mouseClick(
            tree.viewport(), button, pos=tree.visualItemRect(item).center()
        )

    def shots():
        return panel.editor.rows["shots"].value

    panel.catalogue.select(SCHEMA["module"], "Demo")
    shots().widget.setValue(8)
    click(check, Right)
    assert shots().value() == 8 and panel.editor.recipe_name == ""
    assert tree.currentItem() is demo

    panel.catalogue.select_recipe(SCHEMA["module"], "Demo", "check")
    shots().widget.setValue(8)
    click(demo, Right)
    assert shots().value() == 8 and panel.editor.recipe_name == "check"
    assert tree.currentItem() is check

    click(check, Left)
    assert shots().value() == 5

    loaded = []
    panel.catalogue.recipe_selected.connect(loaded.append)
    click(other, Left)
    assert [r["name"] for r in loaded] == ["other"], "loaded exactly once"
