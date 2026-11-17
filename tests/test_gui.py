"""Headless GUI tests. ``__new__`` is used where a real panel's construction
would have side effects (a DevicesPanel spawns subprocesses for active devices).
"""

import pytest

from bin.managed_panel import DevicesPanel, ManagedRow


def test_devices_panel_check_host_is_case_and_whitespace_insensitive():
    panel = DevicesPanel.__new__(DevicesPanel)
    panel.host_addr = "PH-BEAST"
    assert panel.check_host("  ph-beast  ") is True
    assert panel.check_host("PH-BEAST2") is False
    assert panel.check_host(None) is False


def test_machine_name_finds_the_machine_with_that_address():
    from pytweezer.configuration.config import machine_name

    assert machine_name("10.59.3.5") == "ph-bonesaw"
    assert machine_name("10.9.9.9") is None
    assert machine_name(None) is None


def test_this_pc_address_ignores_hostname_case(monkeypatch):
    from pytweezer.configuration import config

    monkeypatch.setattr(config, "MACHINES", {"IC-CZC4287H3W": {"address": "10.59.3.2"}})
    monkeypatch.setattr(config.socket, "gethostname", lambda: "ic-czc4287h3w")
    assert config.this_pc_address() == "10.59.3.2"
    monkeypatch.setattr(config.socket, "gethostname", lambda: "elsewhere")
    assert config.this_pc_address() is None


@pytest.fixture
def make_row(qapp):
    rows = []

    def make(**kwargs):
        rows.append(ManagedRow("Row", script="does_not_exist.py", **kwargs))
        return rows[-1]

    yield make
    for row in rows:
        if row.timer is not None:
            row.timer.stop()


def test_managed_row_toggle_reflects_controllability_and_state(make_row):
    not_started = make_row(run_at_startup=False)
    assert not_started.toggleButton.text() == "Start"
    assert not_started.process is None

    assert make_row(controllable=False).toggleButton is None

    # A composite whose coordinator stood down is still a running process.
    degraded = make_row()
    degraded.apply_status("degraded")
    assert degraded.toggleButton.text() == "Stop"


def test_client_control_panel_marks_portless_servers_unprobeable(qapp, monkeypatch):
    from bin import managed_panel

    config = {"Servers": {"Datalogger": {"script": "x.py", "host": "10.0.0.1"}}}
    monkeypatch.setattr(managed_panel, "get_config", lambda: config)
    panel = managed_panel.ControlPanel("Servers", "Servers", controllable=False)

    row = panel._rows["Datalogger"]
    assert row.property("state") == "unprobeable"
    assert row.stateLabel.text() == "n/a"


_PANEL_CONFIG = {
    "Devices": {
        "Rig": {
            "host": "10.0.0.1",  # not this machine -> nothing is ever spawned
            "port": 1,
            "devices": {
                "Rig SLM": {"class": "pkg:Slm"},
                "Rig Cam": {"class": "pkg:Cam"},
            },
            "coordinator": "pkg:Coord",
        },
    }
}


class _FakeControl:
    def __init__(self):
        self.sent = []

    def request(self, action, device):
        self.sent.append((action, device))

    @property
    def requested(self):
        return self

    def connect(self, _slot):
        pass

    def close(self):
        pass


@pytest.fixture
def devices_panel(qapp, monkeypatch):
    """Build a DevicesPanel from ``config`` without binding any ZMQ socket or running SSH."""
    from bin import machine_lines, managed_panel
    from pytweezer import machines
    from pytweezer.configuration import config as pytweezer_config

    control = _FakeControl()
    # Skip the cross-PC status feed rather than bind a ZMQ socket.
    monkeypatch.setattr(
        managed_panel,
        "DeviceStatusClient",
        lambda *a, **k: (_ for _ in ()).throw(OSError),
    )
    monkeypatch.setattr(managed_panel, "DeviceControlChannel", lambda: control)
    monkeypatch.setattr(managed_panel, "this_pc_address", lambda: None)
    monkeypatch.setattr(pytweezer_config, "SIMULATING", False)
    monkeypatch.setattr(machine_lines.MachineWorker, "start", lambda self: None)
    for action in ("gui_state", "start", "stop"):
        monkeypatch.setattr(
            machines, action, lambda *a, **k: pytest.fail("SSH run in a test")
        )

    def make(config):
        monkeypatch.setattr(managed_panel, "get_config", lambda: config)
        panel = managed_panel.DevicesPanel("Devices")
        return panel, control

    return make


def test_devices_panel_breaks_composites_into_sub_device_rows(devices_panel):
    panel, _control = devices_panel(_PANEL_CONFIG)

    assert set(panel._rows) == {"Rig", "Rig Cam", "Rig SLM"}
    assert panel._rows["Rig Cam"].property("tier") == "child"
    assert panel._rows["Rig"].property("tier") is None

    panel._apply_snapshot(
        {
            "Rig": {"state": "degraded", "host": "10.0.0.1", "port": 1},
            "Rig Cam": {"state": "up", "host": "10.0.0.1", "port": 1},
            "Rig SLM": {"state": "failed", "host": "10.0.0.1", "port": 1},
        }
    )
    assert [
        panel._rows[n].property("state") for n in ("Rig", "Rig Cam", "Rig SLM")
    ] == [
        "degraded",
        "up",
        "failed",
    ]


def test_sub_device_toggle_asks_the_running_rig_not_the_gui_channel(
    devices_panel, monkeypatch
):
    import threading

    from bin import managed_panel

    done = threading.Event()
    calls = []
    monkeypatch.setattr(
        managed_panel,
        "_run_rig_action",
        lambda device, action: (calls.append((action, device)), done.set()),
    )
    panel, control = devices_panel(_PANEL_CONFIG)
    cam = panel._rows["Rig Cam"]
    cam.apply_status("failed")

    cam.toggleButton.click()

    assert done.wait(5)
    assert calls == [("start", "Rig Cam")]
    assert control.sent == []
    assert cam.property("state") == "starting"


def test_remote_device_toggle_sends_requests_and_waits_for_the_feed(devices_panel):
    panel, control = devices_panel(_PANEL_CONFIG)
    rig = panel._rows["Rig"]
    assert rig.process is None

    rig.toggleButton.click()
    assert control.sent == [("start", "Rig")]
    assert rig.process is None
    rig.apply_status("down")
    assert rig.property("state") == "starting"
    rig.apply_status("up")
    assert rig.toggleButton.text() == "Stop"

    rig.toggleButton.click()
    assert control.sent[-1] == ("stop", "Rig")
    # The feed reads "up" until the host has acted on the stop.
    rig.apply_status("up")
    assert rig.property("state") == "stopping"
    rig.apply_status("down")
    assert rig.property("state") == "down"


def test_devices_panel_carries_out_requests_only_for_its_own_devices(
    devices_panel, monkeypatch
):
    from bin import managed_panel

    monkeypatch.setattr(managed_panel, "this_pc_address", lambda: None)  # -> 127.0.0.1
    config = {
        "Devices": {
            "Local Cam": {"host": "127.0.0.1", "port": 1},
            "Remote Cam": {"host": "10.0.0.1", "port": 2},
        }
    }
    panel, _control = devices_panel(config)
    calls = []
    for name, row in panel._rows.items():
        monkeypatch.setattr(
            row, "startProcess", lambda n=name: calls.append(("start", n))
        )
        monkeypatch.setattr(
            row, "stopProcess", lambda n=name: calls.append(("stop", n))
        )

    panel._handle_request("start", "Remote Cam", "PC-A")
    panel._handle_request("stop", "Local Cam", "PC-A")  # already stopped
    assert calls == []

    panel._handle_request("start", "Local Cam", "PC-A")
    assert calls == [("start", "Local Cam")]

    panel._rows["Local Cam"].apply_status("up")
    panel._handle_request("start", "Local Cam", "PC-A")  # already running
    panel._handle_request("stop", "Local Cam", "PC-A")
    assert calls == [("start", "Local Cam"), ("stop", "Local Cam")]


def test_a_device_process_that_dies_on_start_is_logged_with_its_exit_code(
    qapp, tmp_path, caplog
):
    # A native crash in a driver kills the process before Python can log anything,
    # so the GUI that spawned it is the only place left to record it.
    script = tmp_path / "dies.py"
    script.write_text("import os\nos._exit(3)\n")
    row = ManagedRow("Crashing Cam", script=str(script))
    row.startProcess()
    row.process.wait(timeout=10)

    row.apply_status("down")

    assert row.property("state") == "crashed"
    [record] = [r for r in caplog.records if r.levelname == "ERROR"]
    assert "Crashing Cam" in record.getMessage()
    assert "exit code 3" in record.getMessage()


def _crashing_row(tmp_path, delay_s=0.0):
    script = tmp_path / "dies.py"
    script.write_text(f"import os, time\ntime.sleep({delay_s})\nos._exit(3)\n")
    return ManagedRow("Crashing Cam", script=str(script))


def test_a_device_that_crashed_on_start_stays_crashed(qapp, tmp_path):
    row = _crashing_row(tmp_path)
    row.startProcess()
    row.process.wait(timeout=10)

    for _ in range(3):
        row.apply_status("down")

    assert row.property("state") == "crashed"


def test_a_device_that_crashes_while_running_shows_crashed(qapp, tmp_path):
    row = _crashing_row(tmp_path, delay_s=0.5)
    row.startProcess()
    row.apply_status("up")
    row.process.wait(timeout=10)

    row.apply_status("down")
    row.apply_status("down")

    assert row.property("state") == "crashed"


def test_a_crashed_device_reads_running_once_the_feed_sees_it_up(qapp, tmp_path):
    row = _crashing_row(tmp_path)
    row.startProcess()
    row.process.wait(timeout=10)
    row.apply_status("down")

    row.apply_status("up")  # restarted by another GUI on this PC

    assert row.property("state") == "up"


def test_stopping_an_unowned_device_hands_it_to_stop_orphan(make_row):
    orphans = []
    row = make_row(stop_orphan=orphans.append)
    row.apply_status("up")  # running, but not spawned by this row
    row.toggleButton.click()
    assert orphans == ["Row"]
    assert row.property("state") == "stopping"

    # Closing a panel terminates rows; that must not hunt for orphans.
    other = make_row(stop_orphan=orphans.append)
    other.apply_status("up")
    other.terminateProcess()
    assert orphans == ["Row"]


_LAB = {
    "PH-BEAST": {
        "address": "10.59.3.1",
        "user": "tweez",
        "repo": r"C:\pytweezer",
        "os": "windows",
        "launcher": "server",
    },
    "IC-IDLE": {  # hosts no device, so gets no line
        "address": "10.59.3.3",
        "user": "cafmot",
        "repo": r"C:\pytweezer",
        "os": "windows",
        "launcher": "client",
    },
    "ph-bonesaw": {
        "address": "10.59.3.5",
        "user": "cafmot",
        "repo": r"C:\pytweezer",
        "os": "windows",
        "launcher": "client",
    },
}

_LAB_DEVICES = {
    "Devices": {
        "Stray Cam": {"host": "10.0.0.9", "port": 3},
        "Bonesaw SLM": {"host": "10.59.3.5", "port": 2},
        "Beast Cam": {"host": "10.59.3.1", "port": 1},
        "Bonesaw Cam": {"host": "10.59.3.5", "port": 4},
    }
}

_RUNNING = {"role": "client", "user": "cafmot", "pid": 42, "since": "09:12 on 17 Nov"}


@pytest.fixture
def lab_panel(devices_panel, monkeypatch):
    """A DevicesPanel over ``_LAB_DEVICES``, with ``_LAB`` as the machines."""
    from pytweezer.configuration import config as pytweezer_config

    monkeypatch.setattr(pytweezer_config, "MACHINES", _LAB)

    def make():
        panel, _control = devices_panel(_LAB_DEVICES)
        return panel

    return make


def _what_the_tab_lists(panel):
    from bin.machine_lines import MachineLine

    listed = []
    for i in range(panel.layout().count()):
        widget = panel.layout().itemAt(i).widget()
        if isinstance(widget, MachineLine):
            listed.append(("machine", widget.name))
        elif isinstance(widget, ManagedRow):
            listed.append(widget.name)
        elif widget is not None:
            listed.append(("heading", widget.text()))
    return listed


def test_devices_are_grouped_under_their_machine(lab_panel):
    panel = lab_panel()

    assert _what_the_tab_lists(panel) == [
        ("machine", "PH-BEAST"),
        "Beast Cam",
        ("machine", "ph-bonesaw"),
        "Bonesaw Cam",
        "Bonesaw SLM",
        ("heading", "Other"),
        "Stray Cam",
    ]


@pytest.mark.parametrize(
    "this_pc, simulating, reach, owner, expected",
    [
        ("10.59.3.5", False, "ok", None, None),
        ("10.59.3.1", True, "ok", None, ("Start", False)),
        ("10.59.3.1", False, "no key", None, ("Start", False)),
        ("10.59.3.1", False, "host key changed", None, ("Start", False)),
        ("10.59.3.1", False, "ok", _RUNNING, ("Stop", True)),
    ],
    ids=["own PC", "simulating", "no key", "host key changed", "running"],
)
def test_machine_line_buttons(
    lab_panel, monkeypatch, this_pc, simulating, reach, owner, expected
):
    from bin import managed_panel
    from pytweezer.configuration import config as pytweezer_config
    from pytweezer.machines import GuiState

    monkeypatch.setattr(managed_panel, "this_pc_address", lambda: this_pc)
    monkeypatch.setattr(pytweezer_config, "SIMULATING", simulating)
    line = lab_panel()._machine_lines["ph-bonesaw"]

    line.apply(GuiState(reach, owner), polled_at=0.0)

    if expected is None:
        assert line.button is None
    else:
        assert (line.button.text(), line.button.isEnabled()) == expected


def _running_bonesaw(lab_panel, monkeypatch, stop_result="stopped"):
    """ph-bonesaw's line with pytweezer running there, and the calls to ``stop``."""
    from pytweezer import machines

    stopped = []

    def stop(machine, wait_s=30):
        stopped.append(machine.name)
        return machines.Result(machine.name, stop_result == "stopped", stop_result)

    monkeypatch.setattr(machines, "stop", stop)
    line = lab_panel()._machine_lines["ph-bonesaw"]
    line.apply(machines.GuiState("ok", _RUNNING), polled_at=0.0)
    return line, stopped


@pytest.mark.parametrize("answer", ["No", "Yes"])
def test_stop_asks_before_stopping(lab_panel, monkeypatch, qapp, answer):
    from PyQt6.QtWidgets import QMessageBox

    from tests.helpers import until

    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda parent, title, text, *a: (
            asked.append(text),
            getattr(QMessageBox.StandardButton, answer),
        )[1],
    )
    line, stopped = _running_bonesaw(lab_panel, monkeypatch)

    line.button.click()

    assert asked == ["Stop pytweezer on ph-bonesaw? Its GUI and devices will close."]
    if answer == "Yes":
        assert until(lambda: stopped, qapp=qapp) == ["ph-bonesaw"]
    else:
        until(lambda: stopped, timeout=0.3, qapp=qapp)
        assert stopped == []


def test_a_failed_stop_says_why_on_its_line(lab_panel, monkeypatch, qapp):
    from PyQt6.QtWidgets import QMessageBox

    from tests.helpers import until

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes
    )
    line, _stopped = _running_bonesaw(
        lab_panel, monkeypatch, stop_result="did not close within 30 s"
    )

    line.button.click()

    # The reason is elided to fit the line, so only its start is certain.
    assert until(
        lambda: line.stateLabel.text().startswith("stop failed: did not"), qapp=qapp
    )
    assert line.stateLabel.toolTip() == "did not close within 30 s"
    assert (line.button.text(), line.button.isEnabled()) == ("Stop", True)


_STOPPED = ("poll", "ok", None)
_STILL_RUNNING = ("poll", "ok", _RUNNING)


@pytest.mark.parametrize(
    "steps, shown",
    [
        ([_STILL_RUNNING], "Stopping…"),
        ([_STOPPED], "Stopping…"),
        ([("wait", 120), _STILL_RUNNING], "Stopping…"),
        ([("reported",), _STOPPED], "not running"),
        (
            [("reported",), ("wait", 61), _STILL_RUNNING],
            "client · cafmot since 09:12 on 17 Nov",
        ),
    ],
    ids=[
        "poll disagrees",
        "poll agrees while stop still runs",
        "stop still running after 120 s",
        "stop reported then poll agrees",
        "gives up 60 s after the stop reported",
    ],
)
def test_stopping_holds_until_the_stop_has_reported_and_a_poll_agrees(
    lab_panel, monkeypatch, steps, shown
):
    # A stop runs kill_stale after the GUI has closed; starting a GUI before it
    # finishes would have the new GUI's devices killed.
    from PyQt6.QtWidgets import QMessageBox

    from bin import machine_lines
    from pytweezer.machines import GuiState, Result

    now = [1000.0]
    monkeypatch.setattr(machine_lines.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes
    )
    line, _stopped = _running_bonesaw(lab_panel, monkeypatch)
    line.requested.disconnect()  # the stop is reported by hand below
    line.button.click()

    for step in steps:
        if step[0] == "poll":
            line.apply(GuiState(step[1], step[2]), polled_at=now[0])
        elif step[0] == "wait":
            now[0] += step[1]
        else:
            line.action_finished(Result("ph-bonesaw", True, "stopped"))

    assert line.stateLabel.text() == shown
    assert line.button.isEnabled() == (shown != "Stopping…")


def test_stopping_the_server_pc_warns_that_the_shared_servers_stop(
    lab_panel, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    from pytweezer.configuration import config as pytweezer_config

    server_pc = {**_LAB["ph-bonesaw"], "launcher": "server"}
    monkeypatch.setattr(pytweezer_config, "MACHINES", {**_LAB, "ph-bonesaw": server_pc})
    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda parent, title, text, *a: (
            asked.append(text),
            QMessageBox.StandardButton.No,
        )[1],
    )
    line, _stopped = _running_bonesaw(lab_panel, monkeypatch)

    line.button.click()

    assert asked == [
        (
            "Stop pytweezer on ph-bonesaw? Its GUI and devices will close, and so "
            "will the shared servers (hubs, loggers, Experiment Manager) that every "
            "PC relies on."
        )
    ]


_REPORTED_AT = 1000.0


@pytest.mark.parametrize(
    "after_the_failure, still_shown",
    [
        (("poll", _REPORTED_AT - 5, "ok", None), True),
        (("poll", _REPORTED_AT + 5, "ok", _RUNNING), True),
        (("poll", _REPORTED_AT + 5, "ok", None), False),
        (("click",), False),
    ],
    ids=[
        "poll in flight when it failed",
        "later poll, state unchanged",
        "later poll, state changed",
        "click",
    ],
)
def test_a_failed_stop_stays_on_its_line_until_a_click_or_a_change(
    lab_panel, monkeypatch, after_the_failure, still_shown
):
    from PyQt6.QtWidgets import QMessageBox

    from bin import machine_lines
    from pytweezer.machines import GuiState, Result

    monkeypatch.setattr(machine_lines.time, "monotonic", lambda: _REPORTED_AT)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes
    )
    line, _stopped = _running_bonesaw(lab_panel, monkeypatch)
    line.requested.disconnect()  # the stop is reported by hand below
    line.button.click()
    line.action_finished(Result("ph-bonesaw", False, "did not close within 30 s"))

    if after_the_failure[0] == "poll":
        _, polled_at, reach, owner = after_the_failure
        line.apply(GuiState(reach, owner), polled_at=polled_at)
    else:
        line.button.click()

    assert line.stateLabel.text().startswith("stop failed") == still_shown


def test_a_pc_without_ssh_access_is_checked_every_five_minutes_or_on_a_click(
    qapp, monkeypatch
):
    from bin.machine_lines import MachineWorker
    from pytweezer import machines
    from tests.helpers import until

    reach = {"open-pc": "ok", "keyless-pc": "no key"}
    checked = []

    def gui_state(machine):
        checked.append(machine.name)
        return machines.GuiState(reach[machine.name], None)

    monkeypatch.setattr(machines, "gui_state", gui_state)
    monkeypatch.setattr(machines, "start", lambda m: machines.Result(m.name, False, ""))
    open_pc, keyless_pc = (
        machines.Machine(name, "10.0.0.9", "lab", r"C:\pytweezer", "windows", "client")
        for name in reach
    )
    worker = MachineWorker([open_pc, keyless_pc], interval_s=0.01)
    worker.start()
    try:
        assert until(lambda: checked.count("open-pc") >= 5)
        assert checked.count("keyless-pc") == 1

        worker.act("start", open_pc)

        assert until(lambda: checked.count("keyless-pc") == 2)
        polls = checked.count("open-pc")
        assert until(lambda: checked.count("open-pc") >= polls + 5)
        assert checked.count("keyless-pc") == 2
    finally:
        worker.stop()
        worker.wait(2000)
