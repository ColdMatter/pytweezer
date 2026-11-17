"""The SSH layer that updates, starts and stops pytweezer on the lab PCs."""

import io
import subprocess

import pytest

from bin import machines as machines_command
from pytweezer import machines
from pytweezer.configuration import config
from pytweezer.machines import GitState, GuiState, Machine, Result

LAB_PC = Machine(
    name="lab-pc",
    address="10.0.0.9",
    user="lab",
    repo=r"C:\Box\Working folder\pytweezer",
    os="windows",
    launcher="client",
)
RUNNING = {"role": "client", "user": "lab", "pid": 4321, "since": "09:12 on 17 Nov"}


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess("ssh", returncode, stdout, stderr)


class FakeRun:
    """Stands in for ``machines.run``: answers each command by the first matching needle."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.commands = []

    def __call__(self, machine, command, timeout_s=60):
        self.commands.append(command)
        for needle, result in self.answers:
            if needle in command:
                return result
        return completed()

    def ran(self, needle):
        return any(needle in command for command in self.commands)


@pytest.fixture
def fake_run(monkeypatch):
    def install(*answers):
        fake = FakeRun(answers)
        monkeypatch.setattr(machines, "run", fake)
        return fake

    return install


@pytest.mark.parametrize(
    ("current", "requested", "expected"),
    [
        ("main", None, ["git pull --ff-only"]),
        (
            "experiments-use-live-mode",
            None,
            "on experiments-use-live-mode; pass --branch to switch",
        ),
        ("x", "main", ["git checkout main", "git pull --ff-only"]),
        ("main", "main", ["git pull --ff-only"]),
    ],
)
def test_update_steps(current, requested, expected):
    assert machines.update_steps(current, requested) == expected


def test_unknown_machine_name_lists_the_known_ones():
    with pytest.raises(ValueError, match="PH-BEAST"):
        machines.machines(["no-such-pc"])


@pytest.mark.parametrize(
    ("stderr", "reach"),
    [
        ("lab@10.0.0.9: Permission denied (publickey,password).", "no key"),
        ("ssh: connect to host 10.0.0.9 port 22: Connection timed out", "unreachable"),
        ("Host key verification failed.", "host key changed"),
    ],
)
def test_ssh_failures_are_classified(fake_run, stderr, reach):
    fake_run(("", completed(255, stderr=stderr)))
    assert machines.gui_state(LAB_PC) == GuiState(reach, None)


def test_a_lock_check_without_a_valid_answer_is_no_answer(fake_run):
    fake_run(("", completed(0, stdout="")))
    assert machines.gui_state(LAB_PC) == GuiState("no answer", None)


def test_this_pc_runs_locally_not_over_ssh(monkeypatch):
    calls = []

    def fake_subprocess_run(args, **kwargs):
        calls.append(args)
        return completed()

    monkeypatch.setattr(machines, "this_pc_address", lambda: LAB_PC.address)
    monkeypatch.setattr(machines.subprocess, "run", fake_subprocess_run)
    machines.run(LAB_PC, "git status")
    [args] = calls
    command_line = args if isinstance(args, str) else " ".join(args)
    assert not command_line.startswith("ssh")


def test_update_skips_a_machine_on_another_branch_without_running_git_pull(fake_run):
    fake = fake_run(
        ("--abbrev-ref", completed(stdout="experiments-use-live-mode\nc00ce0e\n"))
    )
    result = machines.update(LAB_PC)
    assert not result.ok
    assert result.detail == "on experiments-use-live-mode; pass --branch to switch"
    assert not fake.ran("git pull")


def test_update_reports_gits_refusal(fake_run):
    refusal = (
        "error: Your local changes to the following files would be overwritten "
        "by merge:"
    )
    fake_run(
        ("--abbrev-ref", completed(stdout="main\nc351ace\n")),
        ("git pull", completed(1, stderr=refusal + "\n\tbin/gui.py\nAborting\n")),
    )
    result = machines.update(LAB_PC)
    assert not result.ok
    assert result.detail == refusal


@pytest.mark.parametrize("action", [machines.git_state, machines.update, machines.stop])
def test_unknown_repo_is_reported_not_used(fake_run, action):
    fake = fake_run()
    no_repo = Machine("new-pc", "10.0.0.8", "lab", None, "windows", "client")
    assert action(no_repo) == machines.Result(
        "new-pc", False, "clone path unknown; run setup"
    )
    assert fake.commands == []


def test_stop_reports_a_gui_that_does_not_close(fake_run, monkeypatch):
    fake_run()
    monkeypatch.setattr(machines, "gui_state", lambda machine: GuiState("ok", RUNNING))
    result = machines.stop(LAB_PC, wait_s=0.1)
    assert not result.ok
    assert result.detail == "did not close within 0.1 s"


def test_stop_reports_skipped_cleanup(fake_run, monkeypatch):
    states = iter([GuiState("ok", RUNNING), GuiState("ok", None)])
    monkeypatch.setattr(machines, "gui_state", lambda machine: next(states))
    fake_run(
        (
            "bin.kill_stale",
            completed(2, stdout="Could not list processes: Access denied\n"),
        )
    )
    result = machines.stop(LAB_PC, wait_s=0.1)
    assert result.ok
    assert result.detail.endswith("; cleanup skipped (cannot list processes)")


def test_repo_path_is_quoted_for_cmd(fake_run):
    fake = fake_run(("--abbrev-ref", completed(stdout="main\nc351ace\n")))
    machines.update(LAB_PC)
    assert fake.ran(r'cd /d "C:\Box\Working folder\pytweezer"')


def test_stop_reports_a_uv_failure_as_failed_cleanup(fake_run, monkeypatch):
    states = iter([GuiState("ok", RUNNING), GuiState("ok", None)])
    monkeypatch.setattr(machines, "gui_state", lambda machine: next(states))
    fake_run(("bin.kill_stale", completed(2, stderr="error: Failed to spawn\n")))
    result = machines.stop(LAB_PC, wait_s=0.1)
    assert not result.ok
    assert result.detail.startswith("cleanup failed")


LAB = {
    "zeta": {"address": "10.0.0.1", "user": "u", "repo": "C:/z", "os": "windows"},
    "alpha": {"address": "10.0.0.2", "user": "u", "repo": "C:/a", "os": "windows"},
    "gamma": {"address": "10.0.0.3", "user": "u", "repo": None, "os": "windows"},
}
LAB = {name: {**entry, "launcher": "client"} for name, entry in LAB.items()}


@pytest.fixture
def lab(monkeypatch):
    monkeypatch.setattr(config, "MACHINES", LAB)


def test_status_prints_one_row_per_machine_in_config_order(lab, monkeypatch, capsys):
    states = {
        "zeta": GuiState("ok", RUNNING),
        "alpha": GuiState("no key", None),
        "gamma": GuiState("ok", None),
    }
    monkeypatch.setattr(machines, "gui_state", lambda machine: states[machine.name])
    monkeypatch.setattr(
        machines,
        "git_state",
        lambda machine: GitState("main", "c351ace", 2, 0, False),
    )
    exit_code = machines_command.main(["status"])
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert [row[0] for row in rows] == ["machine", "zeta", "alpha", "gamma"]
    assert rows[1][1:] == [
        "ok",
        "main",
        "c351ace",
        "2",
        "0",
        "no",
        "client",
        "(lab,",
        "since",
        "09:12",
        "on",
        "17",
        "Nov)",
    ]
    assert rows[2] == ["alpha", "no", "key", "-"]
    assert rows[3][-2:] == ["not", "running"]
    assert exit_code == 1


def test_update_exit_code_is_one_when_any_machine_fails(lab, monkeypatch, capsys):
    monkeypatch.setattr(
        machines,
        "update",
        lambda machine, branch=None: Result(
            machine.name, machine.name != "alpha", "c351ace → d058350"
        ),
    )
    assert machines_command.main(["update", "zeta"]) == 0
    capsys.readouterr()
    assert machines_command.main(["update", "ZETA", "alpha"]) == 1
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert [row[:2] for row in rows[1:]] == [["zeta", "ok"], ["alpha", "failed"]]


def test_update_passes_the_branch_to_every_machine(lab, monkeypatch, capsys):
    branches = []
    monkeypatch.setattr(
        machines,
        "update",
        lambda machine, branch=None: (
            branches.append(branch) or Result(machine.name, True, "")
        ),
    )
    machines_command.main(["update", "--branch", "feature"])
    assert branches == ["feature"] * 3


@pytest.mark.parametrize("command", ["start", "stop"])
def test_start_and_stop_without_machine_names_are_an_error(lab, monkeypatch, command):
    monkeypatch.setattr(machines, command, pytest.fail)
    with pytest.raises(SystemExit) as exit_info:
        machines_command.main([command])
    assert exit_info.value.code == 2


def test_output_survives_a_console_that_cannot_print_arrows(lab, monkeypatch, capsys):
    monkeypatch.setattr(
        machines,
        "update",
        lambda machine, branch=None: Result(machine.name, True, "a → b"),
    )
    monkeypatch.setattr("sys.stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    assert machines_command.main(["update", "zeta"]) == 0


def test_setup_installs_the_key_then_registers_tasks_and_lists_clones_when_repo_unknown(
    lab, monkeypatch, capsys, fake_run
):
    calls = []
    monkeypatch.setattr(
        machines,
        "install_key",
        lambda m: calls.append("key") or Result(m.name, True, "added"),
    )
    monkeypatch.setattr(
        machines,
        "register_launch_tasks",
        lambda m: (
            calls.append("tasks")
            or Result(m.name, False, "clone path unknown; add it to MACHINES")
        ),
    )
    clone = r"C:\Users\u\pytweezer"
    fake = fake_run(("pytweezer", completed(stdout=clone + "\n")))
    assert machines_command.main(["setup", "gamma"]) == 1
    assert calls == ["key", "tasks"]
    assert clone in capsys.readouterr().out

    fake.commands.clear()
    machines_command.main(["setup", "zeta"])
    assert fake.commands == []
