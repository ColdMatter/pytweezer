"""The SSH layer that updates, starts and stops pytweezer on the lab PCs."""

import subprocess

import pytest

from pytweezer import machines
from pytweezer.machines import GuiState, Machine

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
