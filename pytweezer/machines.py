"""Run pytweezer commands on the lab PCs in ``MACHINES`` over SSH.

Every command leaves this process through :func:`run`, which uses the system
``ssh`` in batch mode (never prompts), or the local shell for this PC. Each
action returns a result object rather than printing. On Windows the remote shell
is ``cmd``, and PowerShell is passed with ``-EncodedCommand`` because a script
piped to ``powershell -Command -`` over SSH is unreliable.

Whether pytweezer runs on a PC is read from its GUI's single-instance lock
(``bin/single_instance.py``), not from its process list, which non-administrator
accounts may not read over SSH.
"""

import base64
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from pytweezer.configuration import config
from pytweezer.configuration.config import GUI_LOCK_PORT, this_pc_address
from pytweezer.logging_utils import get_logger

logger = get_logger("machines")

_SSH_OPTIONS = [
    "-o", "ConnectTimeout=5",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "LogLevel=ERROR",
]  # fmt: skip
_HOST_KEY_CHANGED = (
    "REMOTE HOST IDENTIFICATION HAS CHANGED",
    "Host key verification failed",
)
_UNREACHABLE = (
    "Connection timed out",
    "No route to host",
    "Connection refused",
    "Could not resolve",
)
_NO_REPO = "clone path unknown; run setup"
_NOT_SUPPORTED = "not supported"


@dataclass(frozen=True)
class Machine:
    """One lab PC, as ``MACHINES`` describes it."""

    name: str
    address: str
    user: str
    repo: str | None
    os: str
    launcher: str | None


@dataclass
class GuiState:
    """Whether a PC answered, and the lock's description of its GUI (``None`` if none runs).

    ``reach`` is ``"ok"``, ``"no key"``, ``"unreachable"``, ``"host key changed"``, or
    ``"no answer"`` when SSH got through but the lock check gave neither ``none``
    nor an owner (something holds the port without replying, such as a hung GUI).
    """

    reach: str
    owner: dict | None


@dataclass
class GitState:
    """A clone's branch, short commit, position relative to its upstream, and local changes."""

    branch: str
    commit: str
    behind: int
    ahead: int
    dirty: bool


@dataclass
class Result:
    """The outcome of an action on one machine."""

    machine: str
    ok: bool
    detail: str


def machines(names: list[str] | None = None) -> list[Machine]:
    """The named machines (case-insensitive), or all of them in ``MACHINES`` order.

    Raises:
        ValueError: a name is not in ``MACHINES``.
    """
    if not names:
        names = list(config.MACHINES)
    by_lower_name = {name.lower(): name for name in config.MACHINES}
    unknown = [name for name in names if name.lower() not in by_lower_name]
    if unknown:
        raise ValueError(
            f"Unknown machine {', '.join(unknown)}; the machines are "
            f"{', '.join(config.MACHINES)}"
        )
    return [
        Machine(
            name=by_lower_name[name.lower()],
            **config.MACHINES[by_lower_name[name.lower()]],
        )
        for name in names
    ]


def run(
    machine: Machine, command: str, timeout_s: float = 60
) -> subprocess.CompletedProcess[str]:
    """Run a shell command on ``machine``: over SSH, or locally if it is this PC.

    A command that outlasts ``timeout_s`` is reported with return code -1.
    """
    if machine.address == this_pc_address():
        # A string, not a list: Python would escape the command's quotes as \", which cmd keeps.
        args = (
            f'cmd /s /c "{command}"'
            if machine.os == "windows"
            else ["sh", "-c", command]
        )
    else:
        args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            *_SSH_OPTIONS,
            f"{machine.user}@{machine.address}",
            command,
        ]
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args, -1, "", f"timed out after {timeout_s:g} s"
        )


def gui_state(machine: Machine) -> GuiState:
    """Ask ``machine``'s GUI lock who holds it, without disturbing the GUI."""
    result = run(machine, _lock_command(machine), timeout_s=20)
    reach = _reach(result)
    if reach != "ok":
        return GuiState(reach, None)
    answer = _last_line(result.stdout)
    if result.returncode == 0 and answer == "none":
        return GuiState("ok", None)
    try:
        owner = json.loads(answer)
    except ValueError:
        owner = None
    if result.returncode != 0 or not isinstance(owner, dict):
        logger.warning(
            "Could not read the GUI lock on %s: %s",
            machine.name,
            _last_line(result.stderr) or answer or "no answer",
        )
        return GuiState("no answer", None)
    return GuiState("ok", owner)


def git_state(machine: Machine) -> GitState | Result:
    """``machine``'s clone after a fetch, or a failed ``Result`` if it can't be read."""
    if machine.repo is None:
        return Result(machine.name, False, _NO_REPO)
    result = run(
        machine,
        _in_repo(
            machine, ["git fetch -q origin", "git status --porcelain=v2 --branch"]
        ),
    )
    if result.returncode != 0:
        return _failure(machine, result)
    headers = {}
    dirty = False
    for line in result.stdout.splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition(" ")
            headers[key] = value
        elif line.strip():
            dirty = True
    ahead, behind = (
        (int(count[1:]) for count in headers["branch.ab"].split())
        if "branch.ab" in headers
        else (0, 0)
    )
    return GitState(
        branch=headers.get("branch.head", "?"),
        commit=headers.get("branch.oid", "?")[:7],
        behind=behind,
        ahead=ahead,
        dirty=dirty,
    )


def update_steps(current: str, requested: str | None) -> list[str] | str:
    """The git commands that bring a clone on ``current`` up to date, or why to skip it.

    With no ``requested`` branch only a clone on ``main`` is updated: a branch is
    switched only when asked for by name.
    """
    if requested is None:
        if current != "main":
            return f"on {current}; pass --branch to switch"
        requested = "main"
    if current == requested:
        return ["git pull --ff-only"]
    return [f"git checkout {requested}", "git pull --ff-only"]


def update(machine: Machine, branch: str | None = None) -> Result:
    """Fetch, fast-forward (switching to ``branch`` if given) and ``uv sync``."""
    if machine.repo is None:
        return Result(machine.name, False, _NO_REPO)
    head = run(
        machine,
        _in_repo(
            machine, ["git rev-parse --abbrev-ref HEAD", "git rev-parse --short HEAD"]
        ),
    )
    if head.returncode != 0:
        return _failure(machine, head)
    current_branch, old_commit = head.stdout.split()[:2]
    steps = update_steps(current_branch, branch)
    if isinstance(steps, str):
        return Result(machine.name, False, steps)
    result = run(
        machine,
        _in_repo(
            machine,
            [
                "git fetch -q origin",
                *steps,
                f"{_uv(machine)} sync",
                "git rev-parse --short HEAD",
            ],
        ),
        timeout_s=600,
    )
    if result.returncode != 0:
        return _failure(machine, result)
    new_commit = _last_line(result.stdout)
    if new_commit == old_commit:
        return Result(machine.name, True, "already up to date")
    return Result(machine.name, True, f"{old_commit} → {new_commit}")


def start(machine: Machine) -> Result:
    """Run ``machine``'s launch task, which starts its GUI on the logged-on desktop."""
    if machine.os != "windows" or machine.launcher is None:
        return Result(machine.name, False, _NOT_SUPPORTED)
    result = run(machine, f"schtasks /run /tn pytweezer-{machine.launcher}")
    if "cannot find" in result.stdout + result.stderr:
        return Result(machine.name, False, "launch task missing; run setup")
    if result.returncode != 0:
        return _failure(machine, result)
    return Result(machine.name, True, "started")


def stop(machine: Machine, wait_s: float = 30) -> Result:
    """Ask ``machine``'s GUI to close as if its window were closed, then clear leftovers.

    MOTMaster is left running. A GUI still open after ``wait_s`` is reported,
    not killed.
    """
    if machine.os != "windows" or machine.launcher is None:
        return Result(machine.name, False, _NOT_SUPPORTED)
    if machine.repo is None:
        return Result(machine.name, False, _NO_REPO)
    state = gui_state(machine)
    if state.reach != "ok":
        return Result(machine.name, False, state.reach)
    if state.owner is None:
        return Result(machine.name, True, "not running")

    sent = run(machine, _lock_command(machine, request="quit"), timeout_s=20)
    if sent.returncode != 0:
        return _failure(machine, sent)
    deadline = time.monotonic() + wait_s
    while True:
        time.sleep(min(2, wait_s))
        state = gui_state(machine)
        if state.reach == "ok" and state.owner is None:
            break
        if time.monotonic() >= deadline:
            return Result(machine.name, False, f"did not close within {wait_s:g} s")

    cleanup = run(
        machine, _in_repo(machine, [f"{_uv(machine)} run python -m bin.kill_stale"])
    )
    # uv also exits 2 on its own errors, so the exit code alone is not kill_stale's refusal.
    if cleanup.returncode == 2 and "Could not list processes" in cleanup.stdout:
        return Result(
            machine.name, True, "stopped; cleanup skipped (cannot list processes)"
        )
    if cleanup.returncode != 0:
        return Result(
            machine.name,
            False,
            f"cleanup failed after the GUI closed: {_failure_line(cleanup)}",
        )
    return Result(machine.name, True, f"stopped; {_last_line(cleanup.stdout)}")


def install_key(machine: Machine) -> Result:
    """Add this PC's public key to ``machine``'s accepted keys, asking for its password.

    The one SSH call allowed to prompt: run it from a terminal. On Windows an
    administrator's key must go in ``administrators_authorized_keys``, which
    sshd reads instead of the account's own file.
    """
    if machine.address == this_pc_address():
        return Result(machine.name, True, "this PC; no key needed")
    key_file = Path.home() / ".ssh" / "id_ed25519.pub"
    if not key_file.exists():
        return Result(
            machine.name, False, f"no key at {key_file}; run ssh-keygen -t ed25519"
        )
    key = key_file.read_text(encoding="utf-8").strip()
    script = _add_key_windows(key) if machine.os == "windows" else _add_key_linux(key)
    result = subprocess.run(
        ["ssh", *_SSH_OPTIONS, f"{machine.user}@{machine.address}", script],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return Result(
            machine.name,
            False,
            _last_line(result.stdout) or f"ssh exited with {result.returncode}",
        )
    return Result(machine.name, True, _last_line(result.stdout))


def register_launch_tasks(machine: Machine) -> Result:
    """Register the scheduled tasks that :func:`start` runs, from the clone's script."""
    if machine.os != "windows":
        return Result(machine.name, False, _NOT_SUPPORTED)
    if machine.repo is None:
        return Result(machine.name, False, "clone path unknown; add it to MACHINES")
    result = run(
        machine,
        "powershell -NoProfile -ExecutionPolicy Bypass "
        f'-File "{machine.repo}\\register_launch_tasks.ps1"',
    )
    if result.returncode != 0:
        return _failure(machine, result)
    return Result(machine.name, True, "launch tasks registered")


def _in_repo(machine: Machine, commands: list[str]) -> str:
    cd = (
        f'cd /d "{machine.repo}"' if machine.os == "windows" else f'cd "{machine.repo}"'
    )
    return " && ".join([cd, *commands])


def _uv(machine: Machine) -> str:
    """How to call uv in a command on ``machine``: it is not on every SSH session's PATH."""
    if machine.os != "windows":
        return '"$(command -v uv || echo ~/.local/bin/uv)"'
    found = run(machine, "where uv")
    if found.returncode == 0 and found.stdout.strip():
        return f'"{found.stdout.splitlines()[0].strip()}"'
    return r'"%USERPROFILE%\.local\bin\uv.exe"'


def _lock_command(machine: Machine, request: str | None = None) -> str:
    """A command printing the GUI lock's line, or ``none`` if nothing holds it.

    With ``request``, that line is then sent to the GUI.
    """
    if machine.os == "windows":
        statements = [
            "$c = New-Object System.Net.Sockets.TcpClient",
            f"try {{ $c.Connect('127.0.0.1', {GUI_LOCK_PORT}) }} catch {{ 'none'; exit 0 }}",
            "$c.ReceiveTimeout = 2000",
            "$s = $c.GetStream()",
            "(New-Object System.IO.StreamReader($s)).ReadLine()",
        ]
        if request:
            statements += [
                "$w = New-Object System.IO.StreamWriter($s)",
                f'$w.Write("{request}`n")',
                "$w.Flush()",
            ]
        return _powershell("; ".join([*statements, "$c.Close()"]))
    send = f"\n    s.sendall(b'{request}\\n')" if request else ""
    code = (
        "import socket\n"
        "try:\n"
        f"    s = socket.create_connection(('127.0.0.1', {GUI_LOCK_PORT}), 2)\n"
        "except OSError:\n"
        "    print('none')\n"
        "else:\n"
        "    s.settimeout(2)\n"
        f"    print(s.makefile().readline().strip()){send}"
    )
    return f"python3 -c {shlex.quote(code)}"


def _add_key_windows(key: str) -> str:
    quoted_key = key.replace("'", "''")
    return _powershell(
        f"$key = '{quoted_key}'\n"
        "$admin = [bool]((whoami /groups) -match 'S-1-5-32-544')\n"
        'if ($admin) { $file = "$env:ProgramData\\ssh\\administrators_authorized_keys" }\n'
        'else { $dir = "$env:USERPROFILE\\.ssh"; New-Item -ItemType Directory -Force $dir | Out-Null; '
        '$file = "$dir\\authorized_keys" }\n'
        "$text = if (Test-Path $file) { [IO.File]::ReadAllText($file) } else { '' }\n"
        'if (($text -split "`r?`n" | ForEach-Object { $_.Trim() }) -contains $key) '
        '{ "already in $file"; exit 0 }\n'
        'if ($text -and -not $text.EndsWith("`n")) { $key = "`r`n$key" }\n'
        '[IO.File]::AppendAllText($file, "$key`r`n")\n'
        "if ($admin) { icacls $file /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Null;"
        ' if ($LASTEXITCODE) { "icacls failed on $file"; exit 1 } }\n'
        '"added to $file"\n'
    )


def _add_key_linux(key: str) -> str:
    return (
        f"key={shlex.quote(key)}; "
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && "
        "chmod 600 ~/.ssh/authorized_keys && "
        'if grep -qxF "$key" ~/.ssh/authorized_keys; then echo "already in ~/.ssh/authorized_keys"; '
        'else echo "$key" >> ~/.ssh/authorized_keys && echo "added to ~/.ssh/authorized_keys"; fi'
    )


def _powershell(script: str) -> str:
    # Progress records reach a redirected stderr as CLIXML noise.
    script = "$ProgressPreference = 'SilentlyContinue'\n" + script
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"


def _reach(result: subprocess.CompletedProcess[str]) -> str:
    """Whether SSH got through, judged from its own failure messages (exit code 255)."""
    if result.returncode != 255:
        return "ok"
    if any(message in result.stderr for message in _HOST_KEY_CHANGED):
        return "host key changed"
    if "Permission denied" in result.stderr:
        return "no key"
    if not any(message in result.stderr for message in _UNREACHABLE):
        logger.warning("Unrecognised SSH failure: %s", _last_line(result.stderr))
    return "unreachable"


def _failure(machine: Machine, result: subprocess.CompletedProcess[str]) -> Result:
    reach = _reach(result)
    return Result(
        machine.name, False, _failure_line(result) if reach == "ok" else reach
    )


def _failure_line(result: subprocess.CompletedProcess[str]) -> str:
    """Why a command failed: git's or uv's first ``error:``/``fatal:`` line, else the last line.

    Git ends a refusal with "Aborting", so its last line alone says little.
    """
    for line in result.stderr.splitlines():
        if line.startswith(("error:", "fatal:")):
            return line.strip()
    return (
        _last_line(result.stderr)
        or _last_line(result.stdout)
        or f"exited with {result.returncode}"
    )


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""
