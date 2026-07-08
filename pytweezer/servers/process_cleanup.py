"""Cross-platform cleanup for orphaned pytweezer server/device/logger processes.

Every managed process is spawned as ``['python3', <script>, <name>]`` (see
``bin/managed_panel.py``/``bin/process_tile_base.py``). If the GUI that
launched them is killed rather than closed normally, these children can
outlive it and keep holding their ZMQ sockets, so the next
``pytweezer-server``/``pytweezer-client`` launch fails to bind with "Address
already in use". :func:`kill_stale_processes` finds and kills any such
leftover process by matching its command line against the script paths in
``CONFIG["Servers"]``/``["Devices"]``/``["Loggers"]``.

This matches on *process*, not *socket*, so it also catches the pure-ZMQ-
subscriber loggers that never bind a listening port at all.

``xsub_xpub.py``'s own ``_terminate_stale_instances`` does something similar
but POSIX-only (it scans ``/proc``) and only for hub instances of the same
name about to start. This module is the general, cross-platform version meant
to be run on demand, from any PC role, against every managed category.
"""

import json
import os
import signal
import subprocess
import time

from pytweezer.servers.configreader import ConfigReader, tweezerpath

MANAGED_CATEGORIES = ("Servers", "Devices", "Loggers")


def _managed_scripts():
    """Return ``{normcased_script_path: [(category, name), ...]}``.

    The path is built exactly the way ``ManagedRow``/``ProcessTile`` build it
    when they spawn the process (``tweezerpath + "/bin/" + script``, left
    unnormalized) so it lines up character-for-character (after
    case/slash-folding) with what actually shows up in the process's argv.
    """
    conf = ConfigReader.getConfiguration()
    scripts = {}
    for category in MANAGED_CATEGORIES:
        for name, params in conf.get(category, {}).items():
            script = params.get("script")
            if not script:
                continue
            raw_path = tweezerpath + "/bin/" + script
            scripts.setdefault(os.path.normcase(raw_path), []).append((category, name))
    return scripts


def _list_processes_windows():
    """Yield ``(pid, commandline)`` for every process, via PowerShell CIM."""
    result = subprocess.run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
        ],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return
    data = json.loads(result.stdout)
    if isinstance(data, dict):
        data = [data]
    for entry in data:
        pid = entry.get("ProcessId")
        cmdline = entry.get("CommandLine") or ""
        if pid is not None:
            yield int(pid), cmdline


def _list_processes_posix():
    """Yield ``(pid, commandline)`` for every process, via ``/proc``."""
    proc_dir = "/proc"
    try:
        pid_strs = os.listdir(proc_dir)
    except OSError:
        return
    for pid_str in pid_strs:
        if not pid_str.isdigit():
            continue
        try:
            with open(os.path.join(proc_dir, pid_str, "cmdline"), "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        except OSError:
            continue
        yield int(pid_str), cmdline


def _list_processes():
    if os.name == "nt":
        yield from _list_processes_windows()
    else:
        yield from _list_processes_posix()


def find_stale_processes():
    """Return ``(pid, category, name, commandline)`` for every running
    process whose command line matches a configured server/device/logger
    script, excluding this process itself."""
    scripts = _managed_scripts()
    if not scripts:
        return []
    this_pid = os.getpid()
    matches = []
    for pid, cmdline in _list_processes():
        if pid == this_pid or not cmdline:
            continue
        cmdline_norm = os.path.normcase(cmdline)
        for script_path, owners in scripts.items():
            if script_path not in cmdline_norm:
                continue
            for category, name in owners:
                if name in cmdline:
                    matches.append((pid, category, name, cmdline))
                    break
            else:
                # Script matched but no configured name did -- still a
                # leftover pytweezer process (e.g. a since-renamed entry).
                category, name = owners[0]
                matches.append((pid, category, f"{name} (script match only)", cmdline))
            break
    return matches


def kill_stale_processes(grace_s=2.0, dry_run=False):
    """Find and terminate every stale pytweezer server/device/logger process.

    Returns the ``(pid, category, name, commandline)`` list that was (or, if
    ``dry_run``, would be) killed.
    """
    matches = find_stale_processes()
    if not matches or dry_run:
        return matches

    pids = [m[0] for m in matches]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    if os.name == "posix":
        # SIGTERM is a graceful request here; give stragglers a grace period,
        # then SIGKILL. (On Windows, os.kill(..., SIGTERM) already maps to
        # TerminateProcess, so there's nothing softer to wait out.)
        deadline = time.time() + grace_s
        remaining = set(pids)
        while remaining and time.time() < deadline:
            remaining = {pid for pid in remaining if _posix_alive(pid)}
            if remaining:
                time.sleep(0.2)
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    return matches


def _posix_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True
