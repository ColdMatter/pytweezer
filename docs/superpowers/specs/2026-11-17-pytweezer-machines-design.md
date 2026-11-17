# Rolling out updates and starting/stopping pytweezer across the lab PCs

## Why

Getting a change onto every lab PC means logging into each one, finding its
clone, pulling, running `uv sync`, and starting or stopping the GUI by hand.
Every PC is different (account, clone path, branch, OS). On 2026-11-17, rolling
out one commit to four machines took an afternoon of SSH by hand. We want:

- one command, run from PH-BEAST, that shows every machine's state and updates
  them;
- start/stop of pytweezer on any PC from the Devices tab.

## Goals

- `pytweezer-machines status | update | setup | start | stop` over SSH, for one,
  several or all machines.
- The Devices tab grouped by machine, with a Start/Stop button on each machine
  line.
- One table in CONFIG for the lab's machines, replacing `HOSTS`: where each
  one is and how to reach it.

## Non-goals

- Safety checks before stopping or updating, beyond one confirmation click on
  Stop in the GUI. It is up to the user to make sure nothing is running that
  matters (user's decision).
- Working from every PC. Running from PH-BEAST is enough (user). Any PC whose
  SSH key is installed on the others gets it for free, but nothing more is
  done for that.
- Start/stop on the Linux laptop. It gets `status` and `update` only.
- Any new long-running process or agent on the PCs.

## Anti-goals

- Never discard local work: no `reset --hard`, no `checkout --force`, no
  `stash`. A machine git refuses to update is reported, not forced.
- Never switch branch unless told to.
- Never start a GUI from the SSH session itself: it gets no window and dies
  when SSH disconnects (see `register_launch_tasks.ps1`).

## Approaches considered

- **A. An SSH-based command (chosen for updates).** No new processes, and it
  builds on SSH and the existing launch tasks.
- **B. A Machines tab in the GUI for updates.** Friendlier, but more to build;
  it can sit on top of A later. The user chose A for updates and put start/stop
  in the existing Devices tab instead.
- **C. An agent process on every PC taking commands over the network.**
  Rejected: it must already be running before it can start anything, it can't
  easily update itself, and it would be an unauthenticated way to run commands
  on every lab PC.

## Design

### Machines replace `HOSTS`

Machines and hosts are one thing (user's decision). `HOSTS` in
`pytweezer/configuration/config.py` becomes a `MACHINES` table, keyed by
machine name, holding the address plus how to reach and run pytweezer there:

| Key | Meaning |
|---|---|
| `address` | The machine's IP address, as `HOSTS` held it; also the SSH address |
| `user` | SSH account |
| `repo` | Absolute path of the pytweezer clone on that machine |
| `os` | `"windows"` or `"linux"` |
| `launcher` | `"server"`, `"client"` or `None`: which launch task Start runs |

Current machines (from today's session and the lab book's *Networked devices*
page):

| Name | Address | User | Repo | OS | Launcher |
|---|---|---|---|---|---|
| PH-BEAST | 10.59.3.1 | tweez | `C:\Users\tweez\ColdMatter\pytweezer` | windows | server |
| ic-czc221cchs (Rb PC, left) | 10.59.3.2 | cafmot | `C:\Users\CaFMOT\ColdMatter\pytweezer` | windows | client |
| IC-CZC4287H3W (Rb PC, right) | 10.59.3.3 | cafmot | unknown, find with `setup` | windows | client |
| ph-bonesaw | 10.59.3.5 | cafmot | `C:\Users\cafmot\pytweezer` | windows | client |
| twalker-laptop | 100.101.42.103 (tailnet; CCMnet port 22 is closed) | twalker | `/home/twalker/RbCaF/pytweezer` | linux | None |

`localhost` stays available for simulation (`SERVER_HOST` uses it), but it is
not a machine: it is not listed by the command or the Devices tab.

Everything that reads `HOSTS` today reads `MACHINES` instead: device, logger
and server `host` entries (`MACHINES["PH-BEAST"]["address"]` or a small helper),
`this_pc_address()`, `SERVER_HOST`, and `DevicesPanel`'s address-to-name
lookup. `host` values stay IP addresses, so nothing that compares or connects
to them changes (my call: switching `host` to machine names would touch every
consumer of CONFIG for no gain here).

A machine's devices are those whose `host` equals the machine's address.

### The SSH layer: `pytweezer/machines.py`

Shared by the command and the GUI. It runs the system `ssh` with
`BatchMode=yes` (never prompts) and a short connect timeout, as
`user@address`. It has one function per action, each returning a result
object, not printing. On Windows the remote shell is `cmd`; quote for it.

- **Is pytweezer running?** The GUI's single-instance lock
  (`bin/single_instance.py`) listens on `127.0.0.1:7277` and, on connect,
  writes one JSON line `{role, user, pid, since}`. Connect on the remote
  machine, read that line, close. Use this rather than listing processes:
  `Get-CimInstance Win32_Process` was refused ("Access denied") over SSH on
  ic-czc221cchs. Simply connecting and reading does not raise the window; only
  a line sent back does.
- **Start:** `schtasks /run /tn pytweezer-<launcher>`. If the task does not
  exist, say to run `setup`.
- **Stop:** ask the GUI to close itself, exactly as if its window had been
  closed. A new `quit` line on the lock socket, alongside the existing raise
  request, triggers the GUI's normal close. Its usual teardown stops its device
  servers and loggers; MOTMaster is not touched and stays running, as after
  any normal close (user's decision). This needs no access to other processes,
  which matters because on ic-czc221cchs `cafmot` is not an administrator and
  both `Get-CimInstance Win32_Process` and `tasklist` are refused over SSH.
  If the lock is still held after 30 s, Stop reports that the GUI did not close
  (my call: no automatic force-kill, since a force-kill of the whole tree would
  take MOTMaster with it). `TabbedGUI.closeEvent` (`bin/gui.py`) has no
  confirm-on-close dialog today; a `quit` request must never wait on one.
  Then run kill-stale in the clone (`uv run python -m bin.kill_stale`, not the
  `.exe` launcher, because of Smart App Control) for leftovers. That is best
  effort: kill-stale lists processes with `Get-CimInstance`, so where that is
  refused (ic-czc221cchs) it finds nothing, and Stop reports the cleanup as
  skipped rather than "nothing found". Kill-stale already leaves MOTMaster
  alone; keep it that way.
- **Git state:** branch, short commit, `behind`/`ahead` of `origin/<branch>`
  after a fetch, and whether the working tree is dirty (ignored files don't
  count).
- **Update to branch B** (default `main`): fetch. If the machine is on another
  branch, skip it with that reason, unless B was given explicitly; then
  `git checkout B`, which git itself refuses if local changes would be lost.
  Then `git pull --ff-only`, then `uv sync`. Any git or uv failure is recorded
  with git's/uv's own message. Find `uv` on the remote with `where uv` or
  `command -v uv`, falling back to `~/.local/bin/uv`: it isn't on the SSH
  session's PATH everywhere.

### The command: `pytweezer-machines`

A console script in `[project.scripts]`. Like the other entry points it can also be run
as `uv run python -m bin.machines` (Smart App Control).

```
pytweezer-machines status [MACHINE ...]
pytweezer-machines update [MACHINE ...] [--branch B]
pytweezer-machines start  MACHINE ...
pytweezer-machines stop   MACHINE ...
pytweezer-machines setup  MACHINE
```

- No machine names means all of them. Names match case-insensitively.
- Machines are handled in parallel, and each command prints one table: machine,
  result, detail. Example `status`:

  ```
  machine         reach  branch                      commit   behind ahead dirty  pytweezer
  PH-BEAST        ok     main                        c351ace  0      0     no     server (tweez, since 09:12)
  ic-czc221cchs   ok     main                        c351ace  0      0     no     not running
  IC-CZC4287H3W   no key -
  ph-bonesaw      ok     experiments-use-live-mode   c00ce0e  9      1     no     client (cafmot, since 10 Nov)
  twalker-laptop  ok     main                        c351ace  0      0     no     -
  ```
- `update` does not check whether pytweezer is running (user's decision).
- `start`/`stop` on a machine with no launcher, or on Linux, report "not
  supported" for that machine.
- `setup MACHINE` is interactive and is the only thing that may prompt for a
  password. It installs this PC's public key: for an account in
  `BUILTIN\Administrators` (`S-1-5-32-544` in `whoami /groups`; `cafmot` is on
  ph-bonesaw but not on ic-czc221cchs), in
  `C:\ProgramData\ssh\administrators_authorized_keys` with sshd's required
  permissions; otherwise in `~\.ssh\authorized_keys`; on Linux, in
  `~/.ssh/authorized_keys`. It does this through one password-prompting SSH
  call. It then registers the launch tasks by running `register_launch_tasks.ps1`
  in the remote clone, over the now key-based SSH. It can also print the clone
  path if `repo` is unknown, by searching the usual places.

### Devices tab

`DevicesPanel` (`bin/managed_panel.py`) lists device rows grouped under a
**machine line** per machine that hosts at least one device, in `MACHINES`
order. Machines with no devices (IC-CZC4287H3W, the laptop today) get no line
(user's decision); `pytweezer-machines start` still reaches them. Device rows
whose host matches no machine go under a final "Other" group.

A machine line shows the machine name, its address, and pytweezer's state on
it: running as server/client (user and since), not running, or unreachable.
It also has a Start/Stop button:

- **Start** runs the launch task. **Stop** asks once for confirmation ("Stop
  pytweezer on ph-bonesaw? Its GUI and devices will close.") and then stops it
  as above.
- The line for the PC the GUI runs on shows its state but has no button.
- In a simulated session the buttons are disabled (my call): a simulated GUI on
  a PC with keys would otherwise stop the real lab.
- The button is disabled, with the tooltip "Run `pytweezer-machines setup
  <name>` on this PC first", when this PC's key is not accepted by the machine.
  It is also disabled for Linux machines and for machines with no launcher.
- The state comes from the SSH check, run every 15 s per machine on a worker
  thread (the GUI-thread rules in the `pytweezer-gui-internals` skill), and
  right after a Start/Stop. While a start or stop is pending the line says
  "Starting…"/"Stopping…" until the check agrees or 60 s pass.

Device rows keep everything they do today (their own toggle, the Commandhub
remote requests, sub-device rows).

## Left to the builder

- The exact layout and styling of the machine line, following the
  `pytweezer-gui-design` skill.
- How `machines.py` runs the remote check on port 7277 (a short PowerShell
  `TcpClient` snippet, or Python in the remote venv).
- Tests, per the `write-tests` skill. The SSH layer gets a fake `ssh` runner.
  Worth testing: parsing of git/lock output into results; the update decision
  (skip on another branch, checkout only when the branch is named); grouping of
  devices under machines, including "Other".
- Updating the `run-pytweezer` skill: its "Restarting a GUI on another lab PC
  over SSH" section becomes "use `pytweezer-machines`", and the stale
  `rb-mm-pc` alias goes. Updating the `pytweezer-architecture` skill for
  `MACHINES` and the `pytweezer-gui-internals` skill for the grouped
  Devices tab.
