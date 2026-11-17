# pytweezer-machines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command to see and update every lab PC's pytweezer over SSH, and Start/Stop of pytweezer per machine from the Devices tab.

**Architecture:** `HOSTS` becomes a `MACHINES` table in `config.py`. A new `pytweezer/machines.py` runs commands on machines over the system `ssh` (locally for this PC) and returns result objects; `bin/machines.py` is its command line and `DevicesPanel` its GUI. Remote status and stop go through the GUI's existing single-instance socket on `127.0.0.1:7277`, which gains a `quit` request.

**Tech Stack:** Python 3.11, PyQt6, OpenSSH client/server (Windows OpenSSH on lab PCs), git, uv, pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-pytweezer-machines-design.md`

## Global Constraints

- Work on branch `pytweezer-machines` with a PR to `main` (CLAUDE.md: features get a branch and a merge request); commit this plan as its first commit.
- Never discard local work: no `reset --hard`, no `checkout --force`, no `stash`. A machine git refuses to update is reported with git's own message.
- Never switch branch unless `--branch` names it.
- Never start a GUI from an SSH session itself; Start only runs the `pytweezer-<launcher>` scheduled task.
- Stop never touches MOTMaster and never force-kills the GUI's process tree.
- Every SSH call except `setup`'s key install uses `-o BatchMode=yes` and never prompts.
- Remote Windows shell is `cmd`; run uv as `uv run python -m ...`, never a `.exe` launcher (Smart App Control).
- Device/logger/server `host` values stay IP-address strings.
- Docstrings Google-style, UK English; comments only for the surprising (CLAUDE.md).

## Review Focus

1. A machine that is switched off or off the network: every command must report "unreachable" for it within the connect timeout and still finish the others.
2. A machine whose `repo` is `None` (IC-CZC4287H3W today): `status`/`update` report "clone path unknown; run setup", never `cd None`.
3. A GUI that ignores or never receives `quit` (hung, or an old version without `quit`): Stop reports "did not close within 30 s" and leaves it running.
4. `status`/`update` on PH-BEAST itself, run from PH-BEAST: runs locally, not over SSH to itself.
5. A branch name or path with spaces (`C:\Users\CaFMOT\ColdMatter\pytweezer` has none today, but `Box` paths do): commands must quote paths for `cmd`.

Each has a test in the task that owns the code (Tasks 3 and 4).

---

### Task 1: `MACHINES` replaces `HOSTS`

**Files:**
- Modify: `pytweezer/configuration/config.py` (HOSTS block at top, `SERVER_HOST`, every `HOSTS[...]` in CONFIG/NOT_IN_USE, `this_pc_address`)
- Modify: `bin/managed_panel.py:38,582`, `pytweezer/GUI/ni_cards/panel.py:15,342`
- Modify: `.claude/skills/pytweezer-architecture/SKILL.md` (mention `MACHINES`, `machine_name`, `LOCALHOST` where it lists `HOSTS`/`this_pc_address`)
- Test: `tests/test_gui.py:18-25`

**Interfaces:**
- Produces:
  - `MACHINES: dict[str, dict]`, each `{"address": str, "user": str, "repo": str | None, "os": "windows" | "linux", "launcher": "server" | "client" | None}`, in this order:
    - `"PH-BEAST"`: `10.59.3.1`, `tweez`, `C:\Users\tweez\ColdMatter\pytweezer`, windows, server
    - `"IC-CZC221CCHS"`: `10.59.3.2`, `cafmot`, `C:\Users\CaFMOT\ColdMatter\pytweezer`, windows, client
    - `"IC-CZC4287H3W"`: `10.59.3.3`, `cafmot`, `None`, windows, client
    - `"ph-bonesaw"`: `10.59.3.5`, `cafmot`, `C:\Users\cafmot\pytweezer`, windows, client
    - `"twalker-laptop"`: `100.101.42.103`, `twalker`, `/home/twalker/RbCaF/pytweezer`, linux, `None`
  - `LOCALHOST = "127.0.0.1"` (replaces `HOSTS["localhost"]`)
  - `machine_name(address: str | None) -> str | None` — the `MACHINES` key with that address, `None` if none
  - `this_pc_address() -> str | None` — unchanged behaviour, reads `MACHINES`

- [ ] **Step 1: Update the test**

In `tests/test_gui.py`, `test_this_pc_address_ignores_hostname_case` monkeypatches `config.MACHINES` to `{"IC-CZC4287H3W": {"address": "10.59.3.2"}}` instead of `HOSTS`. Add:

```python
def test_machine_name_finds_the_machine_with_that_address():
    from pytweezer.configuration.config import machine_name

    assert machine_name("10.59.3.5") == "ph-bonesaw"
    assert machine_name("10.9.9.9") is None
    assert machine_name(None) is None
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_gui.py -q -k "this_pc_address or machine_name"`
Expected: FAIL (`machine_name` / `MACHINES` missing)

- [ ] **Step 3: Implement**

Replace `HOSTS` with `MACHINES` and `LOCALHOST`; every `HOSTS["X"]` in CONFIG becomes `MACHINES["X"]["address"]`, `HOSTS["localhost"]` becomes `LOCALHOST`. `managed_panel` and `ni_cards/panel.py` use `machine_name()` for their address-to-name lookups. Update the architecture skill lines.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest tests/ -q`
Expected: all pass. `git grep -n "HOSTS"` returns nothing under `pytweezer/ bin/ tests/ .claude/`.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/configuration/config.py bin/managed_panel.py pytweezer/GUI/ni_cards/panel.py tests/test_gui.py .claude/skills/pytweezer-architecture/SKILL.md
git commit -m "Replace HOSTS with a MACHINES table"
```

---

### Task 2: The GUI closes on a `quit` request

**Files:**
- Modify: `bin/single_instance.py` (`GuiLock`), `bin/gui.py:263-285` (`_run`)
- Test: `tests/test_gui_single_instance.py`

**Interfaces:**
- Produces: `GuiLock.quit_requested` (`pyqtSignal()`), emitted when a client writes the line `quit`. Wire protocol for Task 3: connect to `127.0.0.1:7277`, read one JSON line `{role, user, pid, since}`; optionally write `quit\n`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_quit_request_asks_the_gui_to_close(qapp, port):
    running = claim_gui("client", port=port)
    quits = []
    running.quit_requested.connect(lambda: quits.append(True))
    raised = []
    running.raise_requested.connect(lambda: raised.append(True))

    def send_quit():
        with socket.create_connection(("127.0.0.1", port)) as s:
            s.makefile().readline()
            s.sendall(b"quit\n")

    thread = threading.Thread(target=send_quit)
    thread.start()
    assert until(lambda: quits, qapp=qapp)
    assert not raised
    thread.join()
    running.close()
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_gui_single_instance.py -q`
Expected: FAIL (`quit_requested` missing)

- [ ] **Step 3: Implement**

`GuiLock._read` emits `quit_requested` on the line `quit`. In `_run`, connect it to the same close path as `on_exit` (`win.close(); app.quit()`); share that code rather than duplicate it.

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/test_gui_single_instance.py tests/test_gui.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bin/single_instance.py bin/gui.py tests/test_gui_single_instance.py
git commit -m "Close the GUI on a quit request to its single-instance port"
```

---

### Task 3: `pytweezer/machines.py` — the SSH layer

**Files:**
- Create: `pytweezer/machines.py`
- Modify: `pytweezer/servers/process_cleanup.py:55-82`, `bin/kill_stale.py`, `bin/managed_panel.py:500-503` (`_stop_orphan`)
- Test: `tests/test_machines.py` (new), `tests/test_servers.py` (process lister)

**Interfaces:**
- Consumes: `MACHINES`, `this_pc_address()` (Task 1); the lock protocol (Task 2).
- Produces:
  - `@dataclass(frozen=True) Machine(name: str, address: str, user: str, repo: str | None, os: str, launcher: str | None)`
  - `machines(names: list[str] | None = None) -> list[Machine]` — all in `MACHINES` order when `names` is empty/None; case-insensitive; unknown name raises `ValueError` listing the known ones
  - `run(machine: Machine, command: str, timeout_s: float = 60) -> subprocess.CompletedProcess[str]` — the one place a command leaves this process: `ssh -o BatchMode=yes -o ConnectTimeout=5 user@address command`, or `cmd /c` / `sh -c` locally when `machine.address == this_pc_address()`. Tests monkeypatch this function.
  - `@dataclass GuiState(reach: str, owner: dict | None)` — `reach` is one of `"ok"`, `"no key"`, `"unreachable"`; `owner` is the lock's JSON or `None` when no GUI runs
  - `gui_state(machine) -> GuiState`
  - `@dataclass GitState(branch: str, commit: str, behind: int, ahead: int, dirty: bool)`
  - `@dataclass Result(machine: str, ok: bool, detail: str)` — what update/start/stop/setup return
  - `git_state(machine) -> GitState | Result` (a `Result` when it can't be read)
  - `update_steps(current: str, requested: str | None) -> list[str] | str` — git commands to run, or the reason to skip
  - `update(machine, branch: str | None = None) -> Result`
  - `start(machine) -> Result`, `stop(machine, wait_s: float = 30) -> Result`
  - `install_key(machine) -> Result` (interactive), `register_launch_tasks(machine) -> Result`
  - `process_cleanup.ProcessListingRefused(OSError)`; `bin.kill_stale` exits with code 2 and prints `Could not list processes: <reason>` when raised

Behaviour the tests pin (exact strings are what the table shows):

- `run` result with returncode 255 and stderr containing `Permission denied` → `"no key"`; `Connection timed out`, `No route to host`, `Connection refused` or `Could not resolve` → `"unreachable"`.
- `gui_state` sends a snippet that prints the lock's line or `none`: on Windows a PowerShell `System.Net.Sockets.TcpClient` one-liner (passed with `-EncodedCommand`, UTF-16LE base64); on Linux `python3 -c` with `socket`.
- `update_steps("main", None)` → `["git pull --ff-only"]`; `update_steps("experiments-use-live-mode", None)` → `"on experiments-use-live-mode; pass --branch to switch"`; `update_steps("x", "main")` → `["git checkout main", "git pull --ff-only"]`; `update_steps("main", "main")` → `["git pull --ff-only"]`.
- `update` runs `git fetch -q origin`, the steps, then uv sync (Windows: `where uv` else `%USERPROFILE%\.local\bin\uv.exe`; Linux: `command -v uv || ~/.local/bin/uv`), each joined with `&&` inside `cd /d "<repo>"` (Windows) or `cd "<repo>"` (Linux); failure → `Result(ok=False, detail=<last non-empty stderr/stdout line>)`; success detail `"<old> → <new>"` short commits, or `"already up to date"`.
- `repo is None` → `Result(ok=False, detail="clone path unknown; run setup")` for `git_state`/`update`/`stop`.
- `start`: Linux or `launcher is None` → `Result(False, "not supported")`; `schtasks` output containing `cannot find` → `Result(False, "launch task missing; run setup")`.
- `stop`: no GUI → `Result(True, "not running")`; otherwise send `quit`, poll `gui_state` every 2 s until `owner is None` or `wait_s`; timeout → `Result(False, "did not close within 30 s")`; then `uv run python -m bin.kill_stale`; its exit code 2 → detail ends `"; cleanup skipped (cannot list processes)"`.
- `install_key` is the only call without `BatchMode`: it pipes this PC's `~/.ssh/id_ed25519.pub` into a remote script that, on Windows, appends it to `%ProgramData%\ssh\administrators_authorized_keys` (then `icacls ... /inheritance:r /grant *S-1-5-32-544:F /grant *S-1-5-18:F`) when `whoami /groups` contains `S-1-5-32-544`, else to `%USERPROFILE%\.ssh\authorized_keys`; on Linux to `~/.ssh/authorized_keys` with modes 700/600. Never appends a key already present.
- `register_launch_tasks` runs `powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\register_launch_tasks.ps1"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_machines.py`, each with `run` monkeypatched to a fake that records commands and returns scripted `CompletedProcess`es:
  - `test_update_steps` (parametrised over the four cases above)
  - `test_unknown_machine_name_lists_the_known_ones`
  - `test_ssh_failures_are_classified` (parametrised: `Permission denied` → `"no key"`, `Connection timed out` → `"unreachable"`)
  - `test_this_pc_runs_locally_not_over_ssh` (monkeypatch `this_pc_address` to the machine's address; patch `subprocess.run`; assert the argv does not start with `ssh`)
  - `test_update_skips_a_machine_on_another_branch_without_running_git_pull`
  - `test_update_reports_gits_refusal` (pull returns 1 with `error: Your local changes...` → `ok is False`, detail is that line)
  - `test_unknown_repo_is_reported_not_used` (asserts no command was run)
  - `test_stop_reports_a_gui_that_does_not_close` (`wait_s=0.1`, `gui_state` keeps returning an owner)
  - `test_stop_reports_skipped_cleanup` (kill_stale returns code 2)
  - `test_repo_path_is_quoted_for_cmd` (repo `C:\Box\Working folder\pytweezer` appears inside double quotes)

  In `tests/test_servers.py`: `test_a_refused_process_listing_raises` — monkeypatch `subprocess.run` in `process_cleanup` to return code 1, `os.name` path Windows; `find_stale_processes()` raises `ProcessListingRefused`.

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_machines.py tests/test_servers.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement** `pytweezer/machines.py` per the interfaces; `_list_processes_windows` raises `ProcessListingRefused(result.stderr.strip())` on a non-zero return; `bin/kill_stale.py` catches it (exit 2); `_stop_orphan`'s thread logs it with `logger.warning` instead of dying with a traceback.

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/machines.py pytweezer/servers/process_cleanup.py bin/kill_stale.py bin/managed_panel.py tests/test_machines.py tests/test_servers.py
git commit -m "Add the SSH layer for running pytweezer commands on lab machines"
```

---

### Task 4: `pytweezer-machines` command

**Files:**
- Create: `bin/machines.py`
- Modify: `pyproject.toml` `[project.scripts]` (`pytweezer-machines = "bin.machines:main"`)
- Modify: `.claude/skills/run-pytweezer/SKILL.md` ("Restarting a GUI on another lab PC over SSH" → use `pytweezer-machines`; drop the `rb-mm-pc` alias line), `CLAUDE.md` entry-point list (one line)
- Test: `tests/test_machines.py`

**Interfaces:**
- Consumes: everything Task 3 produces.
- Produces: `main(argv: list[str] | None = None) -> int`; subcommands `status [MACHINE ...]`, `update [MACHINE ...] [--branch B]`, `start MACHINE ...`, `stop MACHINE ...`, `setup MACHINE`. Exit code 0 when every machine's result is ok, 1 otherwise.

`status` columns, in order: `machine reach branch commit behind ahead dirty pytweezer`; pytweezer column is `server (tweez, since 09:12 on 17 Nov)` / `not running` / `-` (unreachable or Linux without lock). Other subcommands print `machine result detail`, result `ok`/`failed`. Machines run in a `ThreadPoolExecutor`; rows print in `MACHINES` order. `setup` runs `install_key` then `register_launch_tasks` (Windows only), and if `repo is None` prints the pytweezer clones it finds under `C:\Users\*\` and `C:\Users\*\ColdMatter\` (`dir /b /s /ad` limited to those two depths) for the user to put in `MACHINES`.

- [ ] **Step 1: Write the failing tests**
  - `test_status_prints_one_row_per_machine_in_config_order` (fake `gui_state`/`git_state`, capture stdout, assert row order and that an unreachable machine still prints a row)
  - `test_update_exit_code_is_one_when_any_machine_fails`
  - `test_start_without_machine_names_is_an_error` (argparse exit, nothing run)

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/test_machines.py -q`

- [ ] **Step 3: Implement** `bin/machines.py`; add the script entry; update the skill and CLAUDE.md entry-point block (`uv run pytweezer-machines status   # see / update / start / stop pytweezer on the lab PCs`).

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/ -q` — all pass.
Run: `uv sync && uv run python -m bin.machines status` on PH-BEAST — prints a row for all five machines; IC-CZC221CCHS and ph-bonesaw `ok`.

- [ ] **Step 5: Commit**

```bash
git add bin/machines.py pyproject.toml uv.lock .claude/skills/run-pytweezer/SKILL.md CLAUDE.md tests/test_machines.py
git commit -m "Add pytweezer-machines: status, update, start, stop, setup"
```

---

### Task 5: Devices tab grouped by machine

**Files:**
- Modify: `bin/managed_panel.py` (`DevicesPanel.__init__` row layout; new `MachineLine(QFrame)` and `_MachineWorker(QThread)`)
- Modify: `.claude/skills/pytweezer-gui-internals/SKILL.md` (new short section "Machine lines": what they show, SSH every 15 s off the GUI thread, Stop = `quit`, disabled when simulating / no key / own PC)
- Test: `tests/test_gui.py`

**Interfaces:**
- Consumes: `machines.machines`, `gui_state`, `start`, `stop` (Task 3); `SIMULATING` from config.
- Produces: `DevicesPanel._machine_lines: dict[str, MachineLine]`; `MachineLine.apply(state: GuiState)`; `MachineLine.button: QPushButton | None`.

Layout: for each machine with at least one device whose `host` equals its address, in `MACHINES` order: the `MachineLine`, then that machine's device rows (and sub-device rows) as today. Devices on no machine follow under a plain "Other" label. A `MachineLine` shows name, address, a state label (`server · tweez since 09:12 on 17 Nov`, `not running`, `unreachable`, `no SSH key`, `Starting…`, `Stopping…`) and a Start/Stop button. No button on this PC's own line. Button disabled with a tooltip when `SIMULATING` (`"Disabled in a simulated session"`), when `reach == "no key"` (`"Run pytweezer-machines setup <name> on this PC first"`), or for Linux / no launcher. Stop asks `QMessageBox.question(..., "Stop pytweezer on <name>? Its GUI and devices will close.")`. `_MachineWorker` polls `gui_state` for every machine line every 15 s and right after a button press, runs `start`/`stop` off the GUI thread, and reports by signal. "Starting…"/"Stopping…" holds until the polled state agrees or 60 s pass.

- [ ] **Step 1: Write the failing tests** (use the existing `devices_panel` fixture; monkeypatch `managed_panel.machines` functions so no SSH runs, and the worker so it does not start)
  - `test_devices_are_grouped_under_their_machine` (config with devices on 10.59.3.1 and 10.59.3.5 and one on 10.0.0.9: lines for PH-BEAST and ph-bonesaw only, in that order, then "Other")
  - `test_machine_line_buttons` (parametrised: own PC → no button; `SIMULATING` → disabled; `"no key"` → disabled; running owner → button reads "Stop")
  - `test_stop_asks_before_stopping` (patch `QMessageBox.question` to return No → `stop` not called; Yes → called once)

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/test_gui.py -q`

- [ ] **Step 3: Implement** in `bin/managed_panel.py`, styled per the `pytweezer-gui-design` skill (machine line visually a group header above its device rows); update the gui-internals skill.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/ -q` — all pass.
Screenshot the client Devices tab offscreen with the `run-pytweezer` skill's driver (not the real GUI on PH-BEAST: building it starts device servers) and check the grouping reads at a glance.

- [ ] **Step 5: Commit**

```bash
git add bin/managed_panel.py tests/test_gui.py .claude/skills/pytweezer-gui-internals/SKILL.md
git commit -m "Group the Devices tab by machine, with Start/Stop per machine"
```

---

### Task 6: Roll out and check on the lab

Not code: the end-to-end check the suite can't do.

- [ ] **Step 1:** Merge the branch to `main` and push (after review).
- [ ] **Step 2:** `uv run python -m bin.machines status` on PH-BEAST; expect IC-CZC221CCHS and ph-bonesaw `ok`, IC-CZC4287H3W `no key`, twalker-laptop `ok`.
- [ ] **Step 3:** `uv run python -m bin.machines update IC-CZC221CCHS twalker-laptop`; expect both `ok` and on the new commit. ph-bonesaw is on a feature branch: expect `failed`/skip with "on experiments-use-live-mode; pass --branch to switch".
- [ ] **Step 4:** With the user's go-ahead (someone may be using it): `setup IC-CZC221CCHS` to register its launch tasks, then Start and Stop it once from the Devices tab on PH-BEAST; confirm its GUI opens on its desktop and closes, and MOTMaster stays up.
- [ ] **Step 5:** Report IC-CZC4287H3W (`setup` needs its password) and anything that failed to the user.
