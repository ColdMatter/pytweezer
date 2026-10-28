# Lab Setup Changes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Machine changes (install uv, store a share login, add a firewall rule) are written once as folders in `lab_setup/changes/` on `main`; `pytweezer-server`/`-client` show a popup at startup when one is pending on that PC, and the person there chooses to apply it.

**Architecture:** A Windows PowerShell 5.1 tool, `lab_setup/lab-setup.ps1`, fetches `origin/main`, extracts `lab_setup/` at that commit to a temp folder and re-runs itself from there; in `-Check` mode it prints JSON of the changes that apply to this PC, in apply mode it applies the pending ones. `pytweezer/lab_setup.py` runs the check at GUI startup (before anything is built), reports the result to Postgres, and shows `LabSetupDialog`; **Apply now** starts the tool in its own console and exits pytweezer.

**Tech Stack:** Windows PowerShell 5.1, git, Python 3.11, PyQt6, psycopg, Grafana JSON provisioning, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-lab-setup-changes-design.md`

## Global Constraints

- Nothing is applied without the person at the machine choosing to; pytweezer must never start, stop or change anything on the machine by itself.
- The startup check must never stop or noticeably delay pytweezer: overall limit 10 s, any failure → log and start normally.
- `lab_setup/` is always read from `origin/main`; the working copy, its branch and its index are never touched.
- The tool uses only what Windows ships plus git: Windows PowerShell 5.1, no modules, no Python, no uv.
- Secrets are never written to disk, logs, the record or the database, and never committed.
- Elevation is the same lab account (UAC consent or an already-elevated PowerShell); one UAC prompt per apply session at most, none if already elevated.
- Hostnames compare case-insensitively.
- Popup unanswered for 120 s counts as **Later**.
- Windows only in this plan; `lab-setup.sh` (macOS/Linux) is a later plan. Manifests list `systems` so it can slot in.
- Use `get_logger` from `pytweezer/logging_utils.py`; Google-style docstrings; UK English; comment only the surprising. Load the `pytweezer-gui-design` skill before writing the dialog and `write-tests` before writing tests.

## Review Focus

- **GitHub unreachable / no network at startup:** pytweezer starts normally within ~10 s, using the last fetched `origin/main` if there is one. Tested in Task 2 (`test_check_uses_existing_origin_main_when_fetch_fails`) and Task 4 (`test_check_returns_none_on_timeout`).
- **Working copy on another branch with local edits to `lab_setup/`:** the tool still uses `origin/main` and `git status` is unchanged afterwards. Task 2 (`test_check_reads_origin_main_not_the_working_copy`).
- **A check script that throws:** that change reports `error`, the rest are still checked, the popup ignores it. Task 2 (`test_a_throwing_check_is_reported_as_error`).
- **An apply that fails mid-way:** the tool stops, records the failure, leaves later changes alone. Task 3 (`test_apply_stops_at_the_first_failure`).
- **Hostname case** (`PH-BEAST` vs `ph-beast`, `ph-bonesaw` vs `PH-BONESAW`): filters still match. Task 2 (`test_host_filters_ignore_case`).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `lab_setup/lab-setup.ps1` | The tool: fetch/extract `origin/main`, applicability, `-Check` JSON, apply loop, elevation, record, relaunch |
| `lab_setup/changes/<YYYY-MM-DD-slug>/change.json` | Manifest of one change |
| `lab_setup/changes/<...>/check.ps1`, `apply.ps1` | Check (exit 0 done / 1 pending) and apply for Windows |
| `lab_setup/changes/_template/` | Folder to copy for a new change (skipped by the tool: leading `_`) |
| `pytweezer/lab_setup.py` | Python side: run the check, read the record, report to the DB, launch apply |
| `pytweezer/GUI/lab_setup_dialog.py` | `LabSetupDialog` |
| `bin/gui.py` | Startup hook in `_run` |
| `pytweezer/database/schema.sql`, `writer.py` | `machine_changes` table and `DBWriter.record_machine_change` |
| `pytweezer/configuration/config.py` | `DATABASE["server_dsn"]` |
| `deploy/grafana/dashboards/pytweezer-health.json` | "Lab setup" table panel |
| `.claude/skills/add-lab-change/SKILL.md`, `CLAUDE.md` | How to write a change; skill table row |
| `register_launch_tasks.ps1` | Deleted; becomes a change |

### Manifest (`change.json`)

```json
{
  "title": "Store the login for the pytweezer data share",
  "description": "Lets the Results tab read measurement files from PH-BEAST.",
  "systems": ["windows"],
  "hosts": {"except": ["PH-BEAST"]},
  "admin": false,
  "secrets": [
    {"name": "READER_PASSWORD", "prompt": "Password for PH-BEAST\\pytweezer-reader", "ask": "Tom"}
  ]
}
```

`hosts` is optional and holds exactly one of `only` / `except`. `secrets` is optional.

### Script contract

- Both scripts take `param([string]$Repo, [hashtable]$Secrets)`; `$Repo` is the pytweezer working copy (scripts run from the temp copy, so never use `$PSScriptRoot` for repo paths). `check.ps1` gets no secrets.
- `check.ps1`: exit 0 = in place, 1 = pending; anything else or a thrown error = `error`. No admin, no side effects.
- `apply.ps1`: throw on failure. Must be safe to re-run.

### `-Check` output (stdout, one JSON object, nothing else on stdout)

```json
{"commit": "<origin/main sha>", "host": "PH-BEAST",
 "changes": [{"id": "2026-10-28-install-uv", "title": "...", "description": "...",
              "admin": false, "state": "done", "error": null}]}
```

`state` is `done`, `pending` or `error`. Only changes applying to this host and system are listed, sorted by id. Exit 0; non-zero (and no JSON) only when there is no `origin/main` to read.

### Record (`<Repo>/logs/lab_setup.jsonl`, one line per apply attempt)

```json
{"time": "2026-10-28T14:02:11+00:00", "host": "PH-BEAST", "change": "2026-10-28-install-uv",
 "commit": "<sha>", "result": "done", "error": null}
```

### Tool parameters

`lab-setup.ps1 [-Repo <path>] [-Check] [-Yes] [-Relaunch server|client] [-HostName <name>] [-NoFetch] [-Ref <ref>] [-FromMain <commit>]`

- `-Repo` defaults to the parent of the script's folder.
- `-Ref` defaults to `origin/main`; set to e.g. `origin/<branch>` to try a change from a pushed branch before merging. The fetch fetches that branch.
- `-HostName` overrides `$env:COMPUTERNAME` (tests).
- `-NoFetch` skips `git fetch` (tests, offline use).
- `-FromMain` is internal: set when the extracted copy is re-invoked, so it does not fetch/extract again.
- `-Yes` skips the confirmation (the popup has already asked).

---

### Task 1: `machine_changes` table, writer method, server DSN, Grafana panel

**Files:**
- Modify: `pytweezer/database/schema.sql` (append table)
- Modify: `pytweezer/database/writer.py` (`MACHINE_CHANGE_COLUMNS`, `SQL`, `record_machine_change`)
- Modify: `pytweezer/configuration/config.py:33-38` (`DATABASE["server_dsn"]`)
- Modify: `deploy/grafana/dashboards/pytweezer-health.json` (panel)
- Modify: `tests/helpers.py` (`RecordingDB.record_machine_change`)
- Test: `tests/test_db_writer.py`, `tests/test_grafana.py`

**Interfaces:**
- Produces: `DBWriter.record_machine_change(host: str, change: str, state: str, checked_at: datetime, applied_at: datetime | None, commit: str, error: str | None) -> None`; `config.DATABASE["server_dsn"]: str`; `RecordingDB.machine_changes: list[tuple]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db_writer.py
def test_record_machine_change_upserts_on_host_and_change(db, writer):
    checked = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
    writer.record_machine_change(
        "PH-BEAST", "2026-10-28-install-uv", "pending", checked, None, "abc123", None
    )
    assert writer.flush(2)
    (row,) = db.sent["machine_changes"]
    assert row == (
        "PH-BEAST",
        "2026-10-28-install-uv",
        "pending",
        checked,
        None,
        "abc123",
        None,
    )
    assert "ON CONFLICT (host, change)" in writer_module.SQL["machine_changes"]


# tests/test_grafana.py
def test_health_dashboard_shows_lab_setup_per_machine():
    health = _dashboards()["pytweezer-health.json"]
    (panel,) = [p for p in health["panels"] if p.get("title") == "Lab setup"]
    sql = panel["targets"][0]["rawSql"]
    assert "FROM machine_changes" in sql
```

Also add a test that `DATABASE["server_dsn"]` uses `HOSTS[SERVER_PC]` even when simulating, and `PYTWEEZER_DB_DSN` when set (reload `config` under `monkeypatch.setenv`, as `tests/test_simulation_mode.py` does).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_db_writer.py tests/test_grafana.py -q`
Expected: FAIL (`AttributeError: record_machine_change`, no "Lab setup" panel).

- [ ] **Step 3: Implement**

- `schema.sql`: `CREATE TABLE IF NOT EXISTS machine_changes (host text NOT NULL, change text NOT NULL, state text NOT NULL, checked_at timestamptz NOT NULL, applied_at timestamptz, commit text, error text, PRIMARY KEY (host, change));`
- `writer.py`: `MACHINE_CHANGE_COLUMNS = ("host", "change", "state", "checked_at", "applied_at", "commit", "error")`, `SQL["machine_changes"] = _upsert("machine_changes", MACHINE_CHANGE_COLUMNS, ("host", "change"))`, method queues the row as given.
- `config.py`: `"server_dsn": os.environ.get("PYTWEEZER_DB_DSN", f"postgresql://pytweezer:pytweezer@{HOSTS[SERVER_PC]}:5432/pytweezer")`, with a one-line comment that it ignores simulation because machine reports always go to PH-BEAST.
- Grafana: a full-width table panel titled "Lab setup" on the provisioned `pytweezer` datasource: `SELECT change, host, state, checked_at AS "checked", applied_at AS "applied", error FROM machine_changes ORDER BY change DESC, host`; colour `state` (done green, pending amber, error/failed red) the way the "Loggers" panel colours its state column.
- `RecordingDB.record_machine_change` appends the args tuple to `self.machine_changes`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_db_writer.py tests/test_grafana.py tests/test_simulation_mode.py -q`
Expected: PASS. If `PYTWEEZER_TEST_DSN` is set, also `uv run pytest tests/test_db_integration.py -q` (add `machine_changes` to its `TRUNCATE`).

- [ ] **Step 5: Commit**

```bash
git add pytweezer/database tests/test_db_writer.py tests/test_grafana.py tests/test_db_integration.py tests/helpers.py pytweezer/configuration/config.py deploy/grafana
git commit -m "Add machine_changes table and Lab setup Grafana panel"
```

---

### Task 2: `lab-setup.ps1` check mode

**Files:**
- Create: `lab_setup/lab-setup.ps1`
- Create: `lab_setup/changes/_template/{change.json,check.ps1,apply.ps1}`
- Test: `tests/test_lab_setup_tool.py`

**Interfaces:**
- Produces: the `-Check` JSON, parameters and script contract above.

Behaviour:
1. Unless `-FromMain`: `git -C $Repo fetch origin <branch of -Ref>` as a background job, waited for 5 s, then stopped if still running (unless `-NoFetch`). Resolve `-Ref`; if missing, write an error to stderr and exit 2. Extract `lab_setup/` at that commit with `git archive --format=zip` into `$env:TEMP\pytweezer-lab-setup\<sha>` (reuse if present), then `& <extracted>\lab_setup\lab-setup.ps1 @PSBoundParameters -FromMain <sha> -Repo $Repo` and exit with its exit code.
2. With `-FromMain`: enumerate `changes\*` (skip names starting `_`), sorted; keep those whose `systems` includes `windows` and whose `hosts` filter admits `-HostName` (case-insensitive).
3. Run each `check.ps1` in a child `powershell -NoProfile -ExecutionPolicy Bypass -File` (so a throw or `exit` cannot kill the tool); map exit code 0/1/other.
4. `-Check`: print the JSON with `ConvertTo-Json -Depth 4 -Compress`; all diagnostics go to stderr.

Test fixture: a bare `origin` repo and a clone in `tmp_path`; the clone's `main` holds `lab_setup/lab-setup.ps1` copied from the repo under test plus fake changes. A fake change's check tests for a marker file `<Repo>/markers/<id>`, its apply creates it. Run the tool with `subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ...], capture_output=True, timeout=60)`. Skip the module unless `sys.platform == "win32"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_check_lists_pending_and_done(lab_repo):
    lab_repo.add_change("2026-01-01-a")
    lab_repo.add_change("2026-01-02-b")
    lab_repo.mark_done("2026-01-01-a")
    out = lab_repo.check()
    assert [(c["id"], c["state"]) for c in out["changes"]] == [
        ("2026-01-01-a", "done"),
        ("2026-01-02-b", "pending"),
    ]
    assert out["commit"] == lab_repo.origin_main_sha()


def test_host_filters_ignore_case(lab_repo):
    lab_repo.add_change("only-beast", hosts={"only": ["PH-BEAST"]})
    lab_repo.add_change("not-beast", hosts={"except": ["ph-beast"]})
    ids = [c["id"] for c in lab_repo.check(host="ph-beast")["changes"]]
    assert ids == ["only-beast"]


def test_template_and_other_systems_are_skipped(lab_repo):
    lab_repo.add_change("_template")
    lab_repo.add_change("mac-only", systems=["macos"])
    assert lab_repo.check()["changes"] == []


def test_check_reads_origin_main_not_the_working_copy(lab_repo):
    lab_repo.add_change("on-main")
    lab_repo.git("switch", "-c", "elsewhere")
    lab_repo.add_change_uncommitted("only-local")
    status = lab_repo.git("status", "--porcelain")
    assert [c["id"] for c in lab_repo.check()["changes"]] == ["on-main"]
    assert lab_repo.git("status", "--porcelain") == status
    assert lab_repo.git("branch", "--show-current") == "elsewhere"


def test_check_sees_changes_pushed_after_the_clone(lab_repo):
    lab_repo.push_change_from_elsewhere("new")
    assert "new" in [c["id"] for c in lab_repo.check(fetch=True)["changes"]]


def test_check_uses_existing_origin_main_when_fetch_fails(lab_repo):
    lab_repo.add_change("a")
    lab_repo.break_origin_url()
    assert [c["id"] for c in lab_repo.check(fetch=True)["changes"]] == ["a"]


def test_a_throwing_check_is_reported_as_error(lab_repo):
    lab_repo.add_change("bad", check='throw "boom"')
    lab_repo.add_change("good")
    states = {c["id"]: c["state"] for c in lab_repo.check()["changes"]}
    assert states == {"bad": "error", "good": "pending"}
```

`lab_repo.check(host="PH-BEAST", fetch=False)` runs the tool with `-Check -HostName host` (`-NoFetch` unless `fetch`) and returns parsed stdout; `add_change` commits and pushes to `origin/main`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_lab_setup_tool.py -q`
Expected: FAIL (tool missing).

- [ ] **Step 3: Implement `lab-setup.ps1` check mode and `_template`**

`_template/change.json` holds every manifest key with placeholder text; its `check.ps1` exits 1, its `apply.ps1` throws `"not implemented"`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_lab_setup_tool.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lab_setup tests/test_lab_setup_tool.py
git commit -m "Add lab-setup.ps1 check mode reading lab_setup/ from origin/main"
```

---

### Task 3: `lab-setup.ps1` apply mode

**Files:**
- Modify: `lab_setup/lab-setup.ps1`
- Test: `tests/test_lab_setup_tool.py`

**Interfaces:**
- Consumes: Task 2's fetch/extract and applicability.
- Produces: the record format above; `-Yes`, `-Relaunch server|client`.

Behaviour (apply is the default without `-Check`):
1. Run the checks; list the pending changes (title, description, "needs admin"). None → say so and go to step 6.
2. Unless `-Yes`, ask `Apply these N changes? [y/N]`.
3. If any pending change has `admin` and the process is not elevated: `Start-Process powershell -Verb RunAs -Wait` on the extracted tool with `-FromMain <sha> -Repo <Repo> -Yes -HostName <host>` (no `-Relaunch`; the commit is already resolved, so `-Ref` is not needed), then go to step 6. The elevated window does steps 4–5 for every pending change, in order. Already elevated → no prompt.
4. For each pending change in order: prompt for each secret with `Read-Host -AsSecureString` (showing `prompt` and "ask <ask>"), convert to plain text in memory only, call `& apply.ps1 -Repo $Repo -Secrets $secrets` in-process inside `try`, then re-run its check. Done only if apply did not throw **and** the check now exits 0.
5. Append a record line per attempt (`done` or `failed` with the exception message; never secret values). On the first failure, stop and list the remaining changes as not attempted.
6. Print a summary. With `-Relaunch`, wait for Enter ("Press Enter to start pytweezer-server"), then `Start-Process cmd.exe -ArgumentList "/c start_servers.bat"` (or `start_client.bat`) `-WorkingDirectory $Repo`, non-elevated: the relaunch always happens in the original, non-elevated window.

- [ ] **Step 1: Write the failing tests**

```python
def test_apply_applies_pending_in_order_and_records(lab_repo):
    lab_repo.add_change("2026-01-01-a")
    lab_repo.add_change("2026-01-02-b")
    lab_repo.apply(yes=True)
    assert lab_repo.marker_order() == ["2026-01-01-a", "2026-01-02-b"]
    records = lab_repo.records()
    assert [(r["change"], r["result"]) for r in records] == [
        ("2026-01-01-a", "done"),
        ("2026-01-02-b", "done"),
    ]
    assert {c["state"] for c in lab_repo.check()["changes"]} == {"done"}


def test_apply_stops_at_the_first_failure(lab_repo):
    lab_repo.add_change("2026-01-01-a", apply='throw "disk full"')
    lab_repo.add_change("2026-01-02-b")
    lab_repo.apply(yes=True)
    (record,) = lab_repo.records()
    assert record["result"] == "failed" and "disk full" in record["error"]
    assert lab_repo.marker_order() == []


def test_apply_that_does_not_satisfy_its_check_is_failed(lab_repo):
    lab_repo.add_change("a", apply="Write-Host 'did nothing'")
    lab_repo.apply(yes=True)
    assert lab_repo.records()[0]["result"] == "failed"


def test_apply_skips_changes_already_done(lab_repo):
    lab_repo.add_change("a")
    lab_repo.mark_done("a")
    lab_repo.apply(yes=True)
    assert lab_repo.records() == []
```

`lab_repo.apply(yes=True)` runs the tool with `-Yes -NoFetch -HostName PH-BEAST` and stdin closed. Elevation, secrets prompting and relaunch need a person at the screen: they are checked by hand in Task 7.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_lab_setup_tool.py -q`
Expected: the four new tests FAIL.

- [ ] **Step 3: Implement apply mode**

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_lab_setup_tool.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lab_setup/lab-setup.ps1 tests/test_lab_setup_tool.py
git commit -m "Add lab-setup.ps1 apply mode with elevation, record and relaunch"
```

---

### Task 4: Python side, `pytweezer/lab_setup.py`

**Files:**
- Create: `pytweezer/lab_setup.py`
- Test: `tests/test_lab_setup.py`

**Interfaces:**
- Consumes: Task 2's `-Check` JSON; Task 3's record and parameters; Task 1's `record_machine_change`, `DATABASE["server_dsn"]`.
- Produces:
  - `SKIP_ENV = "PYTWEEZER_SKIP_LAB_SETUP"`
  - `@dataclass(frozen=True) class ChangeStatus: id: str; title: str; description: str; admin: bool; state: str; error: str | None`
  - `@dataclass(frozen=True) class CheckResult: commit: str; host: str; changes: list[ChangeStatus]` with property `pending -> list[ChangeStatus]` (state `pending` only)
  - `check(repo: Path = REPO, timeout: float = 10.0) -> CheckResult | None`
  - `last_attempts(repo: Path = REPO) -> dict[str, dict]` (change id → its latest record line)
  - `report(result: CheckResult, writer, repo: Path = REPO) -> None`
  - `launch_apply(role: str, repo: Path = REPO) -> None` (`role` is `"server"` or `"client"`)

`REPO` is the repo root (from `pytweezer/configuration/paths.py`'s `tweezerpath`).

- `check`: returns `None` (and logs why) when `SKIP_ENV` is set, `sys.platform != "win32"`, the tool file is missing, `powershell` fails, exits non-zero, times out or prints invalid JSON. Runs `powershell -NoProfile -ExecutionPolicy Bypass -File <repo>/lab_setup/lab-setup.ps1 -Check -Repo <repo>` with `CREATE_NO_WINDOW`.
- `report`: one `record_machine_change` per change; state becomes `failed` when the check says `pending` and the latest attempt failed; `applied_at` is the latest `done` attempt's time; `error` is the check's error or the latest failed attempt's error. Never waits on the database.
- `launch_apply`: `subprocess.Popen` of the tool with `-Yes -Relaunch <role> -Repo <repo>`, `creationflags=CREATE_NEW_CONSOLE`.

- [ ] **Step 1: Write the failing tests**

```python
def test_check_parses_the_tool_output(fake_tool):
    fake_tool.prints(
        {
            "commit": "abc",
            "host": "PH-BEAST",
            "changes": [
                {
                    "id": "a",
                    "title": "A",
                    "description": "d",
                    "admin": True,
                    "state": "pending",
                    "error": None,
                },
                {
                    "id": "b",
                    "title": "B",
                    "description": "d",
                    "admin": False,
                    "state": "done",
                    "error": None,
                },
            ],
        }
    )
    result = lab_setup.check(fake_tool.repo)
    assert [c.id for c in result.pending] == ["a"] and result.changes[0].admin


@pytest.mark.parametrize(
    "failure", ["timeout", "exit_code", "bad_json", "missing_tool"]
)
def test_check_returns_none_on_failure(fake_tool, failure):
    fake_tool.fails(failure)
    assert lab_setup.check(fake_tool.repo, timeout=1) is None


def test_check_is_skipped_by_the_environment(fake_tool, monkeypatch):
    monkeypatch.setenv(lab_setup.SKIP_ENV, "1")
    assert lab_setup.check(fake_tool.repo) is None
    assert fake_tool.calls == 0


def test_report_marks_failed_attempts_and_last_applied(tmp_path, recording_db):
    write_records(
        tmp_path,
        [
            {
                "time": "2026-10-28T09:00:00+00:00",
                "change": "a",
                "result": "done",
                "error": None,
            },
            {
                "time": "2026-10-28T10:00:00+00:00",
                "change": "b",
                "result": "failed",
                "error": "boom",
            },
        ],
    )
    lab_setup.report(
        result_with(("a", "done"), ("b", "pending"), ("c", "error")),
        recording_db,
        tmp_path,
    )
    rows = {r[1]: r for r in recording_db.machine_changes}
    assert rows["a"][2] == "done" and rows["a"][4] == datetime(
        2026, 10, 28, 9, tzinfo=UTC
    )
    assert rows["b"][2] == "failed" and rows["b"][6] == "boom"
    assert rows["c"][2] == "error"
```

`fake_tool` monkeypatches `subprocess.run` in `pytweezer.lab_setup` (and creates `lab_setup/lab-setup.ps1` under `tmp_path` unless the case is `missing_tool`). Add `os.environ.setdefault("PYTWEEZER_SKIP_LAB_SETUP", "1")` to `tests/conftest.py` next to the Qt line, and have these tests `monkeypatch.delenv` it.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_lab_setup.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `pytweezer/lab_setup.py`**

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_lab_setup.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/lab_setup.py tests/test_lab_setup.py tests/conftest.py
git commit -m "Run the lab setup check from Python and report it to the database"
```

---

### Task 5: Startup dialog and GUI hook

**Files:**
- Create: `pytweezer/GUI/lab_setup_dialog.py`
- Modify: `bin/gui.py` (`_run`, `server_main`, `client_main`)
- Test: `tests/test_lab_setup.py`

**Interfaces:**
- Consumes: Task 4's `check`, `report`, `launch_apply`, `CheckResult.pending`, `ChangeStatus`.
- Produces: `LabSetupDialog(changes: list[ChangeStatus], timeout_s: float = 120, parent=None)` (a `QDialog`; `exec()` returns `Accepted` for **Apply now**); `offer_lab_setup(role: str, *, timeout_s: float = 120) -> bool` in `pytweezer/lab_setup.py`, True when pytweezer should exit because apply was launched.

Load `pytweezer-gui-design` first. The dialog: heading "N changes waiting for this PC", one entry per change (title, description, a "needs admin" tag), a countdown line ("Starting pytweezer as normal in 1:58"), buttons **Apply now** (default; tooltip: pytweezer closes, the changes are applied in a separate window, then pytweezer restarts) and **Later**. The countdown reaching zero rejects.

`offer_lab_setup`: `check()`; `None` → False. Otherwise `report()` with a module-level `DBWriter(DATABASE["server_dsn"])` that is never closed or flushed at startup (its daemon thread sends during the session). No pending → False. Pending → dialog; Accepted → `launch_apply(role)` and True. Any exception inside → log and False.

`bin/gui.py`: `_run(build, role)`; after the `QApplication` and stylesheet and **before** `build()`: `if offer_lab_setup(role): logging.shutdown(); os._exit(0)`. `server_main` passes `"server"`, `client_main` `"client"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_dialog_lists_changes_and_marks_admin(qapp):
    dialog = LabSetupDialog(
        [status("a", "Install uv", admin=False), status("b", "Share", admin=True)]
    )
    text = all_text(dialog)
    assert "Install uv" in text and "Share" in text and "needs admin" in text


def test_dialog_times_out_to_later(qapp):
    dialog = LabSetupDialog([status("a", "A")], timeout_s=0.2)
    assert dialog.exec() == QDialog.DialogCode.Rejected


def test_offer_launches_apply_and_asks_to_exit_when_accepted(monkeypatch, qapp):
    launched = stub_offer(
        monkeypatch, pending=["a"], answer=QDialog.DialogCode.Accepted
    )
    assert lab_setup.offer_lab_setup("client") is True
    assert launched == ["client"]


@pytest.mark.parametrize(
    "case", ["nothing_pending", "later", "check_failed", "dialog_raises"]
)
def test_offer_starts_pytweezer_normally(monkeypatch, qapp, case):
    launched = stub_offer(monkeypatch, case=case)
    assert lab_setup.offer_lab_setup("server") is False
    assert launched == []
```

`stub_offer` monkeypatches `check`, `report`, `launch_apply` and `LabSetupDialog.exec` in `pytweezer.lab_setup`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_lab_setup.py -q`
Expected: new tests FAIL.

- [ ] **Step 3: Implement the dialog, `offer_lab_setup` and the `_run` hook**

- [ ] **Step 4: Run the tests and the GUI suite**

Run: `uv run pytest tests/test_lab_setup.py tests/test_gui.py -q`
Expected: PASS.

- [ ] **Step 5: Screenshot the dialog alone** (not the whole GUI: building it on PH-BEAST starts device servers)

Run a scratch script that builds `LabSetupDialog` with the three real changes from Task 6's titles and saves `dialog.grab()` to a PNG with `QT_QPA_PLATFORM=offscreen`; look at it against `pytweezer-gui-design`.

- [ ] **Step 6: Commit**

```bash
git add pytweezer/GUI/lab_setup_dialog.py pytweezer/lab_setup.py bin/gui.py tests/test_lab_setup.py
git commit -m "Offer pending lab setup changes when pytweezer starts"
```

---

### Task 6: The first changes, the skill and the template's use

**Files:**
- Create: `lab_setup/changes/2026-10-28-install-uv/`, `2026-10-28-db-dsn/`, `2026-10-28-launch-tasks/`, `2026-10-28-pytweezer-data-share/`, `2026-10-28-data-share-login/` (each `change.json`, `check.ps1`, `apply.ps1`)
- Delete: `register_launch_tasks.ps1` (update every reference: `grep -rn register_launch_tasks`)
- Create: `.claude/skills/add-lab-change/SKILL.md`
- Modify: `CLAUDE.md` (skill table row)
- Test: `tests/test_lab_setup_changes.py`

**Interfaces:**
- Consumes: the manifest and script contract.

| Change | hosts | admin | secrets | check (exit 0 when) | apply |
| --- | --- | --- | --- | --- | --- |
| `install-uv` | all | no | – | `uv` is on PATH or `%USERPROFILE%\.local\bin\uv.exe` exists | `winget install --id astral-sh.uv -e --silent --accept-source-agreements --accept-package-agreements` |
| `db-dsn` | all | no | `DB_DSN` ("PYTWEEZER_DB_DSN connection string", ask Tom) | user env `PYTWEEZER_DB_DSN` is non-empty | `[Environment]::SetEnvironmentVariable("PYTWEEZER_DB_DSN", ..., "User")` |
| `launch-tasks` | all | no | – | tasks `pytweezer-server` and `pytweezer-client` exist and their action's working directory is `$Repo` | `register_launch_tasks.ps1`'s body, with `$PSScriptRoot` replaced by `$Repo` |
| `pytweezer-data-share` | only `PH-BEAST` | yes | `READER_PASSWORD` (ask Tom) | local user `pytweezer-reader`, share `pytweezer-data` and firewall rule `pytweezer data share (SMB-In)` all exist | the `setup-data-share.ps1` logic (account, `icacls`, share, firewall rule), with the data path `<Repo>\data` and the password from `$Secrets` |
| `data-share-login` | except `PH-BEAST` | no | `READER_PASSWORD` (ask Tom) | `cmdkey /list:10.59.3.1` shows `pytweezer-reader` | `cmdkey /add:10.59.3.1 /user:PH-BEAST\pytweezer-reader /pass:<secret>` |

The one-off `setup-data-share.ps1` (2026-10-28) is reproduced here so it can be ported:

```powershell
$DataDir = "C:\Users\tweez\ColdMatter\pytweezer\data"   # becomes Join-Path $Repo data
$User = "pytweezer-reader"; $Share = "pytweezer-data"
if (-not (Get-LocalUser -Name $User -ErrorAction SilentlyContinue)) {
    New-LocalUser -Name $User -Password $pw -PasswordNeverExpires -UserMayNotChangePassword `
        -Description "Read-only access to the pytweezer data share" | Out-Null
}
Remove-LocalGroupMember -Group "Users" -Member $User -ErrorAction SilentlyContinue
icacls $DataDir /grant "${env:COMPUTERNAME}\${User}:(OI)(CI)RX" | Out-Null
if (-not (Get-SmbShare -Name $Share -ErrorAction SilentlyContinue)) {
    New-SmbShare -Name $Share -Path $DataDir -ReadAccess "${env:COMPUTERNAME}\$User" `
        -Description "pytweezer measurement files (read-only)" | Out-Null
}
$Rule = "pytweezer data share (SMB-In)"
if (-not (Get-NetFirewallRule -DisplayName $Rule -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $Rule -Direction Inbound -Protocol TCP -LocalPort 445 `
        -Profile Private -RemoteAddress 10.59.0.0/16, 100.64.0.0/10 -Action Allow | Out-Null
}
```

The skill says: what a change is and when to write one; copy `_template`; the manifest keys; the script contract (`$Repo` not `$PSScriptRoot`, check has no side effects and needs no admin, apply throws on failure and is re-runnable); secrets come from `$Secrets` and are never echoed; a change on `main` is not edited except to fix it; before merging, push the branch and try the change on one PC with `lab_setup\lab-setup.ps1 -Ref origin/<branch>`; a new PC needs git and a clone of pytweezer, then one hand-run of `lab_setup\lab-setup.ps1`, after which the startup popup takes over. CLAUDE.md row: `| Rolling out a setup change to the lab PCs (install a tool, a share login, a setting) | add-lab-change |`.

- [ ] **Step 1: Write the failing test**

```python
CHANGES = Path(__file__).parents[1] / "lab_setup" / "changes"


@pytest.mark.parametrize(
    "folder", sorted(p for p in CHANGES.iterdir() if p.is_dir()), ids=lambda p: p.name
)
def test_every_change_is_well_formed(folder):
    manifest = json.loads((folder / "change.json").read_text())
    assert manifest["title"] and manifest["description"]
    assert set(manifest["systems"]) <= {"windows", "macos", "linux"}
    assert len(manifest.get("hosts", {})) <= 1 and set(manifest.get("hosts", {})) <= {
        "only",
        "except",
    }
    assert isinstance(manifest["admin"], bool)
    for secret in manifest.get("secrets", []):
        assert set(secret) == {"name", "prompt", "ask"}
    if "windows" in manifest["systems"]:
        assert (folder / "check.ps1").is_file() and (folder / "apply.ps1").is_file()
    if not folder.name.startswith("_"):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+", folder.name)


def test_changes_do_not_use_psscriptroot_for_repo_paths():
    for script in CHANGES.glob("*/*.ps1"):
        assert "$PSScriptRoot" not in script.read_text(), script


def test_the_first_changes_exist():
    names = {p.name for p in CHANGES.iterdir()}
    assert {
        "2026-10-28-install-uv",
        "2026-10-28-db-dsn",
        "2026-10-28-launch-tasks",
        "2026-10-28-pytweezer-data-share",
        "2026-10-28-data-share-login",
    } <= names
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_lab_setup_changes.py -q`
Expected: `test_the_first_changes_exist` FAILS.

- [ ] **Step 3: Write the five changes, delete `register_launch_tasks.ps1`, write the skill and the CLAUDE.md row**

- [ ] **Step 4: Run the tests and a real check on this PC**

Run: `uv run pytest tests/test_lab_setup_changes.py tests/test_lab_setup_tool.py -q`
Expected: PASS.
Push the branch, then on PH-BEAST run: `powershell -ExecutionPolicy Bypass -File lab_setup\lab-setup.ps1 -Check -Ref origin/lab-setup`
Expected: JSON listing `install-uv` (`done`), `db-dsn`, `launch-tasks`, and `pytweezer-data-share` (`done` if the share was set up by hand on 2026-10-28, else `pending`); no `data-share-login` (PH-BEAST is excluded); no `error` states.

- [ ] **Step 5: Commit**

```bash
git add lab_setup/changes .claude/skills/add-lab-change CLAUDE.md tests/test_lab_setup_changes.py
git rm register_launch_tasks.ps1
git commit -m "Add the first lab setup changes and the add-lab-change skill"
```

---

### Task 7: Whole-flow check by hand and merge request

**Files:** none new.

- [ ] **Step 1: Full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass (the lab setup tool tests run on Windows).

- [ ] **Step 2: Open the merge request** for `lab-setup` against `main`, listing what was checked by hand below as unchecked boxes.

- [ ] **Step 3: After merge, on PH-BEAST (person at the screen):** run `powershell -ExecutionPolicy Bypass -File lab_setup\lab-setup.ps1`. Expected: it fetches, lists the pending changes for PH-BEAST (`pytweezer-data-share` only if not already done, `db-dsn`, `launch-tasks` as applicable), asks to confirm, shows one UAC prompt if an admin change is pending, asks for secrets with "ask Tom", applies, and `logs/lab_setup.jsonl` has one line per attempt with no secret in it.

- [ ] **Step 4: On one client PC:** start `start_client.bat`. Expected: the popup lists `data-share-login` (and anything else pending) before any window or process starts; **Later** starts pytweezer normally; next launch **Apply now** closes it, applies in a console, and "Press Enter" restarts pytweezer-client with no popup. The Grafana "Lab setup" panel shows both machines.

- [ ] **Step 5: Unplug the network on that client and start it again.** Expected: pytweezer starts within ~10 s, no popup, no error dialog.
