---
name: add-lab-change
description: Write a lab setup change - a small folder under lab_setup/changes/ that rolls a machine-level setting out to the lab PCs (install a tool, set an environment variable, register a scheduled task, store a share login, open a firewall rule) and shows in the startup popup until applied. Use whenever the user wants something done on every lab PC (or one named PC), says a new PC needs setting up, asks to "roll out", "install on all the PCs", "add a setup step", or asks how lab_setup/lab-setup.ps1 or the pending-changes popup works - even if they only describe the setting and don't say "change".
---

# Adding a lab setup change

A **change** is one machine-level step the lab PCs need, written once as a
folder on `main`. `lab_setup/lab-setup.ps1` reads the changes from `origin/main`,
runs each `check.ps1` to see which are pending on this PC and applies those.
pytweezer shows a startup popup while any are pending. Write a change for
anything that lives on the machine rather than in the repo (tools, user
environment variables, scheduled tasks, accounts, shares, firewall rules,
stored credentials). Things that live in the repo need a normal commit.

## Writing one

Copy `lab_setup/changes/_template/` to `lab_setup/changes/YYYY-MM-DD-short-name/`
(today's date, lower-case words and digits joined by hyphens; ids sort in
apply order). Keep every file ASCII-only: Windows PowerShell 5.1 reads BOM-less
files as ANSI.

### `change.json`

| Key | Meaning |
| --- | --- |
| `title`, `description` | Shown in the popup. Say what it does and why. |
| `systems` | Any of `windows`, `macos`, `linux`. Windows needs `check.ps1` and `apply.ps1`. |
| `hosts` | Optional; exactly one of `{"only": [...]}` or `{"except": [...]}` of PC names (e.g. `PH-BEAST`; compared case-insensitively). Omit for every PC. |
| `admin` | `true` if apply needs elevation. If any pending change needs admin, *all* pending changes, admin or not, run in order in the one elevated window. |
| `secrets` | Optional list of `{"name", "prompt", "ask"}`: a value the user is asked to type (`prompt`), and who to ask for it (`ask`). |

### Scripts

Both take `param([string]$Repo, [hashtable]$Secrets)`.

- Use `$Repo` (the pytweezer working copy) for every repo path, **never
  `$PSScriptRoot`**: the scripts run from a temporary copy.
- `check.ps1` runs in a separate PowerShell with only `-Repo`. Exit 0 = in
  place, 1 = pending; a thrown error or any other exit code is reported as
  `error`. It must have no side effects and need no admin. A change whose
  check errors is left alone; later changes are still checked and applied.
  `$Repo` may be any clone or worktree, so don't make a check depend on which
  one pytweezer was started from.
- `apply.ps1` is safe to run again and **throws on failure**. A change counts
  as done only if apply did not throw and `check.ps1` then exits 0.
- Secrets arrive as `$Secrets["NAME"]` (plain strings). Never echo, log,
  transform or write them anywhere: recorded errors mask a secret only if it
  appears verbatim. **Never pass a secret on a native command line**
  (`cmdkey /pass:`, `net user ... <password>`): process auditing records
  command lines, and PowerShell 5.1 mangles quotes in them. Use an in-process
  API instead (a cmdlet taking a `SecureString`, or P/Invoke through
  `Add-Type`, as `2026-10-28-data-share-login` does with `CredWrite`).
- pytweezer exits before applying, but a change that touches the venv or
  anything else pytweezer holds open must first check that no pytweezer
  process is still running, and throw if one is.

A change on `main` is not edited once it has been rolled out, except to fix
it; for a new behaviour, add a new change.

## Changing the tool itself

Every branch's pytweezer runs `origin/main`'s `lab-setup.ps1` (through its
own copy as a bootstrap) and parses the `-Check` JSON with its own Python. So
the `-Check` JSON fields (`commit`, `host`, `changes[]` with `id`, `title`,
`description`, `admin`, `state`, `error`) and the bootstrap parameters
(`-Check`, `-Yes`, `-Relaunch`, `-Repo`, `-Ref`, `-NoFetch`) may only be
added to, never renamed or removed.

## Trying it before merging

Push your branch, then on one PC run

```powershell
powershell -ExecutionPolicy Bypass -File lab_setup\lab-setup.ps1 -Check -Ref origin/<branch>
```

to see each change's state as JSON (`done`, `pending` or `error`), then drop
`-Check` to apply. `-NoFetch` skips the fetch. Run `check.ps1` directly
freely; don't run an `apply.ps1` by hand on a PC you don't want changed.
`uv run pytest tests/test_lab_setup_changes.py` checks the manifests are
well formed.

## A new PC

It needs git and a clone of pytweezer. From the clone, run

```powershell
powershell -ExecutionPolicy Bypass -File lab_setup\lab-setup.ps1
```

once by hand (the default execution policy blocks a bare
`lab_setup\lab-setup.ps1`); after that the startup popup takes over.
