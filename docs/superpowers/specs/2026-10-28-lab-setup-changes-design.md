# Lab setup changes

## Goal

Rolling out a machine change to the lab PCs (install uv, store the login for a
network share, add a firewall rule, register a scheduled task) currently means
pasting terminal commands into each PC by hand. Machines get missed, nobody
knows which PC has which change, and each change needs the right admin login
typed on the right PC.

Instead, a change is written once, as a folder in the pytweezer repo. When
`pytweezer-server` or `pytweezer-client` starts on a machine that hasn't got
it, a popup lists what is pending; the person at the machine chooses to apply
it now or later. One overview shows which machine has which change.

## Decisions

These were made by the lab (Tom):

- Built into pytweezer. No extra repo to clone on every machine.
- Nothing is ever applied automatically. The person at the machine chooses
  when, on lab PCs and personal laptops alike.
- They find out through a popup when `pytweezer-server` or `pytweezer-client`
  starts.
- Only machines that run pytweezer matter. Machines that never do are out of
  scope (anyone can still run the tool by hand).
- Every machine has a single regular account, so every change is a machine
  change; there is no per-user handling.
- Lab members have restricted admin rights and the admin login differs per
  machine. Admin steps are therefore elevated at apply time, on the machine,
  by whoever is there. No admin credentials are stored anywhere.
- Elevation is the same lab account, elevated (UAC consent, or the tool run
  from an elevated PowerShell), not a separate admin account. So an admin
  change's per-user effects (stored credentials, scheduled task principal,
  files under `%USERPROFILE%`) land in the lab account, and any change may be
  marked `admin`. When the tool is already elevated it shows no UAC prompt.
- Every lab PC is on `main` or being moved to it. A machine on another branch
  gets the popup once its branch includes `main`; until then the tool can be
  run by hand.

Out of scope:

- Updating pytweezer itself. Machines sit on different branches
  (exp-handler, rearrangement-optimisation-RnD) and that stays a per-person
  decision.
- Undoing a change. A mistake is fixed by a new change.
- Pushing changes to machines from one place.

Anti-goals (a failure even if it works):

- Anything that changes a machine without the person there agreeing, or while
  pytweezer is running. On Windows `uv sync` cannot replace the compiled
  extensions while pytweezer processes hold them, and a half-applied change
  mid-experiment is worse than none.
- pytweezer failing to start, or starting slowly, because GitHub, the
  database or the check is unavailable.
- Secrets in the repo.

## 1. A change

One folder per change under `lab_setup/changes/`, named
`YYYY-MM-DD-<slug>` so that sorting by name is the order to apply them in
(install uv before anything that runs `uv`). A folder holds:

- A manifest:
  - `title`: one line shown in the popup ("Store the login for the pytweezer
    data share").
  - `description`: a few lines on what it does and why.
  - `systems`: any of `windows`, `macos`, `linux`.
  - `hosts`: optional; either a list of hostnames it applies to, or a list it
    excludes. Hostnames compare case-insensitively against
    `socket.gethostname()`. For example, the share setup is PH-BEAST only; the
    share login is every machine except PH-BEAST.
  - `admin`: whether `apply` needs elevation.
  - `secrets`: optional list of values the person is asked for at apply time,
    each with a prompt and who to ask ("password for PH-BEAST\pytweezer-reader,
    ask Tom").
- A `check` script per system it supports: exits 0 if the change is already in
  place on this machine, 1 if it is pending. Never needs admin, never changes
  anything, finishes in well under a second.
- An `apply` script per system: makes the change; must be safe to run again.

A change is pending on a machine when it applies to that machine and its check
says it is not in place. The check is the truth, not a record of having run
it: a fresh machine sees every change as pending, and a change someone undoes
by hand (uninstalls gh) shows up as pending again.

Once a change is on `main` its folder is not edited, except to fix a check or
apply that does not work on some machine.

## 2. Reading changes from `origin/main`

The startup check and the apply tool both read `lab_setup/` from
`origin/main`, never from the working copy, so a machine on a branch or with
uncommitted edits still sees every change as soon as it is on `main`:

- `git fetch origin main`, with a timeout of a few seconds.
- Extract `lab_setup/` at `origin/main` into a temporary directory keyed by the
  commit (e.g. with `git archive`) and run everything from there.
- The working copy, its branch and its index are never touched.

If the fetch fails or times out, use the `origin/main` already present; if
there is none, skip the check.

## 3. The apply tool

A plain script with no dependencies beyond what the system ships (it must work
when uv or the venv is the thing being fixed):

- `lab_setup/lab-setup.ps1` for Windows (Windows PowerShell 5.1).
- `lab_setup/lab-setup.sh` for macOS and Linux.

Two modes:

- **check**: prints the pending changes for this machine in a machine-readable
  form. Used by pytweezer at startup.
- **apply** (the default when run by hand): fetches as in section 2, lists the
  pending changes with their titles, asks for confirmation, asks for any
  secrets, runs each apply in order, re-runs its check, and reports each change
  as done or failed with the error. On a failure it stops, because later
  changes may depend on earlier ones.

Admin changes on Windows run in one elevated child process, so the person sees
one UAC prompt per apply session rather than one per change. Secrets are never
written to disk, logs or the record. Environment variables do not survive
`Start-Process -Verb RunAs`, so an admin change's secrets are asked for inside
the elevated window, and a non-admin change's in the tool's own window; either
way they reach the apply script in memory only.

Every result is appended to a local record, one line per change attempt:
time, host, change, `origin/main` commit, `done`/`failed`, error excerpt.

A new machine runs the tool once by hand after cloning pytweezer (git is the
only prerequisite); after that the popup takes over.

## 4. The startup popup

In `bin/gui.py`, before the GUI is built and before any process is started:

1. Run the tool in check mode, with an overall time limit of about 10 s.
2. If nothing is pending, or anything goes wrong (no git, no network, timeout,
   a check script erroring), log it and start normally. The popup must never
   stop pytweezer from starting.
3. If changes are pending, show a dialog listing each one's title and
   description, marking those that need admin, with **Apply now** and
   **Later**.
   - **Apply now**: start the tool in apply mode in its own console window,
     told to relaunch the same GUI (server or client) when it finishes, then
     exit pytweezer.
   - **Later**: start normally. The popup appears again next launch.
   - Unanswered for about two minutes: treated as **Later**. pytweezer is
     sometimes launched remotely (`ssh <pc> schtasks /run /tn pytweezer-server`,
     see `register_launch_tasks.ps1`) with nobody at the desk, and must not sit
     waiting. (My call.)

This runs for both server and client, in simulation too: the changes are
about the machine, not the session. An environment variable turns the check
off, for the test suite and CI.

## 5. Overview

After the startup check, pytweezer writes the result for each change that
applies to this machine into a new Postgres table on PH-BEAST (alongside
`runs`, `points` and `measurements`, created by `ensure_schema`): host, change,
state (`done`, `pending`, `failed`), when it was checked, when last applied,
and the last error. If the database cannot be reached this is skipped quietly.

The report always goes to the database on PH-BEAST, also in a simulated
session, where the configured DSN otherwise points at localhost. PH-BEAST's
`pg_hba.conf` and firewall admit only 10.59.3.0/24, so a laptop off-site
simply does not report until it is next on CCMnet; that is accepted. Writing
needs `PYTWEEZER_DB_DSN` set on the machine (`docs/notes/database.md`), so
storing it is one of the first changes. (My calls.)

A Grafana panel provisioned from `deploy/grafana/` shows machines against
changes, so "which PCs still need the share login?" is one look.

## 6. Writing a change

- A template folder to copy, and a skill (`add-lab-change`) saying how to write
  a check and an apply, and how to test them on one machine before merging.
- First changes, which also exercise the design:
  - the pytweezer data share on PH-BEAST (from the one-off
    `setup-data-share.ps1` used on 2026-10-28): admin, PH-BEAST only, one
    secret (the `pytweezer-reader` password). Its data path comes from the
    repo's location and CONFIG, not a hardcoded `C:\Users\tweez\...`;
  - the stored login for the share: every machine except PH-BEAST, one secret;
  - `PYTWEEZER_DB_DSN`: every machine, one secret;
  - install uv;
  - register the on-demand launch tasks, replacing the instructions in
    `register_launch_tasks.ps1`.

## Left to the builder

- Manifest format, provided both Windows PowerShell 5.1 and the bash tool can
  read it with what ships with the system.
- Where the local record lives (somewhere per machine and outside version
  control, e.g. the gitignored `logs/`).
- How the tool relaunches pytweezer and how the console window is opened.
- Table and column names, and the Grafana panel's layout.
- Tests: the applicability rules, reading from `origin/main` (against a
  temporary git repo), the popup's skip-on-failure behaviour, and the tool's
  check/apply loop on a fake change, as far as `tests/` can reach them.
- Order of work. My call: Windows first, since every lab PC runs Windows; the
  bash tool after.
