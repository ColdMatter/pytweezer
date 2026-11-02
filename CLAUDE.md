# CLAUDE.md

## Maintaining this file: prefer skills

This file is loaded into context on **every** agent run, so keep it lean -
only conventions that apply to most tasks belong here. Anything situational
(setup recipes, niche workflows, deep reference material) belongs in a Claude
Code skill under `.claude/skills/`, which agents load on demand.

If a user asks you to "remember" something - even if they explicitly say to
add it to this file - and it would not be used in every run, suggest making a
skill instead and explain why: lean always-on context plus progressive
disclosure through skills gets noticeably better performance out of agents.

## Keeping skills current

Whenever a change alters something a skill under `.claude/skills/` describes,
bring the skill up to date as part of that change:

- **Minor** (a renamed function, file or argument; a corrected signature or
  path; a line or table row for something new within the skill's existing
  scope): make the edit yourself, in the same commit as the code.
- **Significant** (a new section; changing or removing a rule or recommended
  approach; restructuring or splitting a skill; changing its `description`):
  tell the user what you would change and why, and wait for their go-ahead.
- **Not covered by any skill** (a new subsystem, workflow or set of
  conventions that future agents will need): propose a new skill to the user,
  naming its scope and trigger, and create it only once they agree.


## What this is

Control software for an atom-tweezer experiment (Rb and CaF systems). PyQt6 GUIs,
ZMQ messaging, and sipyco RPC servers coordinate device drivers (cameras,
MotMaster experiment sequencers) across the lab PCs.

## important branches

exp-handler is the branch developing a full device server and experiment handler, based on artiq, balic, and sipyco. it is in active development. docs are descriptive, not proscriptive. 

rearrangement-optimisation-RnD is a stripped-down branch for optimisation of tweezer control with an SLM. 

## Commands

This project uses uv; the venv is `.venv` in the repo root, on the Python in
`.python-version`.
Code is linted and formatted with ruff, configured in `pyproject.toml`.
There is a pytest suite under `tests/` — run it after
touching code it covers, and add cases there instead of writing one-off
verification scripts when the change fits its scope.
`.github/workflows/tests.yml` runs the suite (against a scratch TimescaleDB) on
every push to main and every PR.

```bash
uv sync                         # install deps, rebuild the C++ extensions and console scripts
uv run pytest tests/ -q         # run the test suite
uv run python <script>.py       # run any script inside the venv
```

**Tests follow the design, never the other way round.** Design for the lab as
it runs; when a design breaks an existing test, change or delete the test.
Do not add parameters, hooks, globals or fallbacks to production code just so
a test can reach it, or keep a worse design because the suite assumes it.

For behavior the suite doesn't cover (hardware drivers, multi-process
server/hub interaction, anything needing a real device), fall back to running
it manually through uv — but check `tests/` first; it may already exercise
what you're about to hand-verify.

Entry points (`[project.scripts]` in `pyproject.toml`):

```bash
uv run pytweezer-server         # full-control GUI; simulates on any PC but the server PC
uv run pytweezer-client         # client GUI, run on client PCs
uv run pytweezer-device <name>  # start one device's RPC server standalone
uv run pytweezer-logger <name>  # start one database logger standalone
uv run pytweezer-experiment-manager <name>  # start the experiment queue manager standalone
uv run pytweezer-kill-stale     # kill leftover processes holding ZMQ ports
uv run pytweezer-db-backfill    # load existing measurement files into the database
```

`start_servers.bat` / `start_client.bat` / `kill_stale.bat` run the same entry
points as `uv run python -m bin.launch [client]` / `bin.kill_stale`, because
Windows Smart App Control blocks uv's unsigned `.exe` launchers.
Run kill-stale after an unclean shutdown, when launch fails with "Address
already in use".

**Headless/offscreen testing:** PyQt6 windows can't construct without a display.
Set `QT_QPA_PLATFORM=offscreen` before running any script that imports Qt widgets
in a non-interactive shell. In cloud sessions `.claude/hooks/session-start.sh`
apt-installs the system libraries PyQt6 needs; if `libEGL.so.1` is still
missing, it says why at session start.


### Ruff and the pre-commit hook

`.pre-commit-config.yaml` runs `ruff check --fix` then `ruff format` on the
staged Python files of every commit, using the ruff in `.venv` (the version
`uv.lock` pins, as VS Code does). Markdown is not formatted. Install the hook
once per clone with `uv run pre-commit install`; cloud sessions do it in
`.claude/hooks/session-start.sh`. CI runs `ruff check .` and
`ruff format --check .` on every push and PR, catching anything that skipped
the hook.

To commit:

1. Run `uv run ruff check --fix <files>` and `uv run ruff format <files>` on
   the Python files you changed.
2. `git add` those files whole. Never commit a file that also has unstaged
   edits: pre-commit stashes them while it runs, and if ruff then changes the
   file it prints "Stashed changes conflicted with hook auto-fixes... Rolling
   back fixes", throws its fixes away and aborts - and the same commit fails
   the same way every time.
3. `git commit`. "files were modified by this hook" means the commit did not
   happen: `git add` the files it names and commit once more. A second failure
   is something else - read the output rather than retrying.
4. Never pass `--no-verify`.

The hook does not check files that a clean merge brings in, so after merging
run `uv run ruff check .` and `uv run ruff format --check .` before pushing.


### Documentation

- Sphinx docs live in `docs/`. Build locally: `cd docs && uv run make html`,
  then open `docs/_build/html/index.html`. Install the toolchain first with
  `uv sync --group docs`.
- The API reference is auto-generated from docstrings by `sphinx-apidoc` at
  build time and is not checked in (`docs/api/`, `docs/_build/` are gitignored).
- Use **Google-style docstrings** for new code (napoleon is enabled); existing
  reStructuredText-markup docstrings also render fine.
- Docstrings should describe the current behaviour only; do not keep change history in them
- Use UK English spelling throughout
- `.github/workflows/docs.yml` builds the docs on every push/PR and deploys
  `main` to GitHub Pages (`https://coldmatter.github.io/pytweezer/`). A build
  failure fails CI, so keep the docs building.


### Lab book

"The labbook" is the shared Obsidian vault github.com/ColdMatter/CaFMOT-LabBook,
cloned at `~/CaFMOT-LabBook`; read its own `CLAUDE.md` before writing to it.


### Comments and self-documenting code

- **Write self-documenting code; do not over-comment.** The people reading this
  are PhD-level physicists - do not explain standard physics. Prefer a good name over a comment: a variable
  `doppler_shift_hz = ` is better than `d = # Doppler shift in Hz`.
- **Comment only what is genuinely surprising**: a non-obvious mechanism, a
  convention we have invented, a sign or edge case that bites. Everything else
  should read straight from the code.
- Over-commenting bloats diffs and hides the code, and every comment is a second
  thing to keep in sync - a stale comment is a latent bug. A one-line bug is far
  easier to spot than one buried under fifty lines of explanation.


### Logging

- Use `get_logger(name)` from `pytweezer/logging_utils.py`, never bare
  `logging.getLogger` - it adds structured JSONL output under `logs/`

### Notebooks

A notebook is a record of what was done at a particular time: never edit one
retroactively, even when an API change breaks it. The exception is a notebook
that demonstrates a feature; keep that one working. When a change breaks other
notebooks, list them for the user instead.

### TODO/FIXME Convention

- Use `TODO` for planned improvements
- Use `FIXME` for temporary bodges that must be removed

### Git branching conventions

- Small, tested fixes may be committed and pushed straight to main, no branch or merge request needed
- Features, and anything that needs review, go on a feature branch
- Every feature branch should have an associated merge request
- Name every branch for what it changes (e.g. `fix-copier-share-chmod`), including branches only Claude works on - never a session ID, date or `wip`/`tmp`
- A line `Depends on owner/repo#N` (or `#N`) in a PR description holds it until that PR merges or that issue closes
- Commit and push as you go: after each coherent, tested step (e.g. one tab restyled), not in one batch at the end. Stage only your own files, leaving unrelated changes (e.g. `.vscode/settings.json`) alone
- `.vscode/settings.json` is shared by everyone who opens the repo. If a commit would include changes to it that the user didn't ask for (often a personal preference saved under the Settings UI's Workspace tab), warn the user and ask before committing them
- **The main branch should always be deployable** - no broken code or failing tests allowed


## Architecture in brief

One **server PC** runs the shared long-lived processes (ZMQ hubs, stream
loggers, Analysis Manager, Device Status); **arbitrarily many client PCs** each
run the devices plugged into them plus their own applets and GUI.

`pytweezer/configuration/config.py` says what runs
where — `CONFIG["Servers"]`/`["Devices"]`/`["Loggers"]`/`["GUI"]`, read through
`get_config()`. An entry's `host` decides which PC that process runs on,
independent of where the GUI was launched. Filesystem-layout constants live in
`pytweezer/configuration/paths.py`; the **root** `configuration/` directory is
unrelated JSON data.

Every managed process is launched the same way — `python <script> <name>`, where
`name` is the CONFIG key that the script looks itself up under.

Load a skill when you need more than that:

| Working on | Skill |
| --- | --- |
| Config, hosts, launching, pub/sub fabric, general orientation | `pytweezer-architecture` |
| Device server/client, `get_device()`, composites, coordinators | `pytweezer-device-framework` |
| GUI shell, tabs, panels, process tiles, teardown | `pytweezer-gui-internals` |
| How a panel should look, read and behave (layout, styling, wording) | `pytweezer-gui-design` |
| Adding a driver / applet / analysis / database logger | `add-device-driver`, `add-applet`, `add-analysis-script`, `add-logger` |
| Writing/queuing an `Experiment`, `submit`/`run_local`, the queue manager | `add-experiment` |
| Writing or accepting an optimiser (`StagedScan`, `BayesianOptimisation`, `Fit`), the Optimisers tab | `add-optimiser` |
| Running shots, scans or auto-tune from a lab-book notebook (`Rig`), MOTMaster zips | `notebook-rig` |
| Running or screenshotting the GUI | `run-pytweezer` |
| Writing, reviewing or deleting tests | `write-tests` |
| MOTMaster channels (which board/port/line a signal is on), scripts, the C# side | Read `../EDMSuite/.claude/skills/motmaster-controller/SKILL.md` — EDMSuite skills don't load from here. Channel maps: `../EDMSuite/DAQ/MoleculeMOTHardware.cs` (CaF), `../EDMSuite/DAQ/RbTweezerHardware.cs` (Rb) |

Each skill is self-contained; there is no separate long-form documentation.
