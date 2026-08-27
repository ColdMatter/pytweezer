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


## What this is

Control software for an atom-tweezer experiment (Rb and CaF systems). PyQt5 GUIs,
ZMQ messaging, and sipyco RPC servers coordinate device drivers (cameras,
MotMaster experiment sequencers) across the lab PCs.

## Commands

This project uses Poetry.
Code is linted and formatted with ruff. There is a pre-commit hook to run ruff on staged files, but you can also run it manually.
There is a pytest suite under `tests/` — run it after
touching code it covers, and add cases there instead of writing one-off
verification scripts when the change fits its scope.

```bash
poetry install                      # (re)generate console scripts after editing pyproject.toml
poetry run pytest tests/ -q         # run the test suite
poetry run python <script>.py       # run any script inside the venv
poetry env info                     # show the venv path (env name looks like pytweezer-<hash>-py3.13)
```

For behavior the suite doesn't cover (hardware drivers, multi-process
server/hub interaction, anything needing a real device), fall back to running
it manually through Poetry — but check `tests/` first; it may already exercise
what you're about to hand-verify.

Entry points (`[tool.poetry.scripts]` in `pyproject.toml`):

```bash
poetry run pytweezer-server         # full-control GUI, run on the server PC
poetry run pytweezer-client         # view-only GUI, run on client PCs
poetry run pytweezer-device <name>  # start one device's RPC server standalone
poetry run pytweezer-logger <name>  # start one InfluxDB logger standalone
poetry run pytweezer-kill-stale     # kill leftover processes holding ZMQ ports
```

`start_servers.bat` / `start_client.bat` / `kill_stale.bat` wrap the above.
Run kill-stale after an unclean shutdown, when launch fails with "Address
already in use".

**Headless/offscreen testing:** PyQt5 windows can't construct without a display.
Set `QT_QPA_PLATFORM=offscreen` before running any script that imports Qt widgets
in a non-interactive shell.


### Documentation

- Use **Google-style docstrings** for Sphinx autodoc
- Docstrings should describe the current behaviour only; do not keep change history in them
- Documentation auto-generates from experiment files
- Build docs locally: `nix run .#docs`
- Documentation deploys to GitLab Pages on master branch
- Use UK English spelling throughout


### Comments and self-documenting code

- **Write self-documenting code; do not over-comment.** The people reading this
  are PhD-level physicists - do not explain standard physics (a Doppler shift, a
  π pulse, a light shift). Prefer a good name over a comment: a variable
  `doppler_shift_hz` needs no comment, and an expression that is obviously a
  Doppler shift needs at most `# Doppler shift` - usually nothing.
- **Comment only what is genuinely surprising**: a non-obvious mechanism, a
  convention we have invented, a sign or edge case that bites. Everything else
  should read straight from the code.
- Over-commenting bloats diffs and hides the code, and every comment is a second
  thing to keep in sync - a stale comment is a latent bug. A one-line bug is far
  easier to spot than one buried under fifty lines of explanation.


### Logging

- Use `get_logger(name)` from `pytweezer/logging_utils.py`, never bare
  `logging.getLogger` - it adds structured JSONL output under `logs/`

### TODO/FIXME Convention

- Use `TODO` for planned improvements
- Use `FIXME` for temporary bodges that must be removed

### Git branching conventions

- New features should usually be developed on feature branches
- All branches should have an associated merge request
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
| Adding a driver / applet / analysis / InfluxDB logger | `add-device-driver`, `add-applet`, `add-analysis-script`, `add-logger` |
| Running or screenshotting the GUI | `run-pytweezer` |

Each skill is self-contained; there is no separate long-form documentation.
