# Getting started

## Install

The project uses [uv](https://docs.astral.sh/uv/); the environment is `.venv` in
the repo root, on the Python version in `.python-version`.

```bash
uv sync                 # runtime + dev dependencies, console scripts
uv sync --group docs    # additionally install the documentation toolchain
```

## Run the test suite

```bash
uv run pytest tests/ -q
```

For behaviour the suite does not cover (hardware drivers, multi-process
server/hub interaction, anything needing a real device), run it manually
through uv — but check `tests/` first.

## Console entry points

Defined under `[project.scripts]` in `pyproject.toml`:

| Command | Purpose |
| --- | --- |
| `uv run pytweezer-server` | full-control GUI, run on the server PC |
| `uv run pytweezer-client` | view-only GUI, run on client PCs |
| `uv run pytweezer-device <name>` | start one device's RPC server standalone |
| `uv run pytweezer-logger <name>` | start one database logger standalone |
| `uv run pytweezer-db-backfill` | load existing measurement files into the database |
| `uv run pytweezer-kill-stale` | kill leftover processes holding ZMQ ports |

`start_servers.bat` / `start_client.bat` / `kill_stale.bat` wrap these on the
lab machines. Run kill-stale after an unclean shutdown, when a launch fails
with "Address already in use".

Qt windows cannot construct without a display. Set
`QT_QPA_PLATFORM=offscreen` before running any script that imports Qt widgets
in a non-interactive shell.

## Build this documentation

```bash
cd docs
uv run make html
```

Open `docs/_build/html/index.html`. The API reference under `docs/api/` is
regenerated from the package docstrings by `sphinx-apidoc` on every build and
is not checked in; `uv run make clean` removes it along with `_build/`.
