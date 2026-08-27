# Getting started

## Install

The project uses [Poetry](https://python-poetry.org/).

```bash
poetry install                 # runtime + dev dependencies, console scripts
poetry install --with docs     # additionally install the documentation toolchain
```

## Run the test suite

```bash
poetry run pytest tests/ -q
```

For behaviour the suite does not cover (hardware drivers, multi-process
server/hub interaction, anything needing a real device), run it manually
through Poetry — but check `tests/` first.

## Console entry points

Defined under `[tool.poetry.scripts]` in `pyproject.toml`:

| Command | Purpose |
| --- | --- |
| `poetry run pytweezer-server` | full-control GUI, run on the server PC |
| `poetry run pytweezer-client` | view-only GUI, run on client PCs |
| `poetry run pytweezer-device <name>` | start one device's RPC server standalone |
| `poetry run pytweezer-logger <name>` | start one InfluxDB logger standalone |
| `poetry run pytweezer-kill-stale` | kill leftover processes holding ZMQ ports |

`start_servers.bat` / `start_client.bat` / `kill_stale.bat` wrap these on the
lab machines. Run kill-stale after an unclean shutdown, when a launch fails
with "Address already in use".

Qt windows cannot construct without a display. Set
`QT_QPA_PLATFORM=offscreen` before running any script that imports Qt widgets
in a non-interactive shell.

## Build this documentation

```bash
cd docs
poetry run make html
```

Open `docs/_build/html/index.html`. The API reference under `docs/api/` is
regenerated from the package docstrings by `sphinx-apidoc` on every build and
is not checked in; `poetry run make clean` removes it along with `_build/`.
