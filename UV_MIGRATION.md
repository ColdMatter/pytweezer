# uv migration: handoff notes

Working notes for the `uv-migration` branch. Delete this file before merging.

## Done (commit b6bc2f0)

- `pyproject.toml` uses the standard `[project]` table; `uv.lock` replaces
  `poetry.lock`. Version constraints are unchanged (carets became `>=x,<y`).
- The build backend is setuptools. `build_tools/build.py` became `setup.py`
  (moved with `git mv`); the extensions and `OptionalBuildExt` are unchanged.
  uv's own backend can't compile extensions, and the automatic converter
  (`migrate-to-uv`) silently drops the C++ build, so don't re-run it.
- `[tool.uv] cache-keys` covers `cpp_implementations/**`, so `uv sync` rebuilds
  the extensions when the C++ changes.
- `[tool.pytest.ini_options] pythonpath = ["."]`: Poetry's editable install put
  the repo root on `sys.path`; setuptools only exposes `pytweezer` and `bin`, so
  `tests.*` stopped importing without it.

Checked on the laptop (Linux, Python 3.13):

- A wheel built with setuptools contains the same 166 files as the old
  poetry-core wheel: data files, icons, SQL, both `.so` and both committed `.pyd`.
- `uv sync` compiles both extensions and they import from `pytweezer/cpp/`.
- `uv run pytest tests/ -q`: 332 passed, 5 skipped.

The laptop's Poetry env was in-project, so `uv sync` took over that `.venv`.
`git switch main && poetry install --sync` puts it back.

## Next: test on the beast

The beast's live Poetry env is in `AppData\Local\pypoetry\Cache`, so a separate
worktree with its own `.venv` won't touch it. uv is already installed there.
In PowerShell:

```powershell
cd <existing pytweezer checkout>
git fetch origin uv-migration
git worktree add ..\pytweezer-uv uv-migration
cd ..\pytweezer-uv
uv sync --python 3.11
git status --short pytweezer/cpp
uv run python -c "from pytweezer.cpp import morph_tophat_cpp, sum_pixel_values_cpp; print('ok')"
uv run pytest tests/ -q
```

What to look for:

- `git status` should show both `.pyd` files as `M`, meaning MSVC rebuilt them.
  The committed cp311 `.pyd` files import even if nothing was built, so a
  successful import alone proves nothing. If they're unmodified, run
  `uv sync --reinstall-package pytweezer -v` and look for `*** WARNING: skipping`.
- Don't launch the servers or GUI from the worktree while the live ones are
  running.

Clean up afterwards with `git worktree remove ..\pytweezer-uv`.

## Still to do on this branch

Swap Poetry commands for uv (`poetry run` → `uv run`, `poetry install` →
`uv sync`, `--with docs` → `--group docs`) in:

- `start_servers.bat`, `start_client.bat`, `kill_stale.bat`
- `.github/workflows/docs.yml`: use `astral-sh/setup-uv` instead of
  pipx/poetry and setup-python, then `uv sync --group docs` and
  `uv run make -C docs html`
- `CLAUDE.md` (Commands section, docs build lines)
- skills: `add-analysis-script`, `add-applet`, `add-device-driver`,
  `add-logger`, `pytweezer-gui-design`, `run-pytweezer` (SKILL.md and the
  scripts' usage docstrings)
- `docs/getting-started.md`, `docs/Makefile`, `docs/notes/database.md`,
  `docs/notes/gui_architecture.md`, `docs/notes/device_framework.md`
- docstrings: `pytweezer/cpp/__init__.py`, `pytweezer/database/backfill.py`,
  `tools/bench_hamcam_rpc.py`, `tools/bench_sipyco_readline.py`
- `.vscode/settings.json`: Poetry is set as the env manager

`git grep -n -i poetry -- ':!*.ipynb' ':!docs/superpowers/plans'` lists them.
Leave the old plans in `docs/superpowers/plans` as they are.

## Decisions before merging

- Pin Python with a `.python-version`? The beast runs 3.11 (that's what the
  committed `.pyd` files are built for); the laptop resolved to 3.13.
- Every lab PC moves at once. The two lock files won't stay in sync, so don't
  run Poetry and uv side by side. Each PC needs uv installed.
- Re-point Jupyter/VS Code kernels at the new `.venv`.

## Optional follow-ups (separate changes)

- `transport.py` and `rearrangement.py` announce the numpy fallback with a bare
  `print`; a `get_logger(...).warning` would make a beast without the C++ show
  up in the logs.
- Move `pytest` and `ruff` into a dev dependency group; drop
  `zmq>=0.0.0,<0.0.1`, an empty placeholder package (`pyzmq` is the real one).
