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

## Checked on the beast (Windows, Python 3.11.4)

Done in a detached worktree (`git worktree add --detach ..\pytweezer-uv
uv-migration`), because the live checkout already has `uv-migration` checked
out and the running servers have its `.pyd` files loaded.

- The first `uv sync` failed: uv had locked `pyqt5-qt5==5.15.19` (pulled in
  by pylablib and qscintilla via PyQt5), which has no Windows wheels. Poetry
  had 5.15.2. `[tool.uv] required-environments` now lists win/AMD64 and
  linux/x86_64, so the lock forks: 5.15.2 on Windows, 5.15.19 on Linux
  (commit ffefdd5).
- `uv sync --python 3.11` rebuilt both `.pyd` files with MSVC (`git status`
  showed `M`); they import from the worktree.
- `uv run pytest tests/ -q`: 331 passed, 5 skipped, 1 failed. The failure,
  `test_script_parameters_leaves_the_selected_script_unchanged`, expects `/`
  path separators. It fails the same way under the Poetry env, so it predates
  this branch.
- `VIRTUAL_ENV` points at the Poetry env in the beast's shells; uv warns and
  ignores it, but it'll need unsetting once Poetry is gone.

`main` has moved on (#27, LoadTweezers), so merge it in before the PR.

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

- Python is pinned to 3.11 by `.python-version` (what the beast runs and the
  committed `.pyd` files are built for). The laptop's `.venv` is 3.13, so
  `uv sync` there will rebuild it on 3.11.
- The beast's live checkout now has a working uv `.venv` (built and tested
  after the servers were stopped; same 1 known failure).
- PyQt5 is gone. It came only from `qscintilla` (unused, dropped) and full
  `pylablib`, now `pylablib-lightweight`, whose newest release is 1.4.3 (was
  1.4.5); the diff between them is Linux portability and FT232 fixes, nothing
  our drivers use. `numba` (TLCamera) and `pyusb` (picomotor) are now direct
  dependencies. On the beast, both builds behave the same: the ImagEM snaps a
  frame; neither configured ThorCam (32148, 38570) is visible to the Thorlabs
  SDK; the picomotor lookup fails with pyusb's "No backend available".
- Every lab PC moves at once. The two lock files won't stay in sync, so don't
  run Poetry and uv side by side. Each PC needs uv installed.
- Re-point Jupyter/VS Code kernels at the new `.venv`.

## Optional follow-ups (separate changes)

- `transport.py` and `rearrangement.py` announce the numpy fallback with a bare
  `print`; a `get_logger(...).warning` would make a beast without the C++ show
  up in the logs.
- Move `pytest` and `ruff` into a dev dependency group; drop
  `zmq>=0.0.0,<0.0.1`, an empty placeholder package (`pyzmq` is the real one).
