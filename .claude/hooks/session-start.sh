#!/bin/bash
# Cloud sessions only: sync .venv and install the pre-commit hook, so commits
# made here go through the same ruff hook as the lab PCs.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# The container image lacks the system libraries PyQt6 loads (first symptom:
# "ImportError: libEGL.so.1"), which every GUI test needs.
qt_libs="libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3"
if ! ldconfig -p | grep -q libEGL.so.1; then
  if ! { apt-get install -y $qt_libs || { apt-get update && apt-get install -y $qt_libs; }; } >/dev/null 2>&1; then
    echo "session-start: could not apt-get install $qt_libs; PyQt6 will not import" >&2
  fi
fi

hashes=.venv/.session-start.sha256
if ! sha256sum --check --status "$hashes" 2>/dev/null; then
  rm -f "$hashes"
  uv sync --frozen
  sha256sum .claude/hooks/session-start.sh pyproject.toml uv.lock .python-version > "$hashes"
fi

uv run --no-sync pre-commit install
