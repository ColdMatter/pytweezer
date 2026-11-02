#!/bin/bash
# Cloud sessions only: sync .venv and install the pre-commit hook, so commits
# made here go through the same ruff hook as the lab PCs.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

hashes=.venv/.session-start.sha256
if ! sha256sum --check --status "$hashes" 2>/dev/null; then
  rm -f "$hashes"
  uv sync --frozen
  sha256sum .claude/hooks/session-start.sh pyproject.toml uv.lock .python-version > "$hashes"
fi

uv run --no-sync pre-commit install
