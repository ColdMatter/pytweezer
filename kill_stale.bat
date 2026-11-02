@echo off
setlocal
cd /d "%~dp0"

uv run python -m bin.kill_stale
pause

endlocal
