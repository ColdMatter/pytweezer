@echo off
setlocal
cd /d "%~dp0"

uv run pytweezer-kill-stale
pause

endlocal
