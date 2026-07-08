@echo off
setlocal
cd /d "%~dp0"

poetry run pytweezer-kill-stale
pause

endlocal
