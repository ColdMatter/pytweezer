@echo off
setlocal
cd /d "%~dp0"

poetry run pytweezer-run dashboard


endlocal
