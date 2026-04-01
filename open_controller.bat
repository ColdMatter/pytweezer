@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m bin.controller
) else (
    poetry run pytweezer-controller
)

endlocal
