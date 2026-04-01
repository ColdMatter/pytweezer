@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m bin.processmanager
) else (
    poetry run pytweezer-dashboard
)

endlocal
