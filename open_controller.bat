@echo off
setlocal
cd /d "%~dp0"

poetry run pytweezer-controller dashboard %1


endlocal
