param([string]$Repo, [hashtable]$Secrets)
# winget updates the registry PATH, not this process's, so re-read it.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
if (Get-Command uv -ErrorAction SilentlyContinue) { exit 0 }
if (Test-Path "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe") { exit 0 }
if (Test-Path "$env:USERPROFILE\.local\bin\uv.exe") { exit 0 }
exit 1
