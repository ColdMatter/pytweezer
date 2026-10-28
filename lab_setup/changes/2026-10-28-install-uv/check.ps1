param([string]$Repo, [hashtable]$Secrets)
if ((Get-Command uv -ErrorAction SilentlyContinue) -or (Test-Path "$env:USERPROFILE\.local\bin\uv.exe")) { exit 0 }
exit 1
