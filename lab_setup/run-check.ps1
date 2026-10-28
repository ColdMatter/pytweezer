# Runs one change's check.ps1 for lab-setup.ps1. Under -File a thrown error
# would also exit 1 and read as "pending", so it is caught and exits 2.
param([string]$Script, [string]$Repo)
$ProgressPreference = 'SilentlyContinue'
try { & $Script -Repo $Repo; exit $LASTEXITCODE }
catch { [Console]::Error.WriteLine($_); exit 2 }
