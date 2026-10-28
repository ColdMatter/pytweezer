param([string]$Repo, [hashtable]$Secrets)
if (-not (Get-LocalUser -Name "pytweezer-reader" -ErrorAction SilentlyContinue)) { exit 1 }
if (-not (Get-SmbShare -Name "pytweezer-data" -ErrorAction SilentlyContinue)) { exit 1 }
if (-not (Get-NetFirewallRule -DisplayName "pytweezer data share (SMB-In)" -ErrorAction SilentlyContinue)) { exit 1 }
exit 0
