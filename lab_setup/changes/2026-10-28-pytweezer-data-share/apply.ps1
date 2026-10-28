param([string]$Repo, [hashtable]$Secrets)
$ErrorActionPreference = "Stop"
$DataDir = Join-Path $Repo "data"
$User = "pytweezer-reader"; $Share = "pytweezer-data"
$pw = ConvertTo-SecureString $Secrets["READER_PASSWORD"] -AsPlainText -Force
if (-not (Get-LocalUser -Name $User -ErrorAction SilentlyContinue)) {
    New-LocalUser -Name $User -Password $pw -PasswordNeverExpires -UserMayNotChangePassword `
        -Description "Read-only access to the pytweezer data share" | Out-Null
}
Remove-LocalGroupMember -Group "Users" -Member $User -ErrorAction SilentlyContinue
icacls $DataDir /grant "${env:COMPUTERNAME}\${User}:(OI)(CI)RX" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "icacls failed with exit code $LASTEXITCODE" }
if (-not (Get-SmbShare -Name $Share -ErrorAction SilentlyContinue)) {
    New-SmbShare -Name $Share -Path $DataDir -ReadAccess "${env:COMPUTERNAME}\$User" `
        -Description "pytweezer measurement files (read-only)" | Out-Null
}
$Rule = "pytweezer data share (SMB-In)"
if (-not (Get-NetFirewallRule -DisplayName $Rule -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $Rule -Direction Inbound -Protocol TCP -LocalPort 445 `
        -Profile Private -RemoteAddress 10.59.0.0/16, 100.64.0.0/10 -Action Allow | Out-Null
}
