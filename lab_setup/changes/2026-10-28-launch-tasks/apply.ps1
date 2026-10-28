param([string]$Repo, [hashtable]$Secrets)
$ErrorActionPreference = "Stop"

# Not $env:USERDOMAIN: an SSH session doesn't set it to the account's domain.
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive
# The default 72 h execution limit would kill a long-running GUI.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

foreach ($launcher in @(
        @{ Name = "pytweezer-server"; Bat = "start_servers.bat" },
        @{ Name = "pytweezer-client"; Bat = "start_client.bat" })) {
    $action = New-ScheduledTaskAction -Execute "cmd.exe" `
        -Argument "/c `"$Repo\$($launcher.Bat)`"" -WorkingDirectory $Repo
    Register-ScheduledTask -TaskName $launcher.Name -Action $action -Principal $principal `
        -Settings $settings -Description "Start pytweezer ($($launcher.Bat)) on the desktop; triggered over SSH." `
        -Force | Out-Null
}
