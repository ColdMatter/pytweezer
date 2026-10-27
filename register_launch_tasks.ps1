# Registers on-demand scheduled tasks "pytweezer-server" and "pytweezer-client"
# that run start_servers.bat / start_client.bat in the logged-on user's desktop
# session. A process started directly from an SSH session lives in session 0
# (no visible window) and dies when the SSH session closes; triggering one of
# these tasks instead, e.g.
#     ssh <pc> schtasks /run /tn pytweezer-server
# starts the GUI on the desktop as if double-clicked.
#
# Run once per lab PC, as the user whose desktop the GUI should appear on:
#     powershell -ExecutionPolicy Bypass -File register_launch_tasks.ps1
# Re-running replaces the tasks. The tasks have no trigger, so they only ever
# run when asked to, and only while that user is logged on.

$ErrorActionPreference = "Stop"

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
# The default 72 h execution limit would kill a long-running GUI.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

foreach ($launcher in @(
        @{ Name = "pytweezer-server"; Bat = "start_servers.bat" },
        @{ Name = "pytweezer-client"; Bat = "start_client.bat" })) {
    $action = New-ScheduledTaskAction -Execute "cmd.exe" `
        -Argument "/c `"$PSScriptRoot\$($launcher.Bat)`"" -WorkingDirectory $PSScriptRoot
    Register-ScheduledTask -TaskName $launcher.Name -Action $action -Principal $principal `
        -Settings $settings -Description "Start pytweezer ($($launcher.Bat)) on the desktop; triggered over SSH." `
        -Force | Out-Null
    Write-Host "Registered $($launcher.Name) -> $PSScriptRoot\$($launcher.Bat)"
}
