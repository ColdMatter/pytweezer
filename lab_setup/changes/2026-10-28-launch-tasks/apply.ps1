param([string]$Repo, [hashtable]$Secrets)
$ErrorActionPreference = "Stop"

# The main worktree, not $Repo: the tasks should not follow whichever worktree happened to run this.
$commonDir = git -C $Repo rev-parse --path-format=absolute --git-common-dir
if ($LASTEXITCODE -ne 0 -or -not $commonDir) { throw "git could not find the repository of $Repo" }
$mainWorktree = Split-Path -Parent ([IO.Path]::GetFullPath($commonDir))

# Not $env:USERDOMAIN: an SSH session doesn't set it to the account's domain.
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive
# The default 72 h execution limit would kill a long-running GUI.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

foreach ($launcher in @(
        @{ Name = "pytweezer-server"; Bat = "start_servers.bat" },
        @{ Name = "pytweezer-client"; Bat = "start_client.bat" })) {
    $bat = Join-Path $mainWorktree $launcher.Bat
    if (-not (Test-Path -PathType Leaf $bat)) { throw "$bat is missing" }
    $action = New-ScheduledTaskAction -Execute "cmd.exe" `
        -Argument "/c `"$bat`"" -WorkingDirectory $mainWorktree
    Register-ScheduledTask -TaskName $launcher.Name -Action $action -Principal $principal `
        -Settings $settings -Description "Start pytweezer ($($launcher.Bat)) on the desktop; triggered over SSH." `
        -Force | Out-Null
}
