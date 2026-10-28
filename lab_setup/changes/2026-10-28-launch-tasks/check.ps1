param([string]$Repo, [hashtable]$Secrets)
# Any clone or worktree will do, so starting pytweezer from a second one does not undo this.
foreach ($launcher in @(
        @{ Name = "pytweezer-server"; Bat = "start_servers.bat" },
        @{ Name = "pytweezer-client"; Bat = "start_client.bat" })) {
    $task = Get-ScheduledTask -TaskName $launcher.Name -ErrorAction SilentlyContinue
    if (-not $task) { exit 1 }
    $workingDirectory = [string]$task.Actions[0].WorkingDirectory
    if (-not $workingDirectory) { exit 1 }
    if (-not (Test-Path -PathType Leaf (Join-Path $workingDirectory $launcher.Bat))) { exit 1 }
}
exit 0
