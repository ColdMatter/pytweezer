param([string]$Repo, [hashtable]$Secrets)
foreach ($name in "pytweezer-server", "pytweezer-client") {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $task) { exit 1 }
    $workingDirectory = [string]$task.Actions[0].WorkingDirectory
    if ($workingDirectory.TrimEnd("\") -ne $Repo.TrimEnd("\")) { exit 1 }
}
exit 0
