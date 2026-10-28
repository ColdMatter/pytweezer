param([string]$Repo, [hashtable]$Secrets)
[Environment]::SetEnvironmentVariable("PYTWEEZER_DB_DSN", $Secrets["DB_DSN"], "User")
