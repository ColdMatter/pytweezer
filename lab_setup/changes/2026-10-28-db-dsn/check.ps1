param([string]$Repo, [hashtable]$Secrets)
if ([Environment]::GetEnvironmentVariable("PYTWEEZER_DB_DSN", "User")) { exit 0 }
exit 1
