param([string]$Repo, [hashtable]$Secrets)
if ((cmdkey /list:10.59.3.1) -match "pytweezer-reader") { exit 0 }
exit 1
