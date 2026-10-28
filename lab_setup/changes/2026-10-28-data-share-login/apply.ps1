param([string]$Repo, [hashtable]$Secrets)
cmdkey /add:10.59.3.1 /user:PH-BEAST\pytweezer-reader /pass:$($Secrets["READER_PASSWORD"]) | Out-Null
if ($LASTEXITCODE -ne 0) { throw "cmdkey failed with exit code $LASTEXITCODE" }
