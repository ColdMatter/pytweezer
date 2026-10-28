param([string]$Repo, [hashtable]$Secrets)
winget install --id astral-sh.uv -e --silent --accept-source-agreements --accept-package-agreements
if ($LASTEXITCODE -ne 0) { throw "winget install uv failed with exit code $LASTEXITCODE" }
