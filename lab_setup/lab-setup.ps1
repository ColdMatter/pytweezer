<#
.SYNOPSIS
    Checks and applies the machine setup changes in lab_setup/changes/.

.DESCRIPTION
    Reads lab_setup/ from origin/main (or -Ref), never from the working copy:
    fetches, extracts lab_setup/ at that commit into %TEMP% and re-runs itself
    from there with -FromMain.

    -Check prints one JSON object on stdout listing the changes that apply to
    this machine and whether each is done, pending or in error; every
    diagnostic goes to stderr.

    Without -Check, lists the pending changes, asks to confirm (unless -Yes),
    then applies them in order, stopping at the first failure, and appends one
    line per attempt to <Repo>\logs\lab_setup.jsonl. If any needs admin, the
    applying happens in one elevated window. -Relaunch then starts pytweezer
    again from this, non-elevated, window.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File lab_setup\lab-setup.ps1 -Check
#>
param(
    [string]$Repo = (Split-Path $PSScriptRoot -Parent),
    [switch]$Check,
    [switch]$Yes,
    [ValidateSet('server', 'client')][string]$Relaunch,
    [string]$HostName = $env:COMPUTERNAME,
    [switch]$NoFetch,
    [string]$Ref = 'origin/main',
    [string]$FromMain
)

$ErrorActionPreference = 'Stop'
$FetchTimeoutMs = 5000
$CheckTimeoutMs = 5000

function Write-Diagnostic([string]$Message) {
    [Console]::Error.WriteLine("lab-setup: $Message")
}

function Stop-ProcessTree([Diagnostics.Process]$Process) {
    # git.exe and powershell.exe start children that hold our stdout/stderr;
    # killing only the parent would leave the caller waiting on those pipes.
    taskkill /T /F /PID $Process.Id | Out-Null
}

function Update-Ref {
    $branch = $Ref.Substring('origin/'.Length)
    $start = New-Object Diagnostics.ProcessStartInfo 'git', "fetch --quiet origin $branch"
    $start.WorkingDirectory = $Repo
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    # A credential prompt would wait for nobody and hold up pytweezer's startup.
    $start.EnvironmentVariables['GIT_TERMINAL_PROMPT'] = '0'
    $start.EnvironmentVariables['GCM_INTERACTIVE'] = 'never'
    $fetch = [Diagnostics.Process]::Start($start)
    if (-not $fetch.WaitForExit($FetchTimeoutMs)) {
        Stop-ProcessTree $fetch
        Write-Diagnostic "git fetch took over $($FetchTimeoutMs / 1000) s; using $Ref as it is"
    } elseif ($fetch.ExitCode -ne 0) {
        Write-Diagnostic "git fetch failed; using $Ref as it is"
    }
}

function Expand-LabSetup([string]$Commit) {
    $root = Join-Path $env:TEMP 'pytweezer-lab-setup'
    $extracted = Join-Path $root $Commit
    if (Test-Path $extracted) { return $extracted }

    # Unpacked beside the final folder and renamed into place, so a run killed
    # half-way never leaves a partial copy that later runs would reuse.
    $staging = Join-Path $root "$Commit.$PID"
    New-Item -ItemType Directory -Force $root | Out-Null
    $zip = "$staging.zip"
    git -C $Repo archive --format=zip -o $zip $Commit lab_setup
    if ($LASTEXITCODE -ne 0) { throw "git archive of lab_setup/ at $Commit failed" }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $staging)
    Remove-Item $zip
    try {
        [IO.Directory]::Move($staging, $extracted)
    } catch {
        # Another run extracted the same commit first.
        Remove-Item -Recurse -Force $staging
    }
    $extracted
}

function Get-Changes {
    $names = [string[]]@(Get-ChildItem -Directory (Join-Path $PSScriptRoot 'changes') -ErrorAction SilentlyContinue |
        ForEach-Object Name | Where-Object { -not $_.StartsWith('_') })
    # Ordinal, as a culture-aware sort ignores the hyphens in YYYY-MM-DD-slug.
    [Array]::Sort($names, [StringComparer]::Ordinal)

    foreach ($id in $names) {
        $folder = Join-Path $PSScriptRoot "changes\$id"
        try {
            $manifest = Get-Content -Raw -Encoding UTF8 (Join-Path $folder 'change.json') | ConvertFrom-Json
        } catch {
            [pscustomobject]@{ Id = $id; Folder = $folder; Title = $id; Description = ''; Admin = $false
                ManifestError = "change.json: $($_.Exception.Message)" }
            continue
        }
        if (Test-AppliesHere $manifest) {
            [pscustomobject]@{ Id = $id; Folder = $folder; Title = $manifest.title
                Description = $manifest.description; Admin = [bool]$manifest.admin
                Secrets = @($manifest.secrets | Where-Object { $_ }); ManifestError = $null }
        }
    }
}

function Test-AppliesHere($Manifest) {
    if ($Manifest.systems -notcontains 'windows') { return $false }
    if ($null -ne $Manifest.hosts.only) { return $Manifest.hosts.only -icontains $HostName }
    if ($null -ne $Manifest.hosts.except) { return $Manifest.hosts.except -inotcontains $HostName }
    $true
}

function Start-Check($Change) {
    $script = (Join-Path $Change.Folder 'check.ps1').Replace("'", "''")
    $repoArgument = $Repo.Replace("'", "''")
    # Run through a wrapper rather than -File: under -File a thrown error also
    # exits 1, which would read as "pending".
    $command = @"
`$ProgressPreference = 'SilentlyContinue'
try { & '$script' -Repo '$repoArgument'; exit `$LASTEXITCODE }
catch { [Console]::Error.WriteLine(`$_); exit 2 }
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $start = New-Object Diagnostics.ProcessStartInfo 'powershell',
        "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encoded"
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($start)
    [pscustomobject]@{
        Process = $process
        Stdout = $process.StandardOutput.ReadToEndAsync()
        Stderr = $process.StandardError.ReadToEndAsync()
    }
}

function Complete-Check($Running, [datetime]$Deadline) {
    $process = $Running.Process
    $remainingMs = [Math]::Max(0, [int]($Deadline - (Get-Date)).TotalMilliseconds)
    if (-not $process.WaitForExit($remainingMs)) {
        Stop-ProcessTree $process
        return @{ State = 'error'; Error = "check did not finish within $($CheckTimeoutMs / 1000) s" }
    }
    switch ($process.ExitCode) {
        0 { return @{ State = 'done'; Error = $null } }
        1 { return @{ State = 'pending'; Error = $null } }
    }
    $message = $Running.Stderr.Result.Trim()
    if (-not $message) { $message = "check exited with code $($process.ExitCode)" }
    @{ State = 'error'; Error = $message }
}

function Get-ChangeStates($Changes) {
    # Checks run side by side so the startup check stays quick as changes accumulate.
    $deadline = (Get-Date).AddMilliseconds($CheckTimeoutMs)
    $running = @{}
    foreach ($change in $Changes) {
        if (-not $change.ManifestError) { $running[$change.Id] = Start-Check $change }
    }
    $states = @{}
    foreach ($change in $Changes) {
        if ($change.ManifestError) {
            $states[$change.Id] = @{ State = 'error'; Error = $change.ManifestError }
        } else {
            $states[$change.Id] = Complete-Check $running[$change.Id] $deadline
        }
    }
    $states
}

function ConvertTo-AsciiJson($Value) {
    $json = ConvertTo-Json $Value -Depth 4 -Compress
    # Non-ASCII escaped so the JSON survives whatever code page the console uses.
    [regex]::Replace($json, '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
}

function Test-Elevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Show-Pending($Pending) {
    Write-Host "$($Pending.Count) change(s) pending on ${HostName}:"
    foreach ($change in $Pending) {
        $admin = if ($change.Admin) { ' (needs admin)' } else { '' }
        Write-Host "  $($change.Title)$admin"
        Write-Host "      $($change.Description)"
    }
}

function Read-Secrets($Change, [hashtable]$Into) {
    foreach ($secret in $Change.Secrets) {
        # Only the tests set this: Read-Host -AsSecureString reads the console, which they cannot type into.
        if ($env:PYTWEEZER_LAB_SETUP_TEST_SECRET) {
            $Into[$secret.name] = $env:PYTWEEZER_LAB_SETUP_TEST_SECRET
            continue
        }
        $secure = Read-Host -AsSecureString "$($secret.prompt) (ask $($secret.ask))"
        $Into[$secret.name] = (New-Object Management.Automation.PSCredential 'unused', $secure).GetNetworkCredential().Password
    }
}

function Invoke-Change($Change) {
    # Returns the error message with secret values masked, or $null once the check
    # confirms the change is in place.
    $secrets = @{}
    $message = $null
    try {
        Read-Secrets $Change $secrets
        & (Join-Path $Change.Folder 'apply.ps1') -Repo $Repo -Secrets $secrets | Out-Host
    } catch {
        $message = if ($_.Exception.Message) { $_.Exception.Message } else { 'apply.ps1 failed' }
    }
    if (-not $message) {
        $after = (Get-ChangeStates @($Change))[$Change.Id]
        $message = switch ($after.State) {
            'done' { $null }
            'pending' { 'apply finished but the check still says pending' }
            default { "check after apply: $($after.Error)" }
        }
    }
    foreach ($value in $secrets.Values) {
        if ($message -and $value) { $message = $message.Replace($value, '***') }
    }
    $message
}

function Write-Record($Change, [string]$Result, $Message) {
    New-Item -ItemType Directory -Force (Split-Path $RecordPath) | Out-Null
    $record = [ordered]@{
        # Invariant culture: in a custom format ':' is the culture's time separator.
        time = [DateTime]::UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'+00:00'", [Globalization.CultureInfo]::InvariantCulture)
        host = $HostName
        change = $Change.Id
        commit = $FromMain
        result = $Result
        error = $Message
    }
    # Not Add-Content -Encoding UTF8, which starts a new file with a BOM.
    [IO.File]::AppendAllText($RecordPath, (ConvertTo-AsciiJson $record) + "`n")
    $record
}

function Read-RecordLines {
    if (Test-Path $RecordPath) { @(Get-Content -Encoding UTF8 $RecordPath) } else { @() }
}

function Invoke-Elevated {
    # The elevated window applies every pending change, admin or not, so they stay in order.
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -FromMain $FromMain -Repo `"$Repo`" -Yes -HostName `"$HostName`""
    $before = (Read-RecordLines).Count
    try {
        $elevated = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList $arguments
    } catch {
        Write-Host "Not elevated, so nothing was applied: $($_.Exception.Message)"
        return @{}
    }
    $results = @{}
    foreach ($line in (Read-RecordLines | Select-Object -Skip $before)) {
        $record = $line | ConvertFrom-Json
        $results[$record.change] = $record
    }
    if (-not $results.Count -and $elevated.ExitCode -ne 0) {
        Write-Host "The elevated window stopped with exit code $($elevated.ExitCode) before applying anything."
    }
    $results
}

function Show-Summary($Pending, $Results) {
    Write-Host ''
    Write-Host 'Summary:'
    foreach ($change in $Pending) {
        $result = $Results[$change.Id]
        $outcome = if (-not $result) { 'not attempted' }
            elseif ($result.result -eq 'done') { 'done' }
            else { "FAILED: $($result.error)" }
        Write-Host "  $($change.Title): $outcome"
    }
}

if (-not $FromMain) {
    $Repo = (Resolve-Path $Repo).Path
    if (-not $NoFetch -and $Ref.StartsWith('origin/')) { Update-Ref }
    $commit = git -C $Repo rev-parse --verify --quiet "$Ref^{commit}"
    if (-not $commit) {
        Write-Diagnostic "no $Ref in $Repo to read lab_setup/ from"
        exit 2
    }
    $extracted = Expand-LabSetup $commit
    $forward = @{}
    foreach ($key in $PSBoundParameters.Keys) { $forward[$key] = $PSBoundParameters[$key] }
    $forward.Repo = $Repo
    $forward.FromMain = $commit
    & (Join-Path $extracted 'lab_setup\lab-setup.ps1') @forward
    exit $LASTEXITCODE
}

$changes = @(Get-Changes)
$states = Get-ChangeStates $changes

if ($Check) {
    $listed = @(foreach ($change in $changes) {
        [ordered]@{
            id = $change.Id
            title = $change.Title
            description = $change.Description
            admin = $change.Admin
            state = $states[$change.Id].State
            error = $states[$change.Id].Error
        }
    })
    [Console]::Out.WriteLine((ConvertTo-AsciiJson ([ordered]@{ commit = $FromMain; host = $HostName; changes = $listed })))
    exit 0
}

$RecordPath = Join-Path $Repo 'logs\lab_setup.jsonl'
foreach ($change in $changes) {
    if ($states[$change.Id].State -eq 'error') {
        Write-Host "Could not check '$($change.Title)', so it is left alone: $($states[$change.Id].Error)"
    }
}
$pending = @($changes | Where-Object { $states[$_.Id].State -eq 'pending' })
$allDone = $true
if ($pending.Count -eq 0) {
    Write-Host "Nothing to apply on $HostName."
} else {
    Show-Pending $pending
    if (-not ($Yes -or ((Read-Host "Apply these $($pending.Count) changes? [y/N]") -match '^y(es)?$'))) {
        Write-Host 'Nothing applied.'
    } else {
        if (@($pending | Where-Object Admin).Count -and -not (Test-Elevated)) {
            $results = Invoke-Elevated
        } else {
            $results = @{}
            foreach ($change in $pending) {
                Write-Host ''
                Write-Host "Applying: $($change.Title)"
                $message = Invoke-Change $change
                $results[$change.Id] = Write-Record $change $(if ($message) { 'failed' } else { 'done' }) $message
                if ($message) { break }
            }
        }
        Show-Summary $pending $results
        $allDone = -not @($pending | Where-Object { $results[$_.Id].result -ne 'done' }).Count
    }
}

if ($Relaunch) {
    $null = Read-Host "Press Enter to start pytweezer-$Relaunch"
    $launcher = @{ server = 'start_servers.bat'; client = 'start_client.bat' }[$Relaunch]
    # A full path, not "cmd /c start_servers.bat": with NoDefaultCurrentDirectoryInExePath
    # set, cmd does not look in the working directory.
    Start-Process (Join-Path $Repo $launcher) -WorkingDirectory $Repo
}
if ($allDone) { exit 0 } else { exit 1 }
