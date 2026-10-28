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
    again from this, non-elevated, window, even if something failed.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File lab_setup\lab-setup.ps1 -Check
#>
# Branches older than main run main's copy of this script and parse its -Check
# JSON, so these parameters and the JSON fields may only be added to, never
# renamed or removed.
param(
    [string]$Repo = (Split-Path $PSScriptRoot -Parent),
    [switch]$Check,
    [switch]$Yes,
    [ValidateSet('server', 'client')][string]$Relaunch,
    [string]$HostName = $env:COMPUTERNAME,
    [switch]$NoFetch,
    [string]$Ref = 'origin/main',
    [string]$FromMain,
    [switch]$PauseOnFailure
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

function Format-Argument([string]$Value) {
    # Backslashes before the closing quote are doubled, or they would escape it.
    '"' + ($Value -replace '\\+$', '$0$0') + '"'
}

function Start-Check($Change) {
    $arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File {0} -Script {1} -Repo {2}' -f
        (Format-Argument (Join-Path $PSScriptRoot 'run-check.ps1')),
        (Format-Argument (Join-Path $Change.Folder 'check.ps1')),
        (Format-Argument $Repo)
    $start = New-Object Diagnostics.ProcessStartInfo 'powershell', $arguments
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
    $line = (ConvertTo-AsciiJson $record) + "`n"
    # Not Add-Content -Encoding UTF8, which starts a new file with a BOM.
    $stream = [IO.File]::Open($RecordPath, 'OpenOrCreate', 'ReadWrite')
    try {
        if ($stream.Length -gt 0) {
            # A run killed mid-line would otherwise glue this record onto that line.
            $null = $stream.Seek(-1, 'End')
            if ($stream.ReadByte() -ne 10) { $line = "`n$line" }
        }
        $null = $stream.Seek(0, 'End')
        $bytes = [Text.Encoding]::ASCII.GetBytes($line)
        $stream.Write($bytes, 0, $bytes.Length)
    } finally {
        $stream.Dispose()
    }
    $record
}

function Read-RecordLines {
    if (Test-Path $RecordPath) { @(Get-Content -Encoding UTF8 $RecordPath) } else { @() }
}

function Invoke-Elevated {
    # The elevated window applies every pending change, admin or not, so they stay in order.
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -FromMain $FromMain -Repo `"$Repo`" -Yes -HostName `"$HostName`" -PauseOnFailure"
    $before = (Read-RecordLines).Count
    try {
        $elevated = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList $arguments
    } catch {
        Write-Host "Not elevated, so nothing was applied: $($_.Exception.Message)"
        return @{}
    }
    $results = @{}
    foreach ($line in (Read-RecordLines | Select-Object -Skip $before)) {
        try { $record = $line | ConvertFrom-Json } catch { continue }
        if ($record.change) { $results[$record.change] = $record }
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

function Wait-Enter([string]$Prompt) {
    # Read-Host throws when there is no console to read; carry on regardless.
    try { $null = Read-Host $Prompt } catch { }
}

function Update-Environment {
    # Whatever started us exited holding the environment it started with, so
    # variables set by a change since (PYTWEEZER_DB_DSN, uv on PATH) are re-read.
    $paths = @()
    foreach ($scope in 'Machine', 'User') {
        $variables = [Environment]::GetEnvironmentVariables($scope)
        foreach ($name in $variables.Keys) {
            if ($name -ieq 'Path') { $paths += $variables[$name] }
            else { [Environment]::SetEnvironmentVariable($name, $variables[$name], 'Process') }
        }
    }
    [Environment]::SetEnvironmentVariable('Path', ($paths -join ';'), 'Process')
}

function Start-Pytweezer {
    Wait-Enter "Press Enter to start pytweezer-$Relaunch"
    $launcher = Join-Path $Repo (@{ server = 'start_servers.bat'; client = 'start_client.bat' }[$Relaunch])
    if (-not (Test-Path -PathType Leaf $launcher)) {
        Write-Host "Cannot start pytweezer-${Relaunch}: $launcher is missing. Start it by hand."
        return
    }
    try {
        Update-Environment
        # A full path, not "cmd /c start_servers.bat": with NoDefaultCurrentDirectoryInExePath
        # set, cmd does not look in the working directory.
        Start-Process $launcher -WorkingDirectory $Repo
    } catch {
        Write-Host "Cannot start pytweezer-${Relaunch}: $($_.Exception.Message)"
        Wait-Enter 'Press Enter to close'
    }
}

if (-not $FromMain) {
    try {
        $Repo = (Resolve-Path $Repo).Path
        if (-not $NoFetch -and $Ref.StartsWith('origin/')) { Update-Ref }
        $commit = git -C $Repo rev-parse --verify --quiet "$Ref^{commit}"
        if (-not $commit) { throw "no $Ref in $Repo to read lab_setup/ from" }
        $extracted = Expand-LabSetup $commit
    } catch {
        Write-Diagnostic $_.Exception.Message
        if ($Relaunch) { Start-Pytweezer }
        exit 2
    }
    $forward = @{}
    foreach ($key in $PSBoundParameters.Keys) { $forward[$key] = $PSBoundParameters[$key] }
    $forward.Repo = $Repo
    $forward.FromMain = $commit
    try {
        & (Join-Path $extracted 'lab_setup\lab-setup.ps1') @forward
    } catch {
        # Reached only if that copy could not start, e.g. it rejected a parameter.
        Write-Diagnostic $_.Exception.Message
        if ($Relaunch) { Start-Pytweezer }
        exit 2
    }
    exit $LASTEXITCODE
}

if ($Check) {
    $changes = @(Get-Changes)
    $states = Get-ChangeStates $changes
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
$allDone = $false
try {
    $changes = @(Get-Changes)
    $states = Get-ChangeStates $changes
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
            $allDone = $false
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
} catch {
    $allDone = $false
    Write-Host ''
    Write-Host "Stopped by an error: $($_.Exception.Message)" -ForegroundColor Red
}

if ($PauseOnFailure -and -not $allDone) { Wait-Enter 'Press Enter to close' }
if ($Relaunch) { Start-Pytweezer }
if ($allDone) { exit 0 } else { exit 1 }
