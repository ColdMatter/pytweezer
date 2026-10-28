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
                Description = $manifest.description; Admin = [bool]$manifest.admin; ManifestError = $null }
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

Write-Diagnostic 'apply mode is not implemented yet; run with -Check'
exit 1
