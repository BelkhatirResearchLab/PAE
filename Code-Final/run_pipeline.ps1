[CmdletBinding()]
param(
    [ValidateSet("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "all")]
    [string]$Step = "all",
    [ValidateSet("full", "quick")]
    [string]$Mode = "full",
    [string]$Config = "config\pipeline_config.json",
    [string]$Python = "python",
    [string]$Matlab = "matlab",
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$codeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$arguments = @(
    (Join-Path $codeRoot "run_all.py"),
    "--step", $Step,
    "--mode", $Mode,
    "--config", $Config,
    "--python", $Python,
    "--matlab", $Matlab
)
if ($Resume) { $arguments += "--resume" }
if ($DryRun) { $arguments += "--dry-run" }

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Pipeline failed with exit code $LASTEXITCODE"
}
