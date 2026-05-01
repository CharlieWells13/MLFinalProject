param(
    [string]$PythonExe = "python",
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$configsDir = Join-Path $scriptDir "configs"
$trainScript = Join-Path $scriptDir "train.py"

if (-not (Test-Path $configsDir)) {
    throw "Configs directory not found: $configsDir"
}
if (-not (Test-Path $trainScript)) {
    throw "Training script not found: $trainScript"
}

$configs = Get-ChildItem -Path $configsDir -Filter "*.yaml" | Sort-Object Name
if ($configs.Count -eq 0) {
    throw "No .yaml config files found in: $configsDir"
}

Write-Host "Found $($configs.Count) config(s) in $configsDir"
$start = Get-Date
$failures = @()

foreach ($cfg in $configs) {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Running config: $($cfg.Name)"
    Write-Host "=================================================="

    & $PythonExe $trainScript --config $cfg.FullName
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "FAILED ($exitCode): $($cfg.Name)"
        $failures += [PSCustomObject]@{
            Config = $cfg.Name
            ExitCode = $exitCode
        }
        if ($StopOnError) {
            break
        }
    } else {
        Write-Host "SUCCESS: $($cfg.Name)"
    }
}

$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host "Grid search completed in $($elapsed.ToString())"

if ($failures.Count -gt 0) {
    Write-Host "Failures: $($failures.Count)"
    $failures | Format-Table -AutoSize
    exit 1
}

Write-Host "All runs completed successfully."
exit 0
