$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$specFile = Join-Path $projectRoot "packaging\CardLayout.spec"

Push-Location $projectRoot
try {
    python -m PyInstaller --noconfirm --clean $specFile
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller exited with code $LASTEXITCODE"
    }

    $application = Join-Path $projectRoot "dist\CardLayout.exe"
    if (-not (Test-Path -LiteralPath $application -PathType Leaf)) {
        throw "Build finished without producing $application"
    }

    Write-Host ""
    Write-Host "Standalone build completed successfully:"
    Write-Host $application
}
finally {
    Pop-Location
}
