$ErrorActionPreference = "Stop"
$Port = 8766
$LogDir = Join-Path $env:LOCALAPPDATA "SMPLXViewer"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "health.log"

try {
    $health = Invoke-WebRequest -UseBasicParsing -TimeoutSec 6 "http://127.0.0.1:$Port/healthz"
    if ($health.StatusCode -eq 200) {
        Add-Content -Path $Log -Value ("{0:o} ok" -f (Get-Date))
        exit 0
    }
    throw "healthz returned $($health.StatusCode)"
} catch {
    Add-Content -Path $Log -Value ("{0:o} unhealthy: {1}" -f (Get-Date), $_.Exception.Message)
    $start = Join-Path $PSScriptRoot "start_viewer.ps1"
    & $start
    exit $LASTEXITCODE
}
