$ErrorActionPreference = "Stop"
$Port = 8766
$LogDir = Join-Path $env:LOCALAPPDATA "SMPLXViewer"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "health.log"

try {
    $health = Invoke-WebRequest -UseBasicParsing -TimeoutSec 6 "http://127.0.0.1:$Port/healthz"
    if ($health.StatusCode -ne 200) { throw "healthz returned $($health.StatusCode)" }
} catch {
    Add-Content -Path $Log -Value ("{0:o} unhealthy: {1}" -f (Get-Date), $_.Exception.Message)
    $start = Join-Path $PSScriptRoot "start_viewer.ps1"
    & $start
    exit $LASTEXITCODE
}

# The app can be healthy while Funnel is absent (e.g. after a reboot or a
# tailnet policy change). Re-apply it on the same 5-minute cadence.
$Tailscale = "D:\ForProgrammingOnly\Tailscale\tailscale.exe"
$funnel = & $Tailscale funnel status 2>&1 | Out-String
if ($funnel -match "No serve config|not enabled") {
    Add-Content -Path $Log -Value ("{0:o} funnel missing; re-applying" -f (Get-Date))
    $start = Join-Path $PSScriptRoot "start_viewer.ps1"
    & $start
    exit $LASTEXITCODE
}
Add-Content -Path $Log -Value ("{0:o} ok" -f (Get-Date))
exit 0
