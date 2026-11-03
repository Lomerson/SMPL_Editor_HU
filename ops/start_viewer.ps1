$ErrorActionPreference = "Stop"
$Wsl = Join-Path $env:SystemRoot "System32\wsl.exe"
$Tailscale = "D:\ForProgrammingOnly\Tailscale\tailscale.exe"
$Port = 8766
$LogDir = Join-Path $env:LOCALAPPDATA "SMPLXViewer"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "start.log"

function Write-Log([string]$Message) {
    $line = "{0:o} {1}" -f (Get-Date), $Message
    Add-Content -Path $Log -Value $line
    Write-Host $line
}

try {
    Write-Log "starting WSL viewer"
    & $Wsl -d Ubuntu -u lomerson -- bash /home/lomerson/MyProjects/smplx_tpose_viewer_package/ops/start_server.sh
    if ($LASTEXITCODE -ne 0) { throw "WSL start_server.sh failed ($LASTEXITCODE)" }

    Write-Log "checking healthz"
    $health = Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 "http://127.0.0.1:$Port/healthz"
    if ($health.StatusCode -ne 200) { throw "healthz returned $($health.StatusCode)" }

    Write-Log "applying Tailscale Funnel"
    $funnelOutput = & $Tailscale funnel --bg $Port 2>&1
    $funnelText = ($funnelOutput | Out-String)
    if ($funnelText -match "not enabled|administrator|error|failed") {
        Write-Log "Funnel unavailable: $($funnelText.Trim())"
        exit 2
    }
    & $Tailscale funnel status | Add-Content -Path $Log
    Write-Log "viewer is up"
} catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    exit 1
}
