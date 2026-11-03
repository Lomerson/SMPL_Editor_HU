# Run from an elevated PowerShell window.
$ErrorActionPreference = "Stop"
$Source = $PSScriptRoot
$Dest = Join-Path $env:ProgramData "SMPLXViewer"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Copy-Item (Join-Path $Source "start_viewer.ps1") $Dest -Force
Copy-Item (Join-Path $Source "healthcheck.ps1") $Dest -Force

$StartCmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Dest\start_viewer.ps1`""
$HealthCmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Dest\healthcheck.ps1`""

schtasks.exe /Create /TN "SMPLX Viewer Start" /TR $StartCmd /SC ONLOGON /RL HIGHEST /F
schtasks.exe /Create /TN "SMPLX Viewer Health" /TR $HealthCmd /SC MINUTE /MO 5 /RL HIGHEST /F
Write-Host "Installed. Run 'schtasks /Run /TN \"SMPLX Viewer Start\"' once to test."
