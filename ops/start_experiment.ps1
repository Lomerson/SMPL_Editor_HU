$ErrorActionPreference = "Stop"
$Wsl = Join-Path $env:SystemRoot "System32\wsl.exe"
& $Wsl -d Ubuntu -u lomerson -- bash /home/lomerson/MyProjects/smplx_tpose_viewer_package/ops/start_experiment.sh
exit $LASTEXITCODE
