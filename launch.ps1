# Starts EchoTwin: one server, one page.
#   Dashboard (laptop):  http://localhost:8000/        Phone:  https://<laptop-ip>:8443/phone
# Run from the repo root:  .\launch.ps1     Press any key to stop it.
# The Python to use comes from .env (ROBOT_PY); the 3D pipeline uses PERCEPTION_PY (see the README).

$root = $PSScriptRoot
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }
}
# Paths in .env may be relative to the repository
function Resolve-Repo($p) { if ([System.IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $root $p } }
$robotPy = if ($env:ROBOT_PY) { Resolve-Repo $env:ROBOT_PY } else { Join-Path $root ".venv-robot\Scripts\python.exe" }

if (-not (Test-Path $robotPy)) {
    Write-Host "ROBOT_PY not found: $robotPy. Set ROBOT_PY in .env (see .env.example and the README)." -ForegroundColor Red
    exit 1
}
if (-not $env:PERCEPTION_PY) {
    Write-Host "PERCEPTION_PY is not set: scans will use the quick one-photo mode. See the README to enable the 3D model." -ForegroundColor Yellow
}

$cmd = "Set-Location '$root'; & '$robotPy' -m echotwin.robot.server"
$bytes = [Text.Encoding]::Unicode.GetBytes($cmd)
$p = Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", ([Convert]::ToBase64String($bytes)) -PassThru

Start-Sleep -Seconds 3
Start-Process "http://localhost:8000"
Write-Host "  EchoTwin: http://localhost:8000   Phone: https://<laptop-ip>:8443/phone"
Write-Host "  Press any key to stop."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
