# Starts the perception web app (port 8765) and the robot server (8000 / 8443).
# Run from the repo root:  .\launch.ps1     Press any key to stop both.
# Python paths come from .env (PERCEPTION_PY, ROBOT_PY).

$root = $PSScriptRoot
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }
}
$perceptionPy = if ($env:PERCEPTION_PY) { $env:PERCEPTION_PY } else { (Get-Command python).Source }
$robotPy      = if ($env:ROBOT_PY) { $env:ROBOT_PY } else { Join-Path $root ".venv\Scripts\python.exe" }

if (-not (Test-Path $robotPy)) {
    Write-Host "ROBOT_PY not found: $robotPy. Set ROBOT_PY in .env (see .env.example)." -ForegroundColor Red
    exit 1
}

$perceptionPort = 8765
$perceptionCmd = "Set-Location '$root'; `$env:PERCEPTION_PY='$perceptionPy'; & '$perceptionPy' apps/perception_web/server.py $perceptionPort"
$robotCmd      = "Set-Location '$root'; & '$robotPy' -m echotwin.robot.server"
$enc = { param($c) [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($c)) }
$p1 = Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", (& $enc $perceptionCmd) -PassThru
$p2 = Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", (& $enc $robotCmd) -PassThru

Start-Sleep -Seconds 3
Start-Process "http://127.0.0.1:$perceptionPort"
Start-Process "http://localhost:8000"
Write-Host "  Perception: http://127.0.0.1:$perceptionPort   Robot: http://localhost:8000   Phone: https://<laptop-ip>:8443/phone"
Write-Host "  Press any key to stop both servers."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
foreach ($p in $p1, $p2) { try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {} }
