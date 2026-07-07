# ExamShield Direct Run (Windows) — for webcam access
# Docker Desktop cannot pass the webcam into containers on Windows.
# This script runs ExamShield locally while the backend stays in Docker.
#
# IMPORTANT: If Docker is running with all services, stop ExamShield first:
#   docker compose stop examshield-frontend

# Anchor CWD to this script's directory so the script works no matter where
# the user invokes it from (project root, desktop, etc.).  Push-Location lets
# us restore the caller's CWD on exit.
Push-Location -Path $PSScriptRoot

try {
    $port = if ($env:STREAMLIT_PORT) { $env:STREAMLIT_PORT } else { "8601" }

    $env:BACKEND_API_URL = "http://localhost:8000"
    $env:BROWSER_API_URL = "http://localhost:8000"

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  ExamShield AI - Direct Run Mode" -ForegroundColor Cyan
    Write-Host "  Port: $port" -ForegroundColor Cyan
    Write-Host "  Backend: $env:BACKEND_API_URL" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "TIP: If port $port is busy, set STREAMLIT_PORT:" -ForegroundColor Yellow
    Write-Host '  $env:STREAMLIT_PORT = "9001"; .\run_direct.ps1' -ForegroundColor Yellow
    Write-Host ""

    # NOTE: Do NOT use --server.address 0.0.0.0 on Windows!
    # Hyper-V (required by Docker Desktop) reserves random port ranges on 0.0.0.0,
    # causing WinError 10013 even on seemingly free ports.
    # Streamlit defaults to localhost which works correctly.
    streamlit run app.py --server.port $port
}
finally {
    Pop-Location
}
