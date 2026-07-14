param (
    [switch]$Bootstrap
)

$VENV_PYTHON = ".\.venv\Scripts\python.exe"
$VENV_UVICORN = ".\.venv\Scripts\uvicorn.exe"
$VENV_STREAMLIT = ".\.venv\Scripts\streamlit.exe"

# If bootstrap flag is supplied, run the data pipelines and train model
if ($Bootstrap) {
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "Running FinLens AI data pipeline & model bootstrap..." -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    & $VENV_PYTHON backend/app/utils/bootstrap.py
    Exit
}

# Check if models and databases exist
if (-not (Test-Path "backend/models/isolation_forest.pkl") -or -not (Test-Path "backend/data/benchmarks.json")) {
    Write-Host "⚠️ Warning: Solvency models or industry benchmark profiles are missing!" -ForegroundColor Yellow
    Write-Host "Please run the data acquisition bootstrap first using:" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 -Bootstrap" -ForegroundColor Green
    Exit
}

Write-Host "🚀 Starting FastAPI Backend (http://127.0.0.1:8000)..." -ForegroundColor Green
Start-Process -NoNewWindow -FilePath $VENV_UVICORN -ArgumentList "backend.app.main:app", "--port", "8000"

# Allow backend 2 seconds to initialize
Start-Sleep -Seconds 2

Write-Host "🎨 Starting Streamlit Dashboard..." -ForegroundColor Green
& $VENV_STREAMLIT run frontend/app.py
