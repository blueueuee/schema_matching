# run_all.ps1 - Automated Setup, Test, Evaluation, and Launch Script
$ErrorActionPreference = "Stop"

Write-Host "=========================================================="
Write-Host "Semantic Schema Matching - End-to-End Execution"
Write-Host "=========================================================="

# 1. Virtual Environment
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[1/5] Creating virtual environment (.venv)..."
    python -m venv .venv
} else {
    Write-Host "[1/5] Virtual environment (.venv) detected."
}

$PYTHON = ".\.venv\Scripts\python.exe"
$STREAMLIT = ".\.venv\Scripts\streamlit.exe"

# 2. Dependencies
Write-Host "[2/5] Installing/verifying dependencies from requirements.txt..."
& $PYTHON -m pip install -q -r requirements.txt

# 3. Tests
Write-Host "[3/5] Running pytest test suite..."
& $PYTHON -m pytest -q

# 4. Evaluation
Write-Host "[4/5] Executing benchmark evaluation across schema pairs..."
& $PYTHON -m evaluation.run_evaluation

# 5. Launch App
Write-Host "[5/5] Starting Streamlit web application..."
# Set PYTHONPATH so Streamlit's script runner finds all project packages
$env:PYTHONPATH = (Resolve-Path ".").Path
& $STREAMLIT run app/streamlit_app.py

