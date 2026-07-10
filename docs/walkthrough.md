# FinSight AI — Setup, Verification & Quickstart Guide

This document describes how to configure, bootstrap, test, and run **FinSight AI** locally or using Docker.

---

## 🛠️ Environment Configuration

The application reads configuration parameters from a local `.env` file. 

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure your settings:
   ```env
   # API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-flash

   # Telemetry & Storage paths
   MLFLOW_TRACKING_URI=sqlite:///backend/data/mlflow.db
   CHROMA_DB_PATH=backend/data/chroma
   ```

---

## 🏃 Launch Instructions

### Option A: Local Execution (PowerShell Setup)
We provide a convenience PowerShell wrapper script `run.ps1` at the root of the repository.

1. **Bootstrap the reference data and models**:
   Before running the app, download industry benchmarks, scrape initial outlier reference profiles from the SEC EDGAR API, and train the Isolation Forest model:
   ```powershell
   .\run.ps1 -Bootstrap
   ```
   *Note: If the SEC API blocks or timeouts, the bootstrapper automatically falls back to generating a high-fidelity synthetic baseline of 500 records to train the model, ensuring zero-setup functionality.*

2. **Boot the FastAPI Backend and Streamlit Dashboard**:
   Launch both services concurrently in a single command:
   ```powershell
   .\run.ps1
   ```
   - Streamlit Dashboard: `http://localhost:8501`
   - FastAPI Backend: `http://localhost:8000`

### Option B: Docker Containerization (Production Deployment)
We have containerized the entire stack for environment-independent deployment:

1. **Build and start the multi-container stack**:
   ```bash
   docker-compose up --build
   ```
2. **Network boundaries**:
   - The FastAPI backend container runs inside a secure, private virtual network. Only the Streamlit frontend container communicates with it via internal DNS (`http://backend:8000`).
   - The user browser only needs public access to Streamlit on `http://localhost:8501`.
   - SQLite files, trained models, and ChromaDB directories persist dynamically in mounted folders `./backend/data` and `./backend/models`.

---

## 🧪 Running Automated Unit Tests

We use `pytest` to run unit tests that check ratio formulas, solvency equations, and anti-hallucination validation rules:

1. Run the test suite:
   ```bash
   python -m pytest tests/
   ```
2. Expected output:
   ```
   tests/test_pipeline.py ..                                              [100%]
   ========================== 2 passed in 10.45s ==========================
   ```

---

## 📊 Telemetry and Experiment Tracking

FinSight AI logs every single analysis run (including uploaded file hashes, calculated ratios, solvency scoring zones, and generated PDF report artifacts) to local **MLflow**.

To view the interactive experiment registry:
1. Start the MLflow UI:
   ```bash
   mlflow ui --port 5000
   ```
2. Open `http://localhost:5000` to compare analysis runs, inspect raw parameters, and download historical reports.
