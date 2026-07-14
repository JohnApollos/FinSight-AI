# FinSight AI — Setup, Verification & Quickstart Guide

This document describes how to configure, bootstrap, test, and run **FinSight AI** locally or using Docker.

---

## Environment Configuration

The application reads configuration parameters from a local `.env` file. 

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure your settings:
   ```env
   # API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-3.5-flash

   # Telemetry & Storage paths
   MLFLOW_TRACKING_URI=sqlite:///backend/data/mlflow.db
   CHROMA_DB_PATH=backend/data/chroma
   ```

---

## Launch Instructions

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

## Running Automated Unit Tests

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

## Telemetry and Experiment Tracking

FinSight AI logs every single analysis run (including uploaded file hashes, calculated ratios, solvency scoring zones, and generated PDF report artifacts) to local **MLflow**.

To view the interactive experiment registry:
1. Start the MLflow UI:
   ```bash
   mlflow ui --port 5000
   ```
2. Open `http://localhost:5000` to compare analysis runs, inspect raw parameters, and download historical reports.

---

## Real-World Validation Sweep

The application has been verified using live annual reports from the Nairobi Securities Exchange (NSE):

1. **East African Breweries Limited (EABL)**: Scored **12.72 (Safe Zone)** and an Anomaly Outlier Score of **25.6% (Standard Signature)**. The PDF report successfully compiled, plotted ratios, and was exported.
2. **Equity Group Holdings Plc**: Scored **0.30 (Distress Zone)** under the standard Fintech/Service profile due to structural differences in commercial bank balance sheets (the **Structural Metric Trap**). When switched to the **Banking** sector, the system overlayed CBK guidelines to resolve it to the **Safe Zone**, with a Capital Adequacy Ratio of **18.1%** and a standard anomaly signature of **1.8%**.

---

## 🚀 Refactoring & Quality Alignment Sweep (July 14, 2026)

We executed a comprehensive refactoring sweep of the FastAPI backend and Streamlit frontend codebases to eliminate inconsistencies and upgrade the system to enterprise grade:

### 1. Core Model & Ratio Alignment (Collinearity Fixed)
* **The Inconsistency**: In `train_model.py`, quick and cash ratios were calculated as simple linear multiples of the current ratio (`0.7` and `0.35` respectively) because `sec_reference_data.csv` lacked specific asset-mix columns. This created a perfect collinearity baseline. At runtime, the actual cash, inventory, and receivables were used, causing the Isolation Forest to flag healthy runtime companies as outliers due to distribution shift.
* **The Fix**: 
  - Updated `sec_downloader.py` to extract `cash`, `inventory`, and `receivables` directly from SEC XBRL facts.
  - Aligned `train_model.py`'s ratio calculations exactly with `analysis.py`'s proxy current asset/liabilities formulations.
  - Updated `generate_synthetic_data` to simulate realistic asset-mix distributions.
  - Re-ran the bootstrapper to train a fresh `isolation_forest.pkl` model based on aligned features.

### 2. Double-Analysis Prevention (MD5 Hashing Cache)
* **The Inconsistency**: Streamlit’s rendering loop automatically re-requested report byte streams from `/report` on page updates, triggering duplicate PDF text extraction, Gemini LLM calls, and vector DB queries. This exhausted Gemini API limits (triggering `429 RESOURCE_EXHAUSTED` codes) and slowed the page down to 30+ seconds.
* **The Fix**: 
  - Added MD5 file hashing cache to `/analyze` and `/report` (saving responses under `backend/data/temp/analysis_{file_hash}_{sector}.json`).
  - Subsequent requests for the same document under the same sector are served instantly from the JSON cache in under `0.01 seconds` using `0 Gemini tokens`.

### 3. Concurrency and Decoupled Image Serving
* **The Inconsistency**: Matplotlib saved charts to static files (`shap_waterfall.png` and `ratio_comparison.png`), causing collisions and image leakage on parallel requests. The frontend also read these files directly from the local disk, violating container boundaries.
* **The Fix**:
  - Saved charts to request-isolated paths: `ratio_comparison_{file_id}.png` and `shap_waterfall_{file_id}.png`.
  - Added a backend GET `/charts/shap/{file_id}` HTTP endpoint. The Streamlit frontend now fetches images cleanly over network requests instead of local disk reads.
  - Comparison charts are deleted immediately after the PDF is built, preserving disk space.

### 4. Pydantic Warning Cleanups
* Refactored `config.py` to use Pydantic Settings V2 (`SettingsConfigDict`), completely eliminating all Pydantic V2 deprecation warnings in the server start console.

### 5. Robust Fallback Expense Parsing
* Added regex keyword matching for `expenses` to `regex_extract_financials` inside `ingestion.py`. Calculated expenses from `revenue - net_income` now runs only as a fallback when not explicitly found in text, ensuring double-entry verification errors are caught.

