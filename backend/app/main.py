import os
import sys
import time
import json
import uuid
import shutil
import hashlib
import sqlite3
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from datetime import datetime

# Setup Python Path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from backend.app.config import settings
from backend.app.utils.logger import setup_logger

# Initialize Logger
logger = setup_logger("finlens_api")

from backend.app.services.ingestion import extract_financial_metrics
from backend.app.services.analysis import evaluate_financial_health
from backend.app.services.narrative import generate_risk_narrative
from backend.app.services.report import compile_pdf_report

# Optimize SQLite with WAL (Write-Ahead Logging) to prevent locks during concurrent access
try:
    db_path = "backend/data/mlflow.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.close()
    logger.info("SQLite database optimized successfully with WAL (Write-Ahead Logging) mode.")
except Exception as db_err:
    logger.warning(f"Could not optimize SQLite with WAL mode: {db_err}")

# Initialize FastAPI
app = FastAPI(
    title="FinLens AI API",
    description="Intelligent financial document ingestion and risk analytics engine.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants & Paths
DATA_DIR = "backend/data"
TEMP_DIR = os.path.join(DATA_DIR, "temp")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize MLflow tracking
try:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("FinLens_AI_Document_Analysis")
    logger.info(f"MLflow initialized with tracking URI: {settings.MLFLOW_TRACKING_URI}")
except Exception as ml_err:
    logger.warning(f"Failed to initialize MLflow: {ml_err}")

# In-memory metrics fallback & persistence
def load_history() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_history(record: Dict[str, Any]):
    history = load_history()
    history.insert(0, record) # Add to front (latest first)
    history = history[:10] # Cap at last 10 entries
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Failed to save history: {e}")

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    model_exists = os.path.exists("backend/models/isolation_forest.pkl")
    benchmarks_exist = os.path.exists("backend/data/benchmarks.json")
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model_exists,
        "benchmarks_loaded": benchmarks_exist,
        "api_key_configured": bool(settings.GEMINI_API_KEY)
    }

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    sector: str = Form("Fintech"),
    gemini_key: str = Form("")
):
    """
    Ingests a document, extracts financials, runs analysis, 
    generates a risk narrative, and logs results to MLflow.
    Utilizes MD5 hash file caching to prevent duplicate API executions.
    """
    start_time = time.time()
    
    # Read file bytes to calculate MD5 hash
    file_bytes = await file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    # We include the sector in the cache file name because solvency metrics/benchmarks
    # are completely different depending on the target analysis sector!
    unique_identifier = f"{file_hash}_{sector.replace(' ', '_')}"
    cache_json_path = os.path.join(TEMP_DIR, f"analysis_{unique_identifier}.json")
    
    if os.path.exists(cache_json_path):
        print(f"Cache hit: Loading analysis for {file.filename} (Hash: {file_hash}) under sector {sector}...")
        try:
            with open(cache_json_path, "r", encoding="utf-8") as cache_f:
                cached_result = json.load(cache_f)
                cached_result["execution_time_seconds"] = round(time.time() - start_time, 3)
                
                # Cache to persistent history log
                history_record = {
                    "id": file_hash,
                    "filename": file.filename,
                    "company_name": cached_result["company_name"],
                    "sector": sector,
                    "reporting_period": cached_result["reporting_period"],
                    "z_score": round(cached_result["analysis"]["solvency"]["score"], 2),
                    "solvency_zone": cached_result["analysis"]["solvency"]["zone"],
                    "anomaly_score": round(cached_result["analysis"]["anomaly"]["score"], 1),
                    "is_anomaly": cached_result["analysis"]["anomaly"]["is_anomaly"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                save_to_history(history_record)
                
                return JSONResponse(content=cached_result)
        except Exception as cache_err:
            print(f"Warning: Failed to load cached analysis: {cache_err}. Re-running pipeline.")
            
    # Cache miss: Save file bytes to temp path
    _, ext = os.path.splitext(file.filename)
    temp_file_path = os.path.join(TEMP_DIR, f"{unique_identifier}{ext}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_bytes)
            
        # 1. Ingestion & Structured Extraction
        api_key = gemini_key.strip() if gemini_key.strip() else settings.GEMINI_API_KEY
        extracted = extract_financial_metrics(temp_file_path, gemini_api_key=api_key)
        extracted["sic"] = sector
        
        # 2. Financial Ratio & Solvency Anomaly Engine
        # Pass unique_identifier to evaluate_financial_health for request-isolated SHAP charts
        analysis = evaluate_financial_health(extracted, unique_identifier)
        
        # Override local image path in JSON with custom API retrieval endpoint
        analysis["anomaly"]["chart_path"] = f"/charts/shap/{unique_identifier}"
        
        # 3. Agentic Risk Narrative
        narrative = generate_risk_narrative(
            extracted["company_name"], 
            sector, 
            analysis["ratios"], 
            analysis["solvency"], 
            analysis["anomaly"], 
            gemini_api_key=api_key
        )
        
        execution_time = time.time() - start_time
        
        # Compile response
        result = {
            "id": file_hash,
            "filename": file.filename,
            "company_name": extracted["company_name"],
            "reporting_period": extracted["reporting_period"],
            "currency": extracted["currency"],
            "extraction_method": extracted["extraction_method"],
            "validation": {
                "is_valid": extracted["is_valid"],
                "warnings": extracted.get("validation_warnings", [])
            },
            "metrics": {
                "assets": extracted["assets"],
                "liabilities": extracted["liabilities"],
                "equity": extracted["equity"],
                "revenue": extracted["revenue"],
                "expenses": extracted.get("expenses", 0.0),
                "net_income": extracted["net_income"],
                "interest_expense": extracted.get("interest_expense", 0.0),
                "working_capital": extracted["working_capital"],
                "cash": extracted["cash"],
                "retained_earnings": extracted["retained_earnings"]
            },
            "analysis": analysis,
            "narrative": narrative,
            "execution_time_seconds": round(execution_time, 2)
        }
        
        # Log to MLflow
        try:
            with mlflow.start_run():
                mlflow.set_tag("company_name", result["company_name"])
                mlflow.set_tag("sector", sector)
                mlflow.set_tag("reporting_period", result["reporting_period"])
                mlflow.set_tag("extraction_method", result["extraction_method"])
                
                # Log metrics
                mlflow.log_metric("assets", result["metrics"]["assets"])
                mlflow.log_metric("liabilities", result["metrics"]["liabilities"])
                mlflow.log_metric("equity", result["metrics"]["equity"])
                mlflow.log_metric("revenue", result["metrics"]["revenue"])
                mlflow.log_metric("net_income", result["metrics"]["net_income"])
                mlflow.log_metric("altman_z_score", analysis["solvency"]["score"])
                mlflow.log_metric("anomaly_score", analysis["anomaly"]["score"])
                mlflow.log_metric("is_anomaly", 1.0 if analysis["anomaly"]["is_anomaly"] else 0.0)
                mlflow.log_metric("execution_time", execution_time)
                
                # Log files as artifacts
                mlflow.log_artifact(temp_file_path, artifact_path="raw_documents")
                
                # Write and log JSON results
                temp_json_path = os.path.join(TEMP_DIR, f"{file_hash}.json")
                with open(temp_json_path, "w", encoding="utf-8") as json_f:
                    json.dump(result, json_f, indent=2)
                mlflow.log_artifact(temp_json_path, artifact_path="analysis_results")
                
                try:
                    os.remove(temp_json_path)
                except Exception:
                    pass
        except Exception as ml_log_err:
            print(f"Warning: Failed to log run to MLflow: {ml_log_err}")
            
        # Write to cache JSON
        try:
            with open(cache_json_path, "w", encoding="utf-8") as cache_f:
                json.dump(result, cache_f, indent=2)
        except Exception as cache_write_err:
            print(f"Warning: Failed to write cache JSON: {cache_write_err}")
            
        # Cache to persistent history
        history_record = {
            "id": file_hash,
            "filename": file.filename,
            "company_name": result["company_name"],
            "sector": sector,
            "reporting_period": result["reporting_period"],
            "z_score": round(analysis["solvency"]["score"], 2),
            "solvency_zone": analysis["solvency"]["zone"],
            "anomaly_score": round(analysis["anomaly"]["score"], 1),
            "is_anomaly": analysis["anomaly"]["is_anomaly"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_to_history(history_record)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Error analyzing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded raw document
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

@app.post("/report")
async def generate_report(
    file: UploadFile = File(...),
    sector: str = Form("Fintech"),
    gemini_key: str = Form("")
):
    """
    Ingests file, processes analysis (using cache if available), compiles the ReportLab PDF, 
    and streams it back as a download.
    """
    file_bytes = await file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    unique_identifier = f"{file_hash}_{sector.replace(' ', '_')}"
    cache_json_path = os.path.join(TEMP_DIR, f"analysis_{unique_identifier}.json")
    pdf_output_path = os.path.join(TEMP_DIR, f"{unique_identifier}.pdf")
    
    try:
        # Check cache JSON first to avoid calling Gemini extraction again
        if os.path.exists(cache_json_path):
            print(f"Cache hit for PDF report: Loading cached data for {file.filename}...")
            with open(cache_json_path, "r", encoding="utf-8") as cache_f:
                result = json.load(cache_f)
                
            extracted = {
                "company_name": result["company_name"],
                "reporting_period": result["reporting_period"],
                "currency": result["currency"],
                "assets": result["metrics"]["assets"],
                "liabilities": result["metrics"]["liabilities"],
                "equity": result["metrics"]["equity"],
                "revenue": result["metrics"]["revenue"],
                "expenses": result["metrics"]["expenses"],
                "net_income": result["metrics"]["net_income"],
                "interest_expense": result["metrics"]["interest_expense"],
                "working_capital": result["metrics"]["working_capital"],
                "cash": result["metrics"]["cash"],
                "retained_earnings": result["metrics"]["retained_earnings"]
            }
            
            # Map anomaly chart path temporarily back to the local file for the PDF builder
            analysis_data = result["analysis"]
            analysis_data["anomaly"]["chart_path"] = os.path.join(TEMP_DIR, f"shap_waterfall_{unique_identifier}.png")
            
            compile_pdf_report(
                pdf_output_path,
                result["company_name"],
                sector,
                result["reporting_period"],
                result["currency"],
                extracted,
                analysis_data,
                result["narrative"]
            )
            
            return FileResponse(
                pdf_output_path, 
                media_type="application/pdf", 
                filename=f"FinLens_Audit_Report_{result['company_name'].replace(' ', '_')}.pdf"
            )
            
        # Fallback (Cache miss)
        _, ext = os.path.splitext(file.filename)
        temp_file_path = os.path.join(TEMP_DIR, f"{unique_identifier}{ext}")
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_bytes)
            
        api_key = gemini_key.strip() if gemini_key.strip() else settings.GEMINI_API_KEY
        extracted = extract_financial_metrics(temp_file_path, gemini_api_key=api_key)
        extracted["sic"] = sector
        
        analysis = evaluate_financial_health(extracted, unique_identifier)
        
        narrative = generate_risk_narrative(
            extracted["company_name"], 
            sector, 
            analysis["ratios"], 
            analysis["solvency"], 
            analysis["anomaly"], 
            gemini_api_key=api_key
        )
        
        compile_pdf_report(
            pdf_output_path,
            extracted["company_name"],
            sector,
            extracted["reporting_period"],
            extracted["currency"],
            extracted,
            analysis,
            narrative
        )
        
        return FileResponse(
            pdf_output_path, 
            media_type="application/pdf", 
            filename=f"FinLens_Audit_Report_{extracted['company_name'].replace(' ', '_')}.pdf"
        )
        
    except Exception as e:
        print(f"Error compiling PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded raw document
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

@app.get("/charts/shap/{file_id}")
def get_shap_chart(file_id: str):
    """Serves the request-isolated generated SHAP waterfall chart."""
    chart_path = os.path.join(TEMP_DIR, f"shap_waterfall_{file_id}.png")
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png")
    # Fallback to default path
    default_path = "backend/data/shap_waterfall.png"
    if os.path.exists(default_path):
        return FileResponse(default_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="SHAP chart not found")

@app.get("/history")
def get_analysis_history():
    """Returns the cached list of the last 10 analyzed files."""
    return JSONResponse(content=load_history())

@app.get("/metrics")
def get_telemetry_metrics():
    """Aggregates telemetry stats from the history log."""
    history = load_history()
    total_runs = len(history)
    
    if total_runs == 0:
        return {
            "total_documents_analyzed": 0,
            "anomaly_rate_percent": 0.0,
            "mean_solvency_score": 0.0,
            "solvency_distribution": {"Safe": 0, "Grey": 0, "Distress": 0}
        }
        
    anomalies = sum(1 for r in history if r.get("is_anomaly"))
    z_scores = [r.get("z_score", 0.0) for r in history]
    
    distribution = {"Safe": 0, "Grey": 0, "Distress": 0}
    for r in history:
        zone = r.get("solvency_zone", "")
        if "safe" in zone.lower():
            distribution["Safe"] += 1
        elif "grey" in zone.lower():
            distribution["Grey"] += 1
        else:
            distribution["Distress"] += 1
            
    return {
        "total_documents_analyzed": total_runs,
        "anomaly_rate_percent": round((anomalies / total_runs) * 100.0, 1),
        "mean_solvency_score": round(sum(z_scores) / total_runs, 2),
        "solvency_distribution": distribution
    }
