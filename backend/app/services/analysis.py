import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Prevent GUI window popups in headless environments
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Tuple

MODEL_PATH = "backend/models/isolation_forest.pkl"

# Ratios list
FEATURES = [
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "roa",
    "roe",
    "net_margin",
    "operating_margin",
    "debt_to_equity",
    "debt_to_assets",
    "asset_turnover"
]

FEATURE_LABELS = {
    "current_ratio": "Current Ratio",
    "quick_ratio": "Quick Ratio",
    "cash_ratio": "Cash Ratio",
    "roa": "Return on Assets (ROA)",
    "roe": "Return on Equity (ROE)",
    "net_margin": "Net Profit Margin",
    "operating_margin": "Operating Margin",
    "debt_to_equity": "Debt-to-Equity",
    "debt_to_assets": "Debt-to-Assets",
    "asset_turnover": "Asset Turnover"
}

def compute_ratios_from_statement(stmt: Dict[str, Any]) -> Dict[str, float]:
    """Computes key financial ratios from absolute figures."""
    assets = max(1.0, stmt.get("assets", 0.0))
    liabilities = max(0.0, stmt.get("liabilities", 0.0))
    equity = max(1.0, stmt.get("equity", 0.0))
    revenue = max(1.0, stmt.get("revenue", 0.0))
    net_income = stmt.get("net_income", 0.0)
    working_capital = stmt.get("working_capital", 0.0)
    cash = stmt.get("cash", 0.0)
    receivables = stmt.get("accounts_receivable", 0.0)
    inventory = stmt.get("inventory", 0.0)
    interest_expense = stmt.get("interest_expense", 0.0)
    
    # Establish proxy current items if missing (for non-financials)
    current_assets = cash + receivables + inventory
    if current_assets == 0.0:
        # Default proxy current assets to 30% of total assets
        current_assets = assets * 0.30
        
    current_liabilities = current_assets - working_capital
    if current_liabilities <= 0.0:
        current_liabilities = max(1.0, liabilities * 0.40) # Proxy current liabilities
        current_assets = working_capital + current_liabilities
        
    # Liquidity
    current_ratio = np.clip(current_assets / current_liabilities, 0.1, 10.0)
    quick_ratio = np.clip((current_assets - inventory) / current_liabilities, 0.1, 10.0)
    cash_ratio = np.clip(cash / current_liabilities, 0.05, 10.0)
    
    # Profitability
    roa = net_income / assets
    roe = net_income / equity
    net_margin = net_income / revenue
    
    # EBIT proxy
    ebit = net_income + interest_expense
    if ebit == 0.0:
        ebit = net_income * 1.30
    operating_margin = ebit / revenue
    
    # Leverage
    debt_to_equity = liabilities / equity
    debt_to_assets = liabilities / assets
    interest_coverage = ebit / max(1.0, interest_expense) if interest_expense > 0 else 100.0
    
    # Efficiency
    asset_turnover = revenue / assets
    receivables_turnover = revenue / max(1.0, receivables) if receivables > 0 else 10.0
    
    # Altman Z-Score Calculations
    # X1 = Working Capital / Total Assets
    x1 = working_capital / assets
    # X2 = Retained Earnings / Total Assets
    x2 = stmt.get("retained_earnings", 0.0) / assets
    # X3 = EBIT / Total Assets
    x3 = ebit / assets
    # X4 = Book Value of Equity / Total Liabilities
    x4 = equity / max(1.0, liabilities)
    # X5 = Revenue / Total Assets
    x5 = revenue / assets
    
    # Altman Classic (Manufacturing - Z-Score)
    z_classic = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
    # Altman modified (Service/Non-Manufacturing - Z''-Score)
    z_service = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    
    return {
        "current_ratio": float(current_ratio),
        "quick_ratio": float(quick_ratio),
        "cash_ratio": float(cash_ratio),
        "roa": float(roa),
        "roe": float(roe),
        "net_margin": float(net_margin),
        "operating_margin": float(operating_margin),
        "debt_to_equity": float(debt_to_equity),
        "debt_to_assets": float(debt_to_assets),
        "interest_coverage": float(interest_coverage),
        "asset_turnover": float(asset_turnover),
        "receivables_turnover": float(receivables_turnover),
        "altman_z_classic": float(z_classic),
        "altman_z_service": float(z_service)
    }

def analyze_anomaly(ratios: Dict[str, float], identifier: str = "") -> Tuple[float, bool, List[Dict[str, Any]], str]:
    """
    Evaluates computed ratios against the trained Isolation Forest.
    Uses SHAP to explain the anomaly score.
    Returns: (anomaly_score, is_anomaly, list of contributing drivers, shap_chart_path)
    """
    # Create input vector
    X = pd.DataFrame([{f: ratios[f] for f in FEATURES}])
    
    # Default outputs if model loading fails
    anomaly_score = 0.0
    is_anomaly = False
    drivers = []
    chart_path = ""
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Isolation Forest model not found at {MODEL_PATH}. Anomaly detection will be skipped.")
        return anomaly_score, is_anomaly, drivers, chart_path

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            
        # Predict: 1 = normal, -1 = anomaly
        pred = model.predict(X)[0]
        is_anomaly = bool(pred == -1)
        
        # Raw score: lower means more anomalous (in scikit-learn, decision_function returns signed scores)
        raw_score = model.decision_function(X)[0]
        # Normalize score to an intuitive 0-100 scale (where >50% means anomalous)
        anomaly_score = float(np.clip((0.2 - raw_score) / 0.4, 0.0, 1.0) * 100.0)
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # SHAP outputs values for each class or single decision output.
        # IsolationForest output in TreeExplainer is shape (1, num_features) or (num_features,)
        if len(shap_values.shape) > 1:
            shap_row = shap_values[0]
        else:
            shap_row = shap_values
            
        # Compile drivers (outlier-contributing features)
        # Negative SHAP values reduce the decision function output, pushing it towards an anomaly.
        for idx, f_name in enumerate(FEATURES):
            val = float(X.iloc[0][f_name])
            shap_val = float(shap_row[idx])
            drivers.append({
                "feature": f_name,
                "label": FEATURE_LABELS[f_name],
                "value": val,
                "shap_value": shap_val
            })
            
        # Sort drivers: most negative first (outlier drivers)
        drivers.sort(key=lambda x: x["shap_value"])
        
        # Generate SHAP Matplotlib chart
        if identifier:
            chart_path = f"backend/data/temp/shap_waterfall_{identifier}.png"
        else:
            chart_path = "backend/data/shap_waterfall.png"
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        
        plt.figure(figsize=(6, 4))
        # Select top 5 drivers for display
        display_drivers = sorted(drivers, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
        display_drivers.reverse() # Plot highest impact at top
        
        labels = [d["label"] for d in display_drivers]
        values = [d["shap_value"] for d in display_drivers]
        
        # Color: red for anomaly-driving (negative), blue for normal-driving (positive)
        colors = ["#e74c3c" if v < 0 else "#3498db" for v in values]
        
        plt.barh(labels, values, color=colors, height=0.6)
        plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.title("SHAP Feature Contributions to Anomaly Score", fontsize=10, fontweight="bold", pad=12)
        plt.xlabel("SHAP Impact Value", fontsize=8)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Exception during anomaly / SHAP analysis: {e}")
        
    return anomaly_score, is_anomaly, drivers, chart_path

def evaluate_financial_health(stmt: Dict[str, Any], identifier: str = "") -> Dict[str, Any]:
    """Runs the complete solvency analysis and anomaly engine."""
    ratios = compute_ratios_from_statement(stmt)
    anomaly_score, is_anomaly, drivers, chart_path = analyze_anomaly(ratios, identifier)
    
    # Traffic light scoring logic for Altman Z-Score
    sic = str(stmt.get("sic", ""))
    
    # Select default Altman Z model based on industry
    # Banking (6022), credit agencies (6159), personal credit (6141), insurance (6311, 6321, 6411), software (7372)
    # Banks, insurance, software, and credit institutions use the Service Z'' model.
    # Manufacturing uses the Classic Z model.
    is_service = sic != "Manufacturing" and sic not in {"Manufacturing", "Auto", "Cement"}
    
    z_score = ratios["altman_z_service"] if is_service else ratios["altman_z_classic"]
    
    # Solvency zones:
    # Service model (Z''): Safe > 2.90, Grey 1.23 - 2.90, Distress < 1.23
    # Classic model (Z): Safe > 2.90, Grey 1.23 - 2.90 (private model thresholds)
    if z_score >= 2.90:
        zone = "Safe (Low Risk)"
        status_color = "green"
    elif z_score >= 1.23:
        zone = "Grey Zone (Moderate Risk)"
        status_color = "amber"
    else:
        zone = "Distress Zone (High Risk)"
        status_color = "red"
        
    return {
        "ratios": ratios,
        "anomaly": {
            "score": anomaly_score,
            "is_anomaly": is_anomaly,
            "drivers": drivers[:3], # Top 3 main anomaly drivers
            "chart_path": chart_path
        },
        "solvency": {
            "score": z_score,
            "zone": zone,
            "status_color": status_color,
            "model_used": "Altman Z'' Service Model" if is_service else "Altman Z' Private Manufacturing Model"
        }
    }

if __name__ == "__main__":
    # Test script offline
    test_stmt = {
        "revenue": 5000000.0,
        "expenses": 4200000.0,
        "net_income": 800000.0,
        "assets": 10000000.0,
        "liabilities": 4000000.0,
        "equity": 6000000.0,
        "retained_earnings": 1500000.0,
        "working_capital": 2000000.0,
        "sic": "7372"
    }
    analysis = evaluate_financial_health(test_stmt)
    import json
    print(json.dumps(analysis, indent=2))
