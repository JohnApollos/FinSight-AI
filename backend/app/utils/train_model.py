import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple

DATA_PATH = "backend/data/sec_reference_data.csv"
MODEL_DIR = "backend/models"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")

# Ratios used as features in the Isolation Forest
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

def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the 10 core financial ratios from absolute statements using identical math to analysis.py."""
    df_ratios = pd.DataFrame(index=df.index)
    
    assets = df["assets"].clip(lower=1.0)
    liabilities = df["liabilities"].clip(lower=0.0)
    equity = df["equity"].clip(lower=1.0)
    revenue = df["revenue"].clip(lower=1.0)
    net_income = df["net_income"]
    working_capital = df.get("working_capital", pd.Series(0.0, index=df.index))
    cash = df.get("cash", pd.Series(0.0, index=df.index))
    receivables = df.get("accounts_receivable", pd.Series(0.0, index=df.index))
    inventory = df.get("inventory", pd.Series(0.0, index=df.index))
    interest_expense = df.get("interest_expense", pd.Series(0.0, index=df.index))
    
    # Calculate proxy current assets
    current_assets = cash + receivables + inventory
    # For rows where current_assets is 0.0, default proxy current assets to 30% of total assets
    is_zero_ca = current_assets == 0.0
    current_assets = current_assets.mask(is_zero_ca, assets * 0.30)
    
    # Calculate current liabilities
    current_liabilities = current_assets - working_capital
    # For rows where current_liabilities <= 0, proxy them as 40% of total liabilities
    is_le_zero_cl = current_liabilities <= 0.0
    cl_proxy = liabilities * 0.40
    cl_proxy = cl_proxy.clip(lower=1.0)
    current_liabilities = current_liabilities.mask(is_le_zero_cl, cl_proxy)
    
    # Recalculate current assets for the proxy rows
    ca_proxy = working_capital + current_liabilities
    current_assets = current_assets.mask(is_le_zero_cl, ca_proxy)
    
    # Liquidity Ratios
    df_ratios["current_ratio"] = (current_assets / current_liabilities).clip(lower=0.1, upper=10.0)
    df_ratios["quick_ratio"] = ((current_assets - inventory) / current_liabilities).clip(lower=0.1, upper=10.0)
    df_ratios["cash_ratio"] = (cash / current_liabilities).clip(lower=0.05, upper=10.0)
    
    # Profitability Ratios
    df_ratios["roa"] = net_income / assets
    df_ratios["roe"] = net_income / equity
    df_ratios["net_margin"] = net_income / revenue
    
    # EBIT proxy
    ebit = net_income + interest_expense
    ebit = ebit.mask(ebit == 0.0, net_income * 1.30)
    df_ratios["operating_margin"] = ebit / revenue
    
    # Leverage Ratios
    df_ratios["debt_to_equity"] = liabilities / equity
    df_ratios["debt_to_assets"] = liabilities / assets
    
    # Efficiency Ratio
    df_ratios["asset_turnover"] = revenue / assets
    
    # Clean inf/nan values
    df_ratios = df_ratios.replace([np.inf, -np.inf], np.nan)
    df_ratios = df_ratios.fillna(df_ratios.median())
    
    return df_ratios

def generate_synthetic_data(num_samples: int = 500) -> pd.DataFrame:
    """Generates a high-fidelity synthetic baseline dataset of healthy companies."""
    print("Generating synthetic reference data of healthy companies...")
    np.random.seed(42)
    
    # Sectors: Fintech (1), Insurance (2), Banking (3), Manufacturing (4), Software (5)
    sectors = ["6022", "6141", "6159", "6311", "6321", "6411", "7372"]
    data = []
    
    for i in range(num_samples):
        sic = np.random.choice(sectors)
        year = np.random.choice([2022, 2023, 2024])
        
        # Scale of assets
        assets = np.exp(np.random.normal(15.0, 1.5)) # Millions to Billions
        
        # Sector-specific profiles (assets distribution, profitability)
        if sic in {"6022", "6159"}: # Banking/Credit
            debt_ratio = np.random.normal(0.88, 0.03) # High leverage
            margin = np.random.normal(0.15, 0.03)
            turnover = np.random.normal(0.08, 0.02)
        elif sic in {"6311", "6321", "6411"}: # Insurance
            debt_ratio = np.random.normal(0.80, 0.05)
            margin = np.random.normal(0.08, 0.02)
            turnover = np.random.normal(0.20, 0.05)
        elif sic == "7372": # Software
            debt_ratio = np.random.normal(0.20, 0.08) # Low leverage
            margin = np.random.normal(0.18, 0.05) # High margin
            turnover = np.random.normal(0.65, 0.15)
        else: # Consumer credit/finance (6141)
            debt_ratio = np.random.normal(0.50, 0.10)
            margin = np.random.normal(0.12, 0.04)
            turnover = np.random.normal(0.40, 0.10)
            
        liabilities = assets * debt_ratio
        equity = assets - liabilities
        revenue = assets * turnover
        net_income = revenue * margin
        retained_earnings = equity * np.random.uniform(0.1, 0.5)
        
        # Working capital & asset mix variables
        if sic in {"6022", "6159", "6311", "6321"}:
            working_capital = 0.0 # Financial institutions do not report current items
            cash = assets * np.random.uniform(0.05, 0.20)
            inventory = 0.0
            accounts_receivable = assets * np.random.uniform(0.10, 0.40)
        else:
            working_capital = assets * np.random.normal(0.25, 0.08)
            cash = assets * np.random.uniform(0.05, 0.15)
            inventory = assets * np.random.uniform(0.10, 0.25) if sic != "7372" else 0.0
            accounts_receivable = assets * np.random.uniform(0.10, 0.25)
            
        data.append({
            "cik": f"{i:010d}",
            "company_name": f"Synthetic Company {i}",
            "sic": sic,
            "year": year,
            "revenue": max(1000.0, revenue),
            "net_income": net_income,
            "assets": assets,
            "liabilities": liabilities,
            "equity": equity,
            "retained_earnings": retained_earnings,
            "working_capital": working_capital,
            "cash": cash,
            "inventory": inventory,
            "accounts_receivable": accounts_receivable
        })
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved synthetic reference dataset to {DATA_PATH}")
    return df

def train_isolation_forest():
    """Trains the Isolation Forest model on the reference dataset."""
    print("Initializing model training pipeline...")
    
    # Load dataset
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) < 100:
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(DATA_PATH)
        if len(df) < 10:
            df = generate_synthetic_data()
            
    print(f"Loaded {len(df)} records for training.")
    
    # Compute Ratios
    df_ratios = calculate_ratios(df)
    X = df_ratios[FEATURES]
    
    # Train Isolation Forest
    # Contamination defines the proportion of outliers we expect in the training data (e.g. 5%)
    model = IsolationForest(
        n_estimators=150, 
        contamination=0.05, 
        random_state=42
    )
    model.fit(X)
    
    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Isolation Forest model trained and saved successfully to {MODEL_PATH}")
    
    # Quick sanity check
    test_scores = model.score_samples(X)
    print(f"Model sanity check: Mean score = {test_scores.mean():.4f}, Min score = {test_scores.min():.4f}")

if __name__ == "__main__":
    train_isolation_forest()
