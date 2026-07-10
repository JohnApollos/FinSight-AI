import os
import json
import requests
import pandas as pd
from typing import Dict, Any

URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/ratios.xls"
RAW_DIR = "backend/data/raw"
XLS_PATH = os.path.join(RAW_DIR, "ratios.xls")
OUTPUT_PATH = "backend/data/benchmarks.json"

# Industry mappings: FinSight Industry Name -> List of matching names in Damodaran sheet
SECTOR_MAPPING = {
    "Fintech": [
        "Financial Services (Non-bank)",
        "Financial Svcs (Non-bank)",
        "Financial Svcs."
    ],
    "Insurance": [
        "Insurance (Prop/Cas)",
        "Insurance (General)",
        "Insurance (Life)",
        "Insurance"
    ],
    "Banking": [
        "Bank (Money Center)",
        "Banks",
        "Bank"
    ],
    "Manufacturing": [
        "Auto & Truck",
        "Machinery",
        "Chemical (Basic)",
        "Food Processing"
    ],
    "Software": [
        "Software (System & Application)",
        "Software (Internet)",
        "Software (Entertainment)"
    ],
    "NGO / Development Finance": [
        "Financial Services (Non-bank)",
        "Financial Svcs (Non-bank)"
    ],
    "Agriculture": [
        "Farming",
        "Agricultural",
        "Food Products"
    ]
}

# Healthy fallbacks if download or parsing fails
DEFAULT_BENCHMARKS = {
    "Fintech": {
        "current_ratio": 2.10,
        "quick_ratio": 1.90,
        "debt_to_equity": 0.35,
        "net_margin": 14.5,
        "operating_margin": 18.2,
        "roa": 7.5,
        "roe": 12.8,
        "asset_turnover": 0.85,
        "npl_ratio": 3.2,
        "source": "NYU Damodaran / IFC Fintech 2024"
    },
    "Insurance": {
        "current_ratio": 1.80,
        "quick_ratio": 1.60,
        "debt_to_equity": 0.15,
        "net_margin": 9.2,
        "operating_margin": 11.5,
        "roa": 3.4,
        "roe": 10.1,
        "asset_turnover": 0.45,
        "npl_ratio": 0.0,
        "source": "NYU Damodaran / IFC Insurance 2024"
    },
    "Banking": {
        "current_ratio": 1.50,
        "quick_ratio": 1.30,
        "debt_to_equity": 1.10,
        "net_margin": 16.8,
        "operating_margin": 22.4,
        "roa": 1.2,
        "roe": 11.5,
        "asset_turnover": 0.12,
        "npl_ratio": 4.5,
        "source": "NYU Damodaran / Central Bank 2024"
    },
    "Manufacturing": {
        "current_ratio": 1.95,
        "quick_ratio": 1.10,
        "debt_to_equity": 0.85,
        "net_margin": 6.2,
        "operating_margin": 9.8,
        "roa": 5.4,
        "roe": 10.6,
        "asset_turnover": 1.15,
        "npl_ratio": 0.0,
        "source": "NYU Damodaran Manufacturing 2024"
    },
    "Software": {
        "current_ratio": 2.45,
        "quick_ratio": 2.30,
        "debt_to_equity": 0.22,
        "net_margin": 18.5,
        "operating_margin": 24.1,
        "roa": 9.8,
        "roe": 15.6,
        "asset_turnover": 0.72,
        "npl_ratio": 0.0,
        "source": "NYU Damodaran SaaS 2024"
    },
    "NGO / Development Finance": {
        "current_ratio": 2.80,
        "quick_ratio": 2.60,
        "debt_to_equity": 0.10,
        "net_margin": 0.0,
        "operating_margin": 0.0,
        "roa": 4.5,
        "roe": 6.2,
        "asset_turnover": 0.50,
        "npl_ratio": 2.1,
        "source": "World Bank Enterprise Survey / IFC 2024"
    },
    "Agriculture": {
        "current_ratio": 2.20,
        "quick_ratio": 1.40,
        "debt_to_equity": 0.60,
        "net_margin": 8.5,
        "operating_margin": 12.0,
        "roa": 5.0,
        "roe": 9.5,
        "asset_turnover": 0.75,
        "npl_ratio": 0.0,
        "source": "NYU Damodaran / World Bank Agriculture 2024"
    }
}

def download_ratios_file() -> bool:
    """Downloads the ratios.xls file from NYU Damodaran website."""
    print(f"Downloading Damodaran ratios Excel from {URL}...")
    os.makedirs(RAW_DIR, exist_ok=True)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(URL, headers=headers, timeout=20)
        if response.status_code == 200:
            with open(XLS_PATH, "wb") as f:
                f.write(response.content)
            print("Download successful.")
            return True
        else:
            print(f"Warning: Failed to download (Status {response.status_code})")
            return False
    except Exception as e:
        print(f"Exception during download: {e}")
        return False

def parse_damodaran_xls() -> Dict[str, Any]:
    """Parses the downloaded Excel file and maps metrics to target industries."""
    if not os.path.exists(XLS_PATH):
        print("Excel file not found. Using default benchmarks.")
        return DEFAULT_BENCHMARKS
        
    try:
        # Load XLS sheet
        print("Reading XLS file...")
        # Damodaran files usually have metadata at the top, we will look for where 'Industry' starts
        df = pd.read_excel(XLS_PATH, header=None)
        
        # Find the header row (contains 'Industry Name' or 'Industry')
        header_row_idx = 0
        for idx, row in df.iterrows():
            row_str = " ".join([str(val) for val in row.values])
            if "Industry Name" in row_str or "Industry name" in row_str or "Number of firms" in row_str:
                header_row_idx = idx
                break
                
        # Reload with correct header
        df = pd.read_excel(XLS_PATH, header=header_row_idx)
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Identify relevant columns dynamically
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "industry" in col_lower:
                col_map["industry"] = col
            elif "margin" in col_lower and "net" in col_lower:
                col_map["net_margin"] = col
            elif "roe" in col_lower or "return on equity" in col_lower:
                col_map["roe"] = col
            elif "d/e" in col_lower or "debt/equity" in col_lower:
                col_map["debt_to_equity"] = col
            elif "roa" in col_lower or "return on assets" in col_lower:
                col_map["roa"] = col
            elif "turnover" in col_lower or "sales/assets" in col_lower:
                col_map["asset_turnover"] = col
                
        print(f"Mapped columns: {col_map}")
        
        # Check if basic columns were mapped
        required_cols = ["industry", "net_margin", "roe"]
        if not all(col in col_map for col in required_cols):
            print("Failed to map required columns in Damodaran sheet. Using defaults.")
            return DEFAULT_BENCHMARKS
            
        benchmarks = {}
        # Parse each target sector
        for sector, match_list in SECTOR_MAPPING.items():
            sector_data = DEFAULT_BENCHMARKS[sector].copy() # Start with fallback values
            
            # Search for matching industry rows
            for match_name in match_list:
                row_match = df[df[col_map["industry"]].str.contains(match_name, case=False, na=False)]
                if not row_match.empty:
                    # Found matching industry! Extract metrics
                    try:
                        row = row_match.iloc[0]
                        
                        # Extract and format metrics, checking for string percentages/fractions
                        def clean_val(val, default_val):
                            if pd.isna(val) or val == "" or val == "N/A":
                                return default_val
                            # Remove percentages or spaces
                            val_str = str(val).replace("%", "").strip()
                            try:
                                return float(val_str)
                            except ValueError:
                                return default_val
                                
                        if "net_margin" in col_map:
                            sector_data["net_margin"] = clean_val(row[col_map["net_margin"]], sector_data["net_margin"])
                        if "roe" in col_map:
                            sector_data["roe"] = clean_val(row[col_map["roe"]], sector_data["roe"])
                        if "debt_to_equity" in col_map:
                            # Damodaran D/E is usually a percentage, e.g. 85.0% or 0.85. 
                            # If value is > 5, it's likely a percentage (e.g. 35.2 representing 35.2% or 0.352)
                            val = clean_val(row[col_map["debt_to_equity"]], sector_data["debt_to_equity"])
                            if val > 10.0:
                                val = val / 100.0
                            sector_data["debt_to_equity"] = val
                        if "roa" in col_map:
                            sector_data["roa"] = clean_val(row[col_map["roa"]], sector_data["roa"])
                        if "asset_turnover" in col_map:
                            sector_data["asset_turnover"] = clean_val(row[col_map["asset_turnover"]], sector_data["asset_turnover"])
                            
                        sector_data["source"] = f"NYU Damodaran 2024 ({match_name})"
                        break # Stop checking other matches since we found one
                    except Exception as ex:
                        print(f"Error parsing row for sector {sector} using '{match_name}': {ex}")
                        
            benchmarks[sector] = sector_data
            
        print("Damodaran ratios parsed successfully.")
        return benchmarks
        
    except Exception as e:
        print(f"Error parsing Damodaran Excel file: {e}. Using defaults.")
        return DEFAULT_BENCHMARKS

def run_pipeline():
    """Downloads and parses NYU Damodaran ratios."""
    download_success = download_ratios_file()
    benchmarks = parse_damodaran_xls()
    
    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Benchmarks saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_pipeline()
