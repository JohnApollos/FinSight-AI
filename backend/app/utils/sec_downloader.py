import os
import time
import json
import csv
import requests
from typing import Dict, List, Optional, Set, Tuple

# Configuration
USER_AGENT = "FinSightAI-DevUser student_dev@finsightai.local"
HEADERS = {"User-Agent": USER_AGENT}
TARGET_SICS = {"6022", "6141", "6159", "6311", "6321", "6411", "7372"}
OUTPUT_PATH = "backend/data/sec_reference_data.csv"

# XBRL Tag Maps (ordered by preference)
TAG_MAPS = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueGoodsNet",
        "OperatingRevenues",
        "InterestAndDividendIncomeSecurities", # Bank specific
        "InterestIncomeNet"
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic"
    ],
    "assets": [
        "Assets"
    ],
    "liabilities": [
        "Liabilities"
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
        "CommonStockholdersEquity"
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit"
    ],
    "current_assets": [
        "AssetsCurrent"
    ],
    "current_liabilities": [
        "LiabilitiesCurrent"
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalents",
        "Cash"
    ],
    "inventory": [
        "InventoryNet",
        "Inventories",
        "Inventory"
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsAndNotesReceivableNet",
        "Receivables"
    ]
}

def get_companies_from_sec() -> List[Tuple[str, str, str]]:
    """
    Queries SEC EFTS API using keywords to discover unique companies matching target SIC codes.
    Returns a list of tuples: (company_name, CIK, SIC)
    """
    print("Searching SEC EDGAR for target companies...")
    url = "https://efts.sec.gov/LATEST/search-index"
    keywords = ["software", "bank", "insurance", "credit", "finance", "capital", "lending"]
    unique_companies: Dict[str, Tuple[str, str, str]] = {} # CIK -> (Name, CIK, SIC)
    
    for kw in keywords:
        params = {
            "q": f'"{kw}"',
            "forms": "10-K",
            "size": 100
        }
        try:
            time.sleep(0.1) # Respect rate limit
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if response.status_code == 200:
                hits = response.json().get("hits", {}).get("hits", [])
                for hit in hits:
                    source = hit.get("_source", {})
                    sics = source.get("sics", [])
                    ciks = source.get("ciks", [])
                    names = source.get("display_names", [])
                    if sics and ciks and names:
                        sic = sics[0]
                        cik = ciks[0]
                        name = names[0].split("  (")[0] # Extract clean name
                        if sic in TARGET_SICS:
                            unique_companies[cik] = (name, cik, sic)
            else:
                print(f"Warning: EFTS search failed for '{kw}' with status {response.status_code}")
        except Exception as e:
            print(f"Exception during SEC search for '{kw}': {e}")
            
    # Group by SIC to see distributions
    by_sic: Dict[str, int] = {sic: 0 for sic in TARGET_SICS}
    results = list(unique_companies.values())
    for _, _, sic in results:
        by_sic[sic] += 1
        
    print(f"Discovered {len(results)} unique companies matching target SIC codes.")
    for sic, count in by_sic.items():
        print(f" - SIC {sic}: {count} companies")
        
    return results

def extract_fact_value(facts: dict, tag_list: List[str], year: int) -> Optional[float]:
    """
    Searches the XBRL facts structure for a matching tag and year.
    Checks 'us-gaap' first.
    """
    for namespace in ["us-gaap", "ifrs"]:
        ns_facts = facts.get(namespace, {})
        for tag in tag_list:
            tag_data = ns_facts.get(tag, {})
            units = tag_data.get("units", {})
            for unit_key in ["USD", "shares"]: # Check USD currency values mainly
                unit_list = units.get(unit_key, [])
                # Filter for 10-K, FY, and matching year
                matching_facts = []
                for f in unit_list:
                    if f.get("form") == "10-K" and f.get("fp") == "FY" and f.get("fy") == year:
                        matching_facts.append(f)
                
                if matching_facts:
                    # Select the latest filed fact
                    matching_facts.sort(key=lambda x: x.get("filed", ""), reverse=True)
                    return float(matching_facts[0]["val"])
    return None

def fetch_company_facts(cik: str) -> Optional[dict]:
    """
    Fetches the XBRL facts for a given CIK from the SEC API.
    """
    # CIK must be padded to 10 digits
    padded_cik = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json"
    try:
        time.sleep(0.1) # SEC limit of 10 requests per second
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Some smaller foreign filers may not have XBRL data
            return None
        else:
            print(f"Warning: Failed to fetch facts for CIK {cik} (Status: {response.status_code})")
    except Exception as e:
        print(f"Exception fetching facts for CIK {cik}: {e}")
    return None

def download_and_parse_sec_data(target_count_per_sic: int = 45):
    """
    Orchestrates the discovery and parsing of SEC financial statement data.
    """
    companies = get_companies_from_sec()
    if not companies:
        print("No companies discovered. Aborting download.")
        return

    # Create directories if missing
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    csv_columns = [
        "cik", "company_name", "sic", "year", 
        "revenue", "net_income", "assets", "liabilities", 
        "equity", "retained_earnings", "working_capital",
        "cash", "inventory", "accounts_receivable"
    ]
    
    records_written = 0
    unique_ciks_by_sic: Dict[str, Set[str]] = {sic: set() for sic in TARGET_SICS}
    
    with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        
        for name, cik, sic in companies:
            # Check if we already have enough companies for this SIC code
            if len(unique_ciks_by_sic[sic]) >= target_count_per_sic:
                continue
                
            print(f"Processing CIK {cik} ({name}) in SIC {sic}...")
            facts_data = fetch_company_facts(cik)
            if not facts_data or "facts" not in facts_data:
                continue
                
            facts = facts_data["facts"]
            
            # Check years 2022, 2023, 2024
            company_has_data = False
            for year in [2022, 2023, 2024]:
                assets = extract_fact_value(facts, TAG_MAPS["assets"], year)
                liabilities = extract_fact_value(facts, TAG_MAPS["liabilities"], year)
                equity = extract_fact_value(facts, TAG_MAPS["equity"], year)
                revenue = extract_fact_value(facts, TAG_MAPS["revenue"], year)
                net_income = extract_fact_value(facts, TAG_MAPS["net_income"], year)
                retained_earnings = extract_fact_value(facts, TAG_MAPS["retained_earnings"], year)
                
                # Check current assets and current liabilities for Working Capital
                current_assets = extract_fact_value(facts, TAG_MAPS["current_assets"], year)
                current_liabilities = extract_fact_value(facts, TAG_MAPS["current_liabilities"], year)
                
                # Check cash, inventory, receivables
                cash = extract_fact_value(facts, TAG_MAPS["cash"], year)
                inventory = extract_fact_value(facts, TAG_MAPS["inventory"], year)
                accounts_receivable = extract_fact_value(facts, TAG_MAPS["accounts_receivable"], year)
                
                # If total assets is missing, we cannot calculate ratios
                if assets is None:
                    continue
                    
                # Standardize missing values
                net_income = net_income if net_income is not None else 0.0
                revenue = revenue if revenue is not None else 0.0
                liabilities = liabilities if liabilities is not None else 0.0
                cash = cash if cash is not None else 0.0
                inventory = inventory if inventory is not None else 0.0
                accounts_receivable = accounts_receivable if accounts_receivable is not None else 0.0
                
                # If equity is missing, calculate as Assets - Liabilities
                if equity is None:
                    equity = assets - liabilities
                if retained_earnings is None:
                    retained_earnings = 0.0 # Default to 0 if not reported
                    
                # Working capital calculation
                if current_assets is not None and current_liabilities is not None:
                    working_capital = current_assets - current_liabilities
                else:
                    # For financial institutions/banks/insurance, current assets are not defined.
                    # Working capital is set to 0.0 to match financial modeling standards.
                    working_capital = 0.0
                    
                record = {
                    "cik": cik,
                    "company_name": name,
                    "sic": sic,
                    "year": year,
                    "revenue": revenue,
                    "net_income": net_income,
                    "assets": assets,
                    "liabilities": liabilities,
                    "equity": equity,
                    "retained_earnings": retained_earnings,
                    "working_capital": working_capital,
                    "cash": cash,
                    "inventory": inventory,
                    "accounts_receivable": accounts_receivable
                }
                
                writer.writerow(record)
                company_has_data = True
                records_written += 1
                
            if company_has_data:
                unique_ciks_by_sic[sic].add(cik)
                
            # Print intermediate progress
            total_companies_saved = sum(len(s) for s in unique_ciks_by_sic.values())
            if total_companies_saved >= 300:
                print("Reached target of 300 companies. Stopping download.")
                break

    print(f"SEC reference database creation complete. Written {records_written} annual records for {total_companies_saved} unique companies.")

if __name__ == "__main__":
    download_and_parse_sec_data()
