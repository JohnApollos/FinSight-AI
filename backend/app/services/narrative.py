import os
import json
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

BENCHMARKS_PATH = "backend/data/benchmarks.json"
CHROMA_PATH = "backend/data/chroma"

# Seed benchmarks text description for ChromaDB
def get_sector_profile_text(sector: str, data: Dict[str, Any]) -> str:
    """Formats sector benchmark data into a descriptive text profile."""
    return (
        f"Sector Profile: {sector}.\n"
        f"Industry financial benchmarks are as follows:\n"
        f"- Current Ratio (Liquidity): {data.get('current_ratio', 1.5):.2f}\n"
        f"- Quick Ratio (Liquidity): {data.get('quick_ratio', 1.2):.2f}\n"
        f"- Cash Ratio (Liquidity): {data.get('cash_ratio', 0.5):.2f}\n"
        f"- Debt-to-Equity (Leverage): {data.get('debt_to_equity', 0.8):.2f}\n"
        f"- Debt-to-Assets (Leverage): {data.get('debt_to_assets', 0.5):.2f}\n"
        f"- Return on Assets (ROA, Profitability): {data.get('roa', 0.05)*100:.1f}%\n"
        f"- Return on Equity (ROE, Profitability): {data.get('roe', 0.10)*100:.1f}%\n"
        f"- Net Profit Margin (Profitability): {data.get('net_margin', 8.0):.1f}%\n"
        f"- Operating Margin (Profitability): {data.get('operating_margin', 12.0):.1f}%\n"
        f"- Asset Turnover (Efficiency): {data.get('asset_turnover', 0.8):.2f}\n"
        f"Non-Performing Loans (NPL) sector threshold: {data.get('npl_ratio', 4.0):.1f}%\n"
        f"Data Source: {data.get('source', 'IFC / World Bank / NYU Damodaran')}\n"
    )

def retrieve_sector_benchmark(sector: str) -> str:
    """
    Attempts to retrieve the sector benchmark profile using local ChromaDB.
    Falls back to direct JSON file parsing if ChromaDB is unavailable or error-prone.
    """
    # Check if benchmarks.json exists
    if not os.path.exists(BENCHMARKS_PATH):
        print(f"Warning: benchmarks.json not found at {BENCHMARKS_PATH}. Using Fintech defaults.")
        from damodaran_downloader import DEFAULT_BENCHMARKS
        benchmarks_data = DEFAULT_BENCHMARKS
    else:
        with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
            benchmarks_data = json.load(f)
            
    # Normalize sector key
    matched_sector = "Fintech"
    for s in benchmarks_data.keys():
        if s.lower() in sector.lower() or sector.lower() in s.lower():
            matched_sector = s
            break
            
    sector_info = benchmarks_data.get(matched_sector, benchmarks_data["Fintech"])
    profile_text = get_sector_profile_text(matched_sector, sector_info)
    
    # Try using ChromaDB for semantic retrieval matching to demonstrate agentic architecture
    try:
        import chromadb
        # Initalize persistent client
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection("sector_benchmarks")
        
        # Populate if empty
        if collection.count() == 0:
            print("Seeding local ChromaDB with sector benchmarks...")
            documents = []
            ids = []
            metadatas = []
            for s, data in benchmarks_data.items():
                txt = get_sector_profile_text(s, data)
                documents.append(txt)
                ids.append(s)
                metadatas.append({"sector": s})
            collection.add(documents=documents, ids=ids, metadatas=metadatas)
            
        # Query ChromaDB using the requested sector name
        results = collection.query(query_texts=[sector], n_results=1)
        if results and results["documents"] and results["documents"][0]:
            print(f"Retrieved benchmark for '{sector}' from ChromaDB.")
            return results["documents"][0][0]
            
    except Exception as e:
        print(f"ChromaDB retrieval skipped ({e}). Falling back to JSON parser.")
        
    print(f"Retrieved benchmark for '{matched_sector}' via JSON fallback.")
    return profile_text

def generate_offline_narrative(company_name: str, sector: str, ratios: Dict[str, float], solvency: Dict[str, Any], anomaly: Dict[str, Any]) -> str:
    """
    Generates a high-fidelity plain-English narrative using programmatic templates (Offline Fallback).
    """
    z_score = solvency.get("score", 0.0)
    z_zone = solvency.get("zone", "Unknown")
    is_anomaly = anomaly.get("is_anomaly", False)
    anomaly_score = anomaly.get("score", 0.0)
    
    # Load benchmarks to compare
    try:
        with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
            benchmarks_data = json.load(f)
    except Exception:
        from damodaran_downloader import DEFAULT_BENCHMARKS
        benchmarks_data = DEFAULT_BENCHMARKS
        
    matched_sector = "Fintech"
    for s in benchmarks_data.keys():
        if s.lower() in sector.lower() or sector.lower() in s.lower():
            matched_sector = s
            break
    bench = benchmarks_data[matched_sector]
    
    # Evaluate Strengths and Concerns
    strengths = []
    concerns = []
    
    # Liquidity
    curr_ratio = ratios.get("current_ratio", 1.0)
    bench_curr = bench.get("current_ratio", 1.5)
    if curr_ratio >= bench_curr:
        strengths.append(f"Strong short-term liquidity: Current Ratio is {curr_ratio:.2f}, outperforming the sector benchmark of {bench_curr:.2f}.")
    else:
        concerns.append(f"Liquidity pressure: Current Ratio is {curr_ratio:.2f}, below the sector median of {bench_curr:.2f}. This indicates tight working capital.")
        
    # Profitability
    net_margin = ratios.get("net_margin", 0.0) * 100
    bench_margin = bench.get("net_margin", 10.0)
    if net_margin >= bench_margin:
        strengths.append(f"Superior profitability: Net profit margin of {net_margin:.1f}% exceeds the industry average of {bench_margin:.1f}%.")
    elif net_margin < 0:
        concerns.append(f"Operating losses: The company is unprofitable with a negative net profit margin of {net_margin:.1f}%.")
    else:
        concerns.append(f"Below-average margins: Net profit margin of {net_margin:.1f}% is lower than the sector median of {bench_margin:.1f}%.")
        
    # Solvency
    roe = ratios.get("roe", 0.0) * 100
    bench_roe = bench.get("roe", 12.0)
    if roe >= bench_roe:
        strengths.append(f"Strong shareholder returns: ROE stands at {roe:.1f}%, beating the sector baseline of {bench_roe:.1f}%.")
        
    debt_equity = ratios.get("debt_to_equity", 0.0)
    bench_debt = bench.get("debt_to_equity", 0.8)
    if debt_equity <= bench_debt:
        strengths.append(f"Conservative capital structure: Debt-to-Equity is {debt_equity:.2f}, indicating low leverage compared to the sector average of {bench_debt:.2f}.")
    else:
        concerns.append(f"Elevated leverage: Debt-to-Equity is {debt_equity:.2f}, which is higher than the sector median of {bench_debt:.2f}. This increases financial risk.")

    if "distress" in z_zone.lower():
        concerns.append(f"Critical Solvency Risk: The Altman Z-Score is {z_score:.2f}, placing the firm in the Distress Zone. This suggests high default risk.")
    elif "safe" in z_zone.lower():
        strengths.append(f"Stable long-term solvency: The Altman Z-Score is {z_score:.2f}, positioning the firm in the Safe Zone.")
        
    if is_anomaly:
        concerns.append(f"Statistical Anomaly Detected: The company's financial profile was flagged as an outlier (Score: {anomaly_score:.1f}%) compared to standard sector baselines.")
        
    # Construct Narrative Sections
    exec_summary = (
        f"FinSight AI completed a comprehensive financial risk evaluation for {company_name} in the {sector} sector. "
        f"The company has an Altman Z-Score of {z_score:.2f} ({z_zone}) using the {solvency.get('model_used')}. "
        f"Statistical anomaly analysis scored the company at {anomaly_score:.1f}%, indicating a "
        f"{'highly anomalous' if is_anomaly else 'standard'} operational signature."
    )
    
    strengths_text = "\n".join([f"- {s}" for s in strengths]) if strengths else "- Operating within standard bounds."
    concerns_text = "\n".join([f"- {c}" for c in concerns]) if concerns else "- No major concerns detected."
    
    recommendation = ""
    if "distress" in z_zone.lower():
        recommendation = (
            "RECOMMENDATION: High credit risk. Standard lending or credit extension is not advised. "
            "Suggest restructuring liabilities, securing additional equity funding, and monitoring cash flows closely. "
            "Any current credit facility should be heavily collateralized."
        )
    elif "grey" in z_zone.lower():
        recommendation = (
            "RECOMMENDATION: Moderate credit risk. Credit facilities can be approved with covenants, "
            "such as quarterly reporting reviews and maintaining a minimum current ratio of 1.2. "
            "Monitor operating margins and debt repayments closely."
        )
    else:
        recommendation = (
            "RECOMMENDATION: Low credit risk. Standard credit lines can be approved. "
            "The company shows healthy liquidity, strong profit potential, and a safe solvency buffer."
        )
        
    narrative = (
        f"### EXECUTIVE SUMMARY\n{exec_summary}\n\n"
        f"### KEY STRENGTHS\n{strengths_text}\n\n"
        f"### KEY CONCERNS\n{concerns_text}\n\n"
        f"### RECOMMENDATION\n{recommendation}"
    )
    
    return narrative

def generate_risk_narrative(
    company_name: str, 
    sector: str, 
    ratios: Dict[str, float], 
    solvency: Dict[str, Any], 
    anomaly: Dict[str, Any], 
    gemini_api_key: Optional[str] = None
) -> str:
    """
    Generates a financial risk narrative by querying the benchmarks vector store
    and reasoning using the Gemini LLM or the offline template fallback.
    """
    benchmark_profile = retrieve_sector_benchmark(sector)
    
    # Check if Gemini key is present
    if gemini_api_key and gemini_api_key.strip():
        print("Generating risk narrative using Google Gemini agent...")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=gemini_api_key,
                temperature=0.3
            )
            
            prompt_template = PromptTemplate(
                input_variables=["company", "sector", "ratios", "solvency", "anomaly", "benchmark"],
                template=(
                    "You are a Senior Credit Analyst and Risk Officer at a development finance institution.\n"
                    "Your job is to write a plain-English risk narrative for a financial statement review.\n\n"
                    "### Company Name: {company}\n"
                    "### Industry Sector: {sector}\n\n"
                    "### Calculated Company Ratios:\n{ratios}\n\n"
                    "### Solvency Status (Altman Z-Score):\n{solvency}\n\n"
                    "### Anomaly Detection (Isolation Forest & SHAP):\n{anomaly}\n\n"
                    "### Sector Benchmark Profile:\n{benchmark}\n\n"
                    "Generate a structured credit risk report containing these EXACT sections:\n"
                    "### EXECUTIVE SUMMARY\n"
                    "Provide a 2-paragraph overview of the company's financial status and overall credit risk rating.\n\n"
                    "### KEY STRENGTHS\n"
                    "Identify 2-3 specific ratios where the company beats the sector benchmark. Explain why this is good.\n\n"
                    "### KEY CONCERNS\n"
                    "Identify 2-3 specific ratios where the company underperforms the benchmark, or mention Altman Z-Score distress/anomaly flags. Explain the risks.\n\n"
                    "### RECOMMENDATION\n"
                    "Provide a specific advisory recommendation regarding loan approval, covenant constraints, or mitigation strategies.\n\n"
                    "Maintain a formal, analytical credit analyst tone. Do not mention API keys or model names."
                )
            )
            
            # Format inputs
            ratios_formatted = json.dumps(ratios, indent=2)
            solvency_formatted = f"Altman Z-Score: {solvency.get('score'):.2f} | Zone: {solvency.get('zone')} ({solvency.get('model_used')})"
            anomaly_formatted = f"Anomaly Score: {anomaly.get('score'):.1f}% | Outlier: {anomaly.get('is_anomaly')} | Top Drivers: {anomaly.get('drivers')}"
            
            prompt = prompt_template.format(
                company=company_name,
                sector=sector,
                ratios=ratios_formatted,
                solvency=solvency_formatted,
                anomaly=anomaly_formatted,
                benchmark=benchmark_profile
            )
            
            response = llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"Gemini narrative generation failed: {e}. Falling back to offline templates.")
            
    # Offline Fallback
    return generate_offline_narrative(company_name, sector, ratios, solvency, anomaly)

if __name__ == "__main__":
    # Test narrative generation offline
    ratios_data = {
        "current_ratio": 0.85,
        "quick_ratio": 0.60,
        "cash_ratio": 0.20,
        "roa": -0.04,
        "roe": -0.15,
        "net_margin": -0.08,
        "operating_margin": -0.05,
        "debt_to_equity": 2.45,
        "debt_to_assets": 0.71,
        "asset_turnover": 0.50
    }
    solvency_data = {
        "score": 0.95,
        "zone": "Distress Zone (High Risk)",
        "model_used": "Altman Z'' Service Model"
    }
    anomaly_data = {
        "score": 75.2,
        "is_anomaly": True,
        "drivers": [{"label": "Current Ratio"}, {"label": "Debt-to-Equity"}]
    }
    narrative_output = generate_risk_narrative("Test Co.", "Fintech", ratios_data, solvency_data, anomaly_data)
    print(narrative_output)
