import os
import json
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

BENCHMARKS_PATH = "backend/data/benchmarks.json"
CHROMA_PATH = "backend/data/chroma"

# Textual regulatory and analytical context for ChromaDB seeding
REGULATORY_DOCUMENTS = [
    {
        "id": "banking_basel_3",
        "document": (
            "Basel III Banking Regulation Framework:\n"
            "- Banks are required to maintain a minimum Capital Adequacy Ratio (CAR) of 8.0% (10.5% including the capital conservation buffer).\n"
            "- Under Central Bank of Kenya (CBK) guidelines, the minimum core capital to total risk-weighted assets is 10.5%, and total capital to risk-weighted assets is 14.5%.\n"
            "- Non-performing loan (NPL) ratios above 5.0% represent critical asset quality deterioration and require provisioning adjustments.\n"
            "- Liquidity Coverage Ratio (LCR) must exceed 100% to survive short-term liquidity shocks."
        ),
        "metadata": {"sector": "Banking", "topic": "Capital Adequacy & NPLs"}
    },
    {
        "id": "insurance_solvency_2",
        "document": (
            "Solvency II Insurance Regulation Standards:\n"
            "- Insurance companies must maintain a Solvency Capital Requirement (SCR) ratio of at least 100%.\n"
            "- Premium receivables represent high credit risk if not collected within 90 days. Net premium margins must exceed 5.0% for sustainable underwriting.\n"
            "- Debt-to-Equity ratios must remain conservative (typically <0.30) to protect policyholder reserves, as debt servicing cannot be funded from restricted policyholder assets."
        ),
        "metadata": {"sector": "Insurance", "topic": "SCR & Underwriting Reserves"}
    },
    {
        "id": "fintech_emerging_markets",
        "document": (
            "Fintech & Mobile Lending Underwriting Guidelines:\n"
            "- Microfinance and PAYG (pay-as-you-go) consumer finance entities operate with high leverage, but require a current ratio above 1.50 for basic liquidity.\n"
            "- Non-performing loans (NPLs) for digital credit products must remain below a 5.0% threshold. Portfolios with NPLs exceeding 10.0% represent severe credit risk.\n"
            "- Debt-to-equity ratios above 1.50 are common due to leverage funding, but interest coverage ratios (EBIT / Interest Expense) must exceed 2.0x to avoid insolvency."
        ),
        "metadata": {"sector": "Fintech", "topic": "Digital Credit & Cash Turnover"}
    },
    {
        "id": "manufacturing_ifrs_standards",
        "document": (
            "IFRS/IAS Manufacturing Credit Risk Guidelines:\n"
            "- Capital-heavy manufacturing firms require a current ratio above 1.50 for basic working capital health.\n"
            "- Fixed asset turnover must be evaluated alongside asset impairment indicators (IAS 36). Low asset turnover (<0.80) signals underutilized capacity.\n"
            "- Debt-to-equity ratios above 1.00 increase insolvency risks unless operating margins (EBIT / Revenue) exceed 10.0% to comfortably service long-term obligations."
        ),
        "metadata": {"sector": "Manufacturing", "topic": "Asset Impairment & Working Capital"}
    },
    {
        "id": "development_finance_ngo",
        "document": (
            "NGO and Development Finance Institutions (DFIs) Underwriting Guidelines:\n"
            "- Non-profit and developmental organizations prioritize liquidity over commercial profit margins.\n"
            "- The primary credit focus is on grant-dependency and asset turnover (sustainable deployment of funds rather than ROE).\n"
            "- Altman Z'' solvency scores below 1.20 indicate serious operational survival risks, but traditional leverage models are adjusted for grant receivables."
        ),
        "metadata": {"sector": "NGO / Development Finance", "topic": "Fund Utilization & Liquidity"}
    },
    {
        "id": "agriculture_emerging_markets",
        "document": (
            "Agricultural Credit Risk & Underwriting Standards:\n"
            "- Agribusiness credit is highly cyclical and vulnerable to weather and commodity price shocks.\n"
            "- Current ratio must exceed 1.80 to provide a liquidity cushion during off-seasons. Inventory turnover is a critical operational health metric.\n"
            "- Debt-to-equity ratios above 0.80 increase default risk. Underwriters require robust debt service coverage ratios (DSCR > 1.25x) and cash buffers."
        ),
        "metadata": {"sector": "Agriculture", "topic": "Agribusiness Cycles & Seasonality"}
    },
    {
        "id": "software_saas_metrics",
        "document": (
            "SaaS and Software Credit Risk Standards:\n"
            "- Asset-light SaaS companies have minimal physical collateral. Credit evaluations focus on recurring revenue streams and customer acquisition cost (CAC) efficiency.\n"
            "- Current ratios must exceed 2.00 due to prepaid deferred revenue structures. Net margin should exceed 10.0% to support high research & development (R&D) reinvestment.\n"
            "- Debt-to-equity should remain below 0.50, as leverage is typically replaced by equity financing (venture capital/growth equity) in early-to-mid stages."
        ),
        "metadata": {"sector": "Software", "topic": "Recurring Revenue & Leverage"}
    }
]

def format_sector_profile(sector: str, data: Dict[str, Any]) -> str:
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
    Retrieves the structured sector benchmark numeric profile directly from benchmarks.json.
    """
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
    return format_sector_profile(matched_sector, sector_info)

def retrieve_regulatory_context(sector: str) -> str:
    """
    Retrieves relevant regulatory and underwriting guidelines from ChromaDB (Vector Store).
    Falls back to matching offline text if ChromaDB is unavailable.
    """
    # Try using ChromaDB for semantic retrieval
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection("regulatory_context")
        
        # Populate if empty
        if collection.count() == 0:
            print("Seeding local ChromaDB with regulatory and accounting frameworks...")
            documents = []
            ids = []
            metadatas = []
            for doc in REGULATORY_DOCUMENTS:
                documents.append(doc["document"])
                ids.append(doc["id"])
                metadatas.append(doc["metadata"])
            collection.add(documents=documents, ids=ids, metadatas=metadatas)
            
        # Query ChromaDB semantically using the requested sector name
        results = collection.query(query_texts=[sector], n_results=1)
        if results and results["documents"] and results["documents"][0]:
            print(f"Retrieved regulatory context for '{sector}' from ChromaDB.")
            return results["documents"][0][0]
            
    except Exception as e:
        print(f"ChromaDB regulatory context skipped ({e}). Using key-based fallback.")
        
    # Python-based fallback matching
    for doc in REGULATORY_DOCUMENTS:
        if doc["metadata"]["sector"].lower() in sector.lower() or sector.lower() in doc["metadata"]["sector"].lower():
            return doc["document"]
            
    return REGULATORY_DOCUMENTS[2]["document"] # Return fintech as default

def generate_offline_narrative(company_name: str, sector: str, ratios: Dict[str, float], solvency: Dict[str, Any], anomaly: Dict[str, Any]) -> str:
    """
    Generates a financial risk narrative using programmatic templates (Offline Fallback).
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
    Generates a financial risk narrative using LangChain + Gemini, combining calculated ratios,
    structured numeric benchmarks, and semantically retrieved regulatory context from ChromaDB.
    """
    benchmark_profile = retrieve_sector_benchmark(sector)
    regulatory_context = retrieve_regulatory_context(sector)
    
    if gemini_api_key and gemini_api_key.strip():
        print("Generating risk narrative using Google Gemini agent...")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=gemini_api_key,
                temperature=0.3
            )
            
            prompt_template = PromptTemplate(
                input_variables=["company", "sector", "ratios", "solvency", "anomaly", "benchmark", "regulatory_context"],
                template=(
                    "You are a Senior Credit Analyst and Risk Officer at a development finance institution.\n"
                    "Your job is to write a plain-English risk narrative for a financial statement review.\n\n"
                    "### Company Name: {company}\n"
                    "### Industry Sector: {sector}\n\n"
                    "### Calculated Company Ratios:\n{ratios}\n\n"
                    "### Solvency Status (Altman Z-Score):\n{solvency}\n\n"
                    "### Anomaly Detection (Isolation Forest & SHAP):\n{anomaly}\n\n"
                    "### Sector Benchmark Profile:\n{benchmark}\n\n"
                    "### Regulatory & Methodological Context (Retrieved from Vector Store):\n{regulatory_context}\n\n"
                    "Generate a structured credit risk report containing these EXACT sections:\n"
                    "### EXECUTIVE SUMMARY\n"
                    "Provide a 2-paragraph overview of the company's financial status and overall credit risk rating. "
                    "Make sure to reference the company's specific ratios and how they align with the retrieved regulatory guidelines (e.g., Basel III capital requirements or Solvency II benchmarks if applicable).\n\n"
                    "### KEY STRENGTHS\n"
                    "Identify 2-3 specific ratios where the company beats the sector benchmark. Explain why this is good.\n\n"
                    "### KEY CONCERNS\n"
                    "Identify 2-3 specific ratios where the company underperforms the benchmark, or mention Altman Z-Score distress/anomaly flags. Explain the risks in terms of regulatory capital or debt service.\n\n"
                    "### RECOMMENDATION\n"
                    "Provide a specific advisory recommendation regarding loan approval, covenant constraints, or mitigation strategies.\n\n"
                    "Maintain a formal, analytical credit analyst tone. Do not mention API keys or model names."
                )
            )
            
            ratios_formatted = json.dumps(ratios, indent=2)
            solvency_formatted = f"Altman Z-Score: {solvency.get('score'):.2f} | Zone: {solvency.get('zone')} ({solvency.get('model_used')})"
            anomaly_formatted = f"Anomaly Score: {anomaly.get('score'):.1f}% | Outlier: {anomaly.get('is_anomaly')} | Top Drivers: {anomaly.get('drivers')}"
            
            prompt = prompt_template.format(
                company=company_name,
                sector=sector,
                ratios=ratios_formatted,
                solvency=solvency_formatted,
                anomaly=anomaly_formatted,
                benchmark=benchmark_profile,
                regulatory_context=regulatory_context
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
