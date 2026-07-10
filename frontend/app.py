import os
import requests
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# Config Page Layout
st.set_page_config(
    page_title="FinSight AI — Risk Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# Custom CSS for rich aesthetics and clean cards
st.markdown("""
<style>
    /* Premium glassmorphic card design */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-top: 4px solid #6366f1; /* Neon Indigo Accent */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-title {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f1f5f9;
    }
    .status-green { color: #10b981; font-weight: 700; }
    .status-amber { color: #f59e0b; font-weight: 700; }
    .status-red { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# App Sidebar
st.sidebar.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=60)
st.sidebar.markdown("""
<h2 style="font-family: 'Inter', sans-serif; font-weight: 700; font-size: 22px; margin-top: 10px; margin-bottom: 2px;">
    FinSight AI
</h2>
<p style="color: #64748b; font-family: 'Inter', sans-serif; font-size: 12px; margin-bottom: 20px;">
    Intelligent Financial Risk & Solvency Engine
</p>
""", unsafe_allow_html=True)

st.sidebar.divider()

# API Keys Configuration
st.sidebar.subheader("Configuration")
user_gemini_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Add your free Google Gemini API key to enable AI-powered risk narratives. If left blank, the system uses the default backend key or runs in Offline Fallback Mode."
)

selected_sector = st.sidebar.selectbox(
    "Target Analysis Sector",
    [
        "Fintech",
        "Insurance",
        "Banking",
        "Manufacturing",
        "Software",
        "NGO / Development Finance"
    ]
)

st.sidebar.divider()

# System Health Check
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success("Backend: Connected")
        if not health_data["api_key_configured"] and not user_gemini_key:
            st.sidebar.warning("LLM API: Offline Mode (No key)")
        else:
            st.sidebar.info("LLM API: Enabled")
    else:
        st.sidebar.error("Backend: Error")
except Exception:
    st.sidebar.error("Backend: Disconnected")

st.sidebar.divider()
st.sidebar.info(
    "Compliance: FinSight AI metrics are advisory. Decisions require human credit underwriting controls."
)

# Main Page Header (Premium Gradient Banner)
st.markdown("""
<div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 30px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
    <h1 style="color: #ffffff; margin: 0; font-family: 'Inter', sans-serif; font-weight: 700; font-size: 28px; letter-spacing: -0.5px;">
        FinSight AI &mdash; Risk Intelligence Dashboard
    </h1>
    <p style="color: #94a3b8; margin: 6px 0 0 0; font-family: 'Inter', sans-serif; font-size: 14px;">
        Ingest financial statements, compute solvency metrics, score anomalies, and synthesize credit narratives instantly.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Financial Document (PDF, DOCX, XLSX, XLS)",
    type=["pdf", "docx", "xlsx", "xls"],
    help="Upload balance sheets and income statement reports."
)

if uploaded_file is not None:
    # Trigger Ingestion
    with st.spinner("Processing document... Ingesting text, extracting financials, and scoring risk profiles..."):
        try:
            # Prepare files for multipart upload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"sector": selected_sector, "gemini_key": user_gemini_key}
            
            # Call FastAPI /analyze
            response = requests.post(f"{API_URL}/analyze", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                st.success("Document analyzed successfully!")
                
                # Extract components
                metrics = result["metrics"]
                analysis = result["analysis"]
                ratios = analysis["ratios"]
                solvency = analysis["solvency"]
                anomaly = analysis["anomaly"]
                
                # --- ACTION BUTTONS & SUMMARY CARDS ---
                col_btn, col_blank = st.columns([1.5, 4.5])
                with col_btn:
                    # Download PDF Report button routing to /report
                    # To avoid downloading twice, we query the endpoint and fetch the raw bytes
                    st.write("")
                    try:
                        # Request report
                        report_data = {
                            "sector": selected_sector,
                            "gemini_key": user_gemini_key
                        }
                        # Re-send bytes since uploader file pointer resets
                        files_report = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        
                        pdf_resp = requests.post(f"{API_URL}/report", files=files_report, data=report_data)
                        if pdf_resp.status_code == 200:
                            st.download_button(
                                label="Download Audit-Ready PDF Report",
                                data=pdf_resp.content,
                                file_name=f"FinSight_Audit_Report_{result['company_name'].replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to generate PDF download bytes.")
                    except Exception as report_err:
                        st.error(f"Report generation error: {report_err}")
                
                # Metric Cards
                col1, col2, col3, col4 = st.columns(4)
                
                # Altman Z-Score Zone color
                z_color = "status-green"
                if "grey" in solvency["zone"].lower():
                    z_color = "status-amber"
                elif "distress" in solvency["zone"].lower():
                    z_color = "status-red"
                    
                # Anomaly color
                anom_color = "status-red" if anomaly["is_anomaly"] else "status-green"
                anom_label = "SUSPICIOUS OUTLIER" if anomaly["is_anomaly"] else "STANDARD SIGNATURE"
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Company Name</div>
                        <div class="metric-value" style="font-size: 20px;">{result['company_name']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Solvency Status (Altman Z)</div>
                        <div class="metric-value {z_color}">{solvency['score']:.2f}</div>
                        <div style="font-size:12px; color:#64748b;">Zone: {solvency['zone']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Anomaly Outlier Score</div>
                        <div class="metric-value {anom_color}">{anomaly['score']:.1f}%</div>
                        <div style="font-size:12px; color:#64748b;">Class: {anom_label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Extraction Pipeline</div>
                        <div class="metric-value" style="font-size: 22px; color: #3498db;">{result['extraction_method']}</div>
                        <div style="font-size:12px; color:#64748b;">Time: {result['execution_time_seconds']}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Check for double-entry discrepancies
                if not result["validation"]["is_valid"]:
                    st.warning("Core Statement Validation Warnings Found:")
                    for warn in result["validation"]["warnings"]:
                        st.write(f"- {warn}")
                
                # --- TAB LAYOUT ---
                tab_dash, tab_narrative, tab_shap, tab_raw = st.tabs([
                    "Ratio Analysis Dashboard", 
                    "AI Credit Narrative", 
                    "Anomaly (SHAP) Diagnostics", 
                    "Raw Financial Metadata"
                ])
                
                with tab_dash:
                    col_t, col_c = st.columns([2.5, 3.5])
                    with col_t:
                        st.subheader("Financial Ratio Table")
                        # Format as pandas DataFrame for display
                        df_rows = []
                        # Ratios list
                        r_keys = [
                            ("current_ratio", "Current Ratio", "Liquidity"),
                            ("quick_ratio", "Quick Ratio", "Liquidity"),
                            ("cash_ratio", "Cash Ratio", "Liquidity"),
                            ("debt_to_equity", "Debt-to-Equity", "Leverage"),
                            ("debt_to_assets", "Debt-to-Assets", "Leverage"),
                            ("roa", "Return on Assets (ROA)", "Profitability"),
                            ("roe", "Return on Equity (ROE)", "Profitability"),
                            ("net_margin", "Net Profit Margin", "Profitability"),
                            ("operating_margin", "Operating Margin", "Profitability"),
                            ("asset_turnover", "Asset Turnover", "Efficiency")
                        ]
                        for k, lbl, cat in r_keys:
                            val = ratios.get(k, 0.0)
                            is_pct = k in ["roa", "roe", "net_margin", "operating_margin"]
                            df_rows.append({
                                "Category": cat,
                                "Ratio Name": lbl,
                                "Calculated Value": f"{val*100:.2f}%" if is_pct else f"{val:.2f}"
                            })
                        st.dataframe(df_rows, use_container_width=True, hide_index=True)
                        
                    with col_c:
                        st.subheader("Plotly Ratio Benchmark Comparison")
                        # Let's create an interactive bar chart of ratios compared to averages
                        # Fetch benchmark values
                        try:
                            with open("backend/data/benchmarks.json", "r", encoding="utf-8") as f_bench:
                                all_bench = json.load(f_bench)
                        except Exception:
                            all_bench = {}
                            
                        # Locate matching sector
                        matched_s = "Fintech"
                        for s in all_bench.keys():
                            if s.lower() in selected_sector.lower() or selected_sector.lower() in s.lower():
                                matched_s = s
                                break
                                
                        bench_values = all_bench.get(matched_s, {})
                        
                        # Metrics to plot
                        metrics_plot = ["current_ratio", "debt_to_equity", "roa", "roe", "net_margin"]
                        labels_plot = ["Current Ratio", "Debt/Equity", "ROA", "ROE", "Net Margin"]
                        
                        comp_y = []
                        bench_y = []
                        for m in metrics_plot:
                            scale = 100.0 if m in ["roa", "roe", "net_margin"] else 1.0
                            comp_y.append(ratios.get(m, 0.0) * scale)
                            
                            b_val = bench_values.get(m, 0.0)
                            if m in ["roa", "roe", "net_margin"] and b_val < 1.0:
                                b_val = b_val * 100.0
                            bench_y.append(b_val)
                            
                        fig = go.Figure(data=[
                            go.Bar(name="Company", x=labels_plot, y=comp_y, marker_color="#1b2a4a"),
                            go.Bar(name=f"Sector Average ({matched_s})", x=labels_plot, y=bench_y, marker_color="#7f8c8d")
                        ])
                        fig.update_layout(
                            barmode="group",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=320
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                with tab_narrative:
                    st.subheader("Agentic Solvency narrative")
                    st.write(
                        "Below is the plain-English risk narrative generated by the LangChain agent. "
                        "It analyzes core solvency zones, checks outliers, and draws comparisons from World Bank/IFC/Damodaran averages."
                    )
                    st.divider()
                    
                    # Split narrative by sections and render nicely
                    sec_blocks = result["narrative"].split("### ")
                    for block in sec_blocks:
                        if not block.strip():
                            continue
                        block_lines = block.split("\n", 1)
                        sec_title = block_lines[0].strip()
                        sec_body = block_lines[1].strip() if len(block_lines) > 1 else ""
                        
                        st.markdown(f"#### {sec_title}")
                        st.markdown(sec_body)
                        st.divider()
                        
                with tab_shap:
                    st.subheader("Anomaly Contribution Diagnostics (SHAP)")
                    st.write(
                        "SHAP (Shapley Additive Explanations) measures how much each financial ratio pushed the "
                        "company's classification score away from the average baseline of healthy companies. "
                        "Negative values (Red bars) drive the score towards anomalous classification, indicating metrics "
                        "with unusual variance."
                    )
                    
                    # Embed Matplotlib SHAP chart
                    if os.path.exists(anomaly["chart_path"]):
                        img = Image.open(anomaly["chart_path"])
                        st.image(img, use_column_width=False, width=650)
                    else:
                        st.write("No SHAP waterfall chart generated. Confirm that the Isolation Forest model has been trained.")
                        
                    # Show drivers in table
                    st.write("##### Key Contributing Ratio Deficits:")
                    driver_rows = []
                    for driver in anomaly.get("drivers", []):
                        driver_rows.append({
                            "Ratio": driver["label"],
                            "Calculated Value": f"{driver['value']*100:.1f}%" if driver["feature"] in ["roa", "roe", "net_margin", "operating_margin"] else f"{driver['value']:.2f}",
                            "SHAP Impact Score": f"{driver['shap_value']:.4f}"
                        })
                    st.dataframe(driver_rows, hide_index=True)
                    
                with tab_raw:
                    st.subheader("Extracted Absolute Statements (Ingestion Output)")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.write("##### Income Statement Metrics")
                        st.write(f"- **Revenue:** {result['currency']} {metrics['revenue']:,.2f}")
                        st.write(f"- **Expenses:** {result['currency']} {metrics['expenses']:,.2f}")
                        st.write(f"- **Net Income:** {result['currency']} {metrics['net_income']:,.2f}")
                        st.write(f"- **Interest Expenses:** {result['currency']} {metrics['interest_expense']:,.2f}")
                    with col_m2:
                        st.write("##### Balance Sheet Metrics")
                        st.write(f"- **Total Assets:** {result['currency']} {metrics['assets']:,.2f}")
                        st.write(f"- **Total Liabilities:** {result['currency']} {metrics['liabilities']:,.2f}")
                        st.write(f"- **Total Shareholders Equity:** {result['currency']} {metrics['equity']:,.2f}")
                        st.write(f"- **Retained Earnings:** {result['currency']} {metrics['retained_earnings']:,.2f}")
                        st.write(f"- **Working Capital:** {result['currency']} {metrics['working_capital']:,.2f}")
                        st.write(f"- **Cash & Equivalents:** {result['currency']} {metrics['cash']:,.2f}")
                        
            else:
                st.error(f"Error analyzing document: Backend returned status code {response.status_code}")
                st.write(response.text)
        except Exception as e:
            st.error(f"Failed to communicate with analysis server: {e}")

st.divider()

# --- SYSTEM STATS & PAST RUNS HISTORY ---
st.subheader("System Analytics & History")
col_h1, col_h2 = st.columns([4, 2])

with col_h1:
    st.write("##### Recent Ingested Documents (Last 10 Runs)")
    try:
        hist_resp = requests.get(f"{API_URL}/history", timeout=3)
        if hist_resp.status_code == 200:
            hist_records = hist_resp.json()
            if hist_records:
                # Format for display
                display_hist = []
                for idx, r in enumerate(hist_records, 1):
                    display_hist.append({
                        "No.": idx,
                        "Timestamp": r.get("timestamp"),
                        "Filename": r.get("filename"),
                        "Company Name": r.get("company_name"),
                        "Sector": r.get("sector"),
                        "Z-Score": r.get("z_score"),
                        "Anomaly Index": f"{r.get('anomaly_score')}%"
                    })
                st.dataframe(display_hist, hide_index=True, use_container_width=True)
            else:
                st.write("No files processed yet.")
        else:
            st.write("Error loading history.")
    except Exception:
        st.write("Could not retrieve file history.")

with col_h2:
    st.write("##### System Telemetry Metrics")
    try:
        met_resp = requests.get(f"{API_URL}/metrics", timeout=3)
        if met_resp.status_code == 200:
            met_data = met_resp.json()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Processed</div>
                <div class="metric-value" style="font-size: 20px;">{met_data['total_documents_analyzed']} Documents</div>
                <div style="font-size:12px; color:#64748b;">Average Z-Score: {met_data['mean_solvency_score']:.2f}</div>
                <div style="font-size:12px; color:#64748b;">Outlier Rate: {met_data['anomaly_rate_percent']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Error loading telemetry.")
    except Exception:
        st.write("Could not retrieve telemetry.")
