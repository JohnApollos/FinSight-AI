import os
from datetime import datetime
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

BENCHMARKS_PATH = "backend/data/benchmarks.json"

def generate_comparison_chart(company_ratios: Dict[str, float], sector: str) -> str:
    """Generates a comparison bar chart: Company vs Sector Benchmarks."""
    # Load benchmarks
    try:
        with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
            benchmarks_data = json.load(f)
    except Exception:
        from backend.app.utils.damodaran_downloader import DEFAULT_BENCHMARKS
        benchmarks_data = DEFAULT_BENCHMARKS

    matched_sector = "Fintech"
    for s in benchmarks_data.keys():
        if s.lower() in sector.lower() or sector.lower() in s.lower():
            matched_sector = s
            break
    bench = benchmarks_data[matched_sector]

    # Select main metrics to compare
    metrics = ["current_ratio", "debt_to_equity", "roa", "roe", "net_margin"]
    labels = ["Current Ratio", "Debt/Equity", "ROA", "ROE", "Net Margin"]
    
    comp_vals = []
    bench_vals = []
    
    for m in metrics:
        # Scale percentages for visual alignment on the chart
        scale = 100.0 if m in ["roa", "roe", "net_margin"] else 1.0
        # If margins are in benchmarks as raw percentages (like 14.5% instead of 0.145)
        # Damodaran stores them as absolute percent numbers, while our computed ratios are fractions
        comp_val = company_ratios.get(m, 0.0) * scale
        
        bench_val = bench.get(m, 0.0)
        # If the benchmark is already scaled (e.g. 14.5 for Net Margin), but scale factor was 100
        # we do not multiply it again.
        if m in ["roa", "roe", "net_margin"] and bench_val < 1.0:
            bench_val = bench_val * 100.0
            
        comp_vals.append(comp_val)
        bench_vals.append(bench_val)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 3.5))
    plt.bar(x - width/2, comp_vals, width, label="Company", color="#1b2a4a")
    plt.bar(x + width/2, bench_vals, width, label=f"Sector ({matched_sector})", color="#7f8c8d")
    
    plt.ylabel("Value / Percentage (%)")
    plt.title("Financial Ratios vs. Sector Benchmark", fontsize=10, fontweight="bold", pad=10)
    plt.xticks(x, labels, fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    chart_path = "backend/data/ratio_comparison.png"
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path

def compile_pdf_report(
    output_pdf_path: str,
    company_name: str,
    sector: str,
    reporting_period: str,
    currency: str,
    extracted_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
    narrative: str
):
    """Compiles the full ReportLab PDF report."""
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    ratios = analysis_results.get("ratios", {})
    solvency = analysis_results.get("solvency", {})
    anomaly = analysis_results.get("anomaly", {})
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=54, leftMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=colors.HexColor("#1b2a4a"),
        alignment=0, # Left-aligned
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#7f8c8d"),
        spaceAfter=40
    )
    
    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#1b2a4a"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=8
    )
    
    meta_label_style = ParagraphStyle(
        "MetaLabel",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1b2a4a")
    )
    
    meta_val_style = ParagraphStyle(
        "MetaVal",
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#2c3e50")
    )

    story = []
    
    # ================= PAGE 1: COVER PAGE =================
    story.append(Spacer(1, 1 * inch))
    story.append(Paragraph("FINSIGHT AI", ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#3498db"), spaceAfter=10)))
    story.append(Paragraph("Intelligent Financial Risk & Solvency Analysis", title_style))
    story.append(Paragraph(f"Commercial credit review and anomaly detection engine.", subtitle_style))
    
    # Metadata Box Table
    metadata_data = [
        [Paragraph("Company Name:", meta_label_style), Paragraph(company_name, meta_val_style)],
        [Paragraph("Sector / Industry:", meta_label_style), Paragraph(sector, meta_val_style)],
        [Paragraph("Reporting Period:", meta_label_style), Paragraph(reporting_period, meta_val_style)],
        [Paragraph("Currency:", meta_label_style), Paragraph(currency, meta_val_style)],
        [Paragraph("Analysis Date:", meta_label_style), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), meta_val_style)],
        [Paragraph("Solvency Status:", meta_label_style), Paragraph(f"<b>{solvency.get('zone')}</b> (Score: {solvency.get('score'):.2f})", meta_val_style)],
        [Paragraph("Anomaly Signature:", meta_label_style), Paragraph(f"{'SUSPICIOUS OUTLIER' if anomaly.get('is_anomaly') else 'STANDARD'} (Score: {anomaly.get('score'):.1f}%)", meta_val_style)]
    ]
    
    meta_table = Table(metadata_data, colWidths=[2.0 * inch, 4.0 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1"))
    ]))
    
    story.append(meta_table)
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("<b>CONFIDENTIALITY NOTICE</b>: This report is for internal credit committee review only and contains proprietary model insights.", ParagraphStyle("Notice", fontName="Helvetica-Oblique", fontSize=8, textColor=colors.HexColor("#94a3b8"))))
    story.append(PageBreak())
    
    # ================= PAGE 2: EXECUTIVE DASHBOARD =================
    story.append(Paragraph("Executive Financial Dashboard", h1_style))
    story.append(Paragraph("The dashboard below displays the calculated financial metrics alongside industry benchmark targets. Status highlights are color-coded to indicate strengths (Green), warnings (Amber), and critical deficits (Red).", body_style))
    story.append(Spacer(1, 5))
    
    # Table headers
    ratio_table_data = [
        [
            Paragraph("<b>Ratio Category / Name</b>", meta_label_style), 
            Paragraph("<b>Value</b>", meta_label_style), 
            Paragraph("<b>Benchmark</b>", meta_label_style), 
            Paragraph("<b>Status</b>", meta_label_style)
        ]
    ]
    
    # Build ratio rows dynamically
    # Load benchmarks
    try:
        with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
            benchmarks_data = json.load(f)
    except Exception:
        from backend.app.utils.damodaran_downloader import DEFAULT_BENCHMARKS
        benchmarks_data = DEFAULT_BENCHMARKS
        
    matched_sector = "Fintech"
    for s in benchmarks_data.keys():
        if s.lower() in sector.lower() or sector.lower() in s.lower():
            matched_sector = s
            break
    bench = benchmarks_data[matched_sector]
    
    # Ratios to display
    display_ratios = [
        ("current_ratio", "Current Ratio (Liquidity)", True, ">= 1.5", "green", "red"),
        ("debt_to_equity", "Debt-to-Equity (Leverage)", False, "<= 0.8", "red", "green"),
        ("roa", "Return on Assets (ROA)", True, ">= 5.0%", "green", "red"),
        ("roe", "Return on Equity (ROE)", True, ">= 12.0%", "green", "red"),
        ("net_margin", "Net Profit Margin", True, ">= 10.0%", "green", "red"),
        ("asset_turnover", "Asset Turnover (Efficiency)", True, ">= 0.8", "green", "red")
    ]
    
    row_styles = [] # List of tuples: (row_idx, bg_color, text_color)
    
    for idx, (key, label, higher_is_better, target_desc, ok_color, fail_color) in enumerate(display_ratios, 1):
        val = ratios.get(key, 0.0)
        # Format percentage metrics
        is_pct = key in ["roa", "roe", "net_margin"]
        val_str = f"{val*100:.1f}%" if is_pct else f"{val:.2f}"
        
        bench_val = bench.get(key, 0.0)
        # If the benchmark is already a percentage in the json (e.g. 14.5 instead of 0.145)
        if is_pct and bench_val < 1.0:
            bench_val = bench_val * 100.0
            
        bench_str = f"{bench_val:.1f}%" if is_pct else f"{bench_val:.2f}"
        
        # Color coding status
        val_comp = val * 100 if is_pct else val
        is_ok = (val_comp >= bench_val) if higher_is_better else (val_comp <= bench_val)
        
        status_text = "Outperform" if is_ok else "Underperform"
        status_bg = "#d4edda" if is_ok else "#f8d7da"
        status_text_color = "#155724" if is_ok else "#721c24"
        
        ratio_table_data.append([
            Paragraph(label, meta_val_style),
            Paragraph(val_str, meta_val_style),
            Paragraph(bench_str, meta_val_style),
            Paragraph(f"<font color='{status_text_color}'><b>{status_text}</b></font>", meta_val_style)
        ])
        
        row_styles.append((idx, status_bg))
        
    dashboard_table = Table(ratio_table_data, colWidths=[2.5 * inch, 1.3 * inch, 1.3 * inch, 1.4 * inch])
    
    # Base table styling
    t_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1"))
    ]
    
    # Add dynamic cell color backgrounds for Status column
    for r_idx, bg_col in row_styles:
        t_styles.append(("BACKGROUND", (3, r_idx), (3, r_idx), colors.HexColor(bg_col)))
        
    dashboard_table.setStyle(TableStyle(t_styles))
    story.append(dashboard_table)
    story.append(Spacer(1, 15))
    
    # Embed Comparison Chart
    chart_path = generate_comparison_chart(ratios, sector)
    if os.path.exists(chart_path):
        story.append(Image(chart_path, width=5.5 * inch, height=3.2 * inch))
        
    story.append(PageBreak())
    
    # ================= PAGE 3: ANOMALY & RISK NARRATIVE =================
    story.append(Paragraph("Anomaly Detection & Risk Analytics", h1_style))
    story.append(Paragraph(
        f"The Isolation Forest model evaluates the multidimensional variance of {company_name}'s ratios. "
        f"The firm scored an anomaly index of <b>{anomaly.get('score'):.1f}%</b>. "
        f"Below is the SHAP contribution chart mapping the primary financial drivers of this assessment.",
        body_style
    ))
    
    # Embed SHAP chart if available
    shap_path = anomaly.get("chart_path", "")
    if shap_path and os.path.exists(shap_path):
        story.append(Image(shap_path, width=5.0 * inch, height=3.3 * inch))
    else:
        story.append(Paragraph("<i>[SHAP Explainer Chart omitted or unavailable]</i>", body_style))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("AI-Generated Risk Narrative & Solvency Outlook", h1_style))
    
    # Format and append Narrative sections
    sections = narrative.split("### ")
    for sec in sections:
        if not sec.strip():
            continue
        # Split title and body
        lines = sec.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        
        if title:
            story.append(Paragraph(title.title(), ParagraphStyle("SecSub", fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#2c3e50"), spaceBefore=8, spaceAfter=4)))
            
        # Clean markdown bullets and format paragraphs
        paragraphs = body.split("\n\n")
        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            
            # Format markdown bolding `**text**` to HTML `<b>text</b>`
            p_clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", p_clean)
            
            # Format bullet points
            if p_clean.startswith("- ") or p_clean.startswith("* "):
                p_bullets = p_clean.split("\n")
                for bullet in p_bullets:
                    b_text = bullet.strip()[2:]
                    story.append(Paragraph(f"&bull; {b_text}", ParagraphStyle("Bullet", parent=body_style, leftIndent=12, spaceAfter=3)))
            else:
                story.append(Paragraph(p_clean, body_style))
                
    story.append(PageBreak())
    
    # ================= PAGE 4: COMPLIANCE & METHODOLOGY =================
    story.append(Paragraph("Methodology & Compliance Disclosure", h1_style))
    story.append(Paragraph("<b>Analysis Methodology:</b>", ParagraphStyle("SubH", fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#2c3e50"), spaceBefore=6)))
    story.append(Paragraph(
        "FinSight AI processes absolute financial figures via a multi-layered classification pipeline. "
        "Liquidity, leverage, profitability, and operational efficiency ratios are calculated locally. "
        "Solvency zones are classified using Altman's double-prime model (Z''-Score) for service, software, bank, and insurance institutions, "
        "and Altman's modified private manufacturing model (Z'-Score) for industrial entities. "
        "Anomaly detection is evaluated via an Isolation Forest trained on a reference baseline of 300+ healthy financial statements "
        "across corresponding SIC codes. Driver attributions are calculated using Shapley Additive Explanations (SHAP) to ensure "
        "transparency and auditability.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Limitation of Liability & AI Guardrails:</b>", ParagraphStyle("SubH", fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#2c3e50"))))
    story.append(Paragraph(
        "This analysis is advisory and generated automatically using a combination of machine learning models and LLM reasoning. "
        "Model outputs are not a substitute for human professional judgment, audits, or detailed credit underwriting. "
        "FinSight AI does not guarantee the absolute accuracy of parsed figures and does not assume liability for credit decisions, "
        "portfolio losses, or compliance infractions resulting from the use of this report.",
        body_style
    ))
    
    story.append(Spacer(1, 20))
    
    # Compliance Box Table
    compliance_box_data = [[
        Paragraph(
            "<b>COMPLIANCE NOTICE (DFI / REGULATED STANDARDS):</b><br/>"
            "This report adheres to the Model Risk Management (MRM) standards and compliance guidelines. "
            "The explanation drivers are statistical contributions (SHAP) and are explainable and auditable. "
            "Human-in-the-loop control is required before credit approval.",
            ParagraphStyle("CompText", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#721c24"))
        )
    ]]
    comp_box_table = Table(compliance_box_data, colWidths=[6.0 * inch])
    comp_box_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8d7da")),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f5c6cb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    
    story.append(comp_box_table)
    
    # Build Document
    doc.build(story)
    print(f"PDF report compiled successfully at {output_pdf_path}")
    
    # Cleanup chart images from temp folder
    for path in [chart_path, "backend/data/ratio_comparison.png"]:
        if os.path.exists(path):
            try:
                # Keep them if needed for web view, or cleanup
                pass
            except Exception:
                pass

if __name__ == "__main__":
    # Test compilation
    test_ratios = {
        "current_ratio": 1.25,
        "quick_ratio": 0.95,
        "cash_ratio": 0.35,
        "roa": 0.045,
        "roe": 0.112,
        "net_margin": 0.085,
        "operating_margin": 0.125,
        "debt_to_equity": 0.65,
        "debt_to_assets": 0.38,
        "asset_turnover": 0.75,
        "altman_z_classic": 2.15,
        "altman_z_service": 2.35
    }
    test_solvency = {
        "score": 2.35,
        "zone": "Grey Zone (Moderate Risk)",
        "model_used": "Altman Z'' Service Model"
    }
    test_anomaly = {
        "score": 42.5,
        "is_anomaly": False,
        "drivers": [],
        "chart_path": "backend/data/shap_waterfall.png"
    }
    compile_pdf_report(
        "backend/data/test_report.pdf",
        "Acme Software Ltd",
        "Software",
        "FY 2024",
        "USD",
        {},
        {"ratios": test_ratios, "solvency": test_solvency, "anomaly": test_anomaly},
        "### EXECUTIVE SUMMARY\nAcme Software shows healthy operational status.\n\n### KEY STRENGTHS\nROE and Net margins outperform sector averages.\n\n### KEY CONCERNS\nLiquidity is slightly tight.\n\n### RECOMMENDATION\nApprove credit with standard covenants."
    )
