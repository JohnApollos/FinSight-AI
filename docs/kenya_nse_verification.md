# FinSight AI — Kenyan NSE Verification & Benchmark Guide

This document describes how to test **FinSight AI** using real financial statements from companies listed on the **Nairobi Securities Exchange (NSE)** in Kenya, and compares their metrics against our integrated **World Bank Sub-Saharan Africa** benchmarks.

---

## 🇰🇪 Selected NSE Kenya Test Entities

To verify the ingestion and solvency analysis engine, we recommend testing with public annual reports from the following listed entities:

| Company Name | Sector | Key Financial Characteristics | Recommended Solvency model |
| :--- | :--- | :--- | :--- |
| **EABL (East African Breweries)** | Manufacturing | Large asset base, inventory cycles, high operating margins | Altman Z' (Classic Manufacturing) |
| **Bamburi Cement** | Manufacturing | Asset-heavy, industrial operations, sensitive to infrastructure capital | Altman Z' (Classic Manufacturing) |
| **Equity Group Holdings** | Banking | Highly liquid asset structures, no traditional current assets | Altman Z'' (Non-Manufacturing/Banking) |
| **KCB Group** | Banking | High volume of commercial loans, liquidity ratio focus | Altman Z'' (Non-Manufacturing/Banking) |
| **Stanbic Holdings Kenya** | Banking / Fintech | Mix of traditional commercial banking and digital/mobile lending | Altman Z'' (Non-Manufacturing/Banking) |
| **Britam Holdings** | Insurance | Insurance premiums, long-term reserves, premium receivables | Altman Z'' (Non-Manufacturing/Banking) |

---

## 📊 Comparison with World Bank & IFC African Benchmarks

When you upload an annual report of a Kenyan NSE company, the system compares its calculated ratios against pre-compiled sector averages. These averages are sourced from the **World Bank Enterprise Surveys (Sub-Saharan Africa)** and the **IFC SME Finance Gap Assessment**:

### 1. Manufacturing (e.g. EABL, Bamburi Cement)
- **World Bank Capacity Utilization**: Sub-Saharan Africa manufacturing sector average capacity utilization is **72.5%**. Bamburi Cement and EABL's asset turnover ratios are compared against this baseline to evaluate operational productivity.
- **World Bank Credit Constraint**: **38.4%** of manufacturing firms in Sub-Saharan Africa report severe access to finance constraints. Comparing Bamburi's Debt-to-Equity ratio helps classify if the company is overleveraged or has healthy access to credit.

### 2. Banking & Fintech (e.g. Equity, KCB, Stanbic)
- **IFC African Fintech NPL (Non-Performing Loan) Benchmark**: The regional average NPL for mobile/fintech lenders is **3.2%**, and traditional banking averages **4.5%**. FinSight AI compares the target bank's NPL ratio to these regional benchmarks.
- **ROE Benchmarks**: Regional average ROE for healthy banking institutions is **11.5%**.

---

## 📋 Step-by-Step Manual Verification Tutorial

1. **Download the PDF**: Go to the Investor Relations portal of [EABL](https://www.eabl.com/) or [Equity Group](https://equitygroupholdings.com/) and download their latest Annual Report PDF (e.g., FY 2024).
2. **Open the Dashboard**: Start uvicorn and streamlit, and navigate to the Streamlit page (`http://localhost:8501`).
3. **Upload the Document**:
   - Drag and drop the PDF into the file uploader.
   - Select the target sector (e.g. "Manufacturing" for EABL).
4. **Inspect Equations Validation**:
   - Scroll to the validation section. Check if the balance sheet matches.
   - *Note: If a bank's statement does not report traditional working capital (which is common for financial institutions), the parser defaults it to 0.0, and the Altman Z'' model handles this gracefully.*
5. **View Anomaly SHAP charts**:
   - Check the **Outlier Score**.
   - Review the **SHAP explanation chart** to see if any anomalies (e.g. extremely high debt-to-equity ratio or unusual cash buffers) pushed the entity towards an outlier classification.
6. **Compile and Download**:
   - Click **Download Audit-Ready PDF Report**.
   - Open the PDF and verify that the cover page, color-coded solvency dashboard, Matplotlib sector comparison charts, and compliance notices are formatted cleanly.
