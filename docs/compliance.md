# FinSight AI — Model Risk Management & Compliance Policy

This document outlines the Model Risk Management (MRM) frameworks, data governance standards, and anti-hallucination compliance controls implemented in **FinSight AI**.

---

## Model Governance Framework

Traditional financial institutions and credit departments operate under strict regulatory guidelines (e.g., US SR 11-7 / OCC 2011-12 on Model Risk Management). Unsupervised machine learning models and large language models (LLMs) pose unique governance risks. FinSight AI addresses these issues using a four-tier risk containment architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Compliance Framework                          │
├───────────────────┬───────────────────┬───────────────────┬─────────────┤
│  Model Validation │   Explainability  │ Telemetry Registry│  Governance │
├───────────────────┼───────────────────┼───────────────────┼─────────────┤
│  Double-entry     │  SHAP attribution │  MLflow immutable │ User consent│
│  equations check  │  weights plot     │  audit logs       │ disclaimer  │
└───────────────────┴───────────────────┴───────────────────┴─────────────┘
```

---

## Anti-Hallucination & Ingestion Controls

Large Language Models (LLMs) excel at processing unstructured text but are prone to calculation errors or column-shifting when reading tables. FinSight AI enforces strict data validation controls:

1. **Deterministic Double-Entry Auditing**: 
   All extracted metrics are programmatically validated before entering the credit scoring engine:
   - Balance Sheet validation: $\text{Assets} = \text{Liabilities} + \text{Equity}$
   - Income Statement validation: $\text{Net Income} = \text{Revenue} - \text{Expenses}$
2. **Pre-defined Tolerance Limits**:
   Discrepancies exceeding a $2\%$ rounding buffer flag validation warnings.
3. **Auditable Warnings**:
   Any statement discrepancy is written directly into the final compiled PDF report and Streamlit alert box, forcing credit analysts to perform manual verification.

---

## Model Explainability & Transparency (XAI)

Unsupervised anomaly models can flag credit profiles without explaining why. This violates fair-lending rules and audit standards. We solve this by integrating **SHAP (Shapley Additive Explanations)** on top of our Isolation Forest:

- **Attribution Mapping**: Every anomaly score is split into Shapley values, calculating how much each ratio (e.g., high Debt/Equity, low ROA) contributed to the outlier classification.
- **Visual Auditing**: The SHAP waterfall chart provides a clear explanation for compliance officers, showing exactly why a profile was flagged as a statistical outlier.

---

## Telemetry and Audit Registry

FinSight AI utilizes an immutable experiment registry backed by **MLflow** and a local SQLite database:
- **Reproducibility**: Every upload logs the file hash, LLM parameters, calculated ratios, and solvency classifications.
- **Traceability**: Generated ReportLab PDF reports are saved as artifacts inside the database, creating an unalterable audit history for regulatory inspections.

---

## Liability Disclaimer & Human-in-the-Loop Controls

FinSight AI is designed as a **Human-in-the-Loop decision support system** rather than an automated credit execution model. The generated PDF report embeds the following regulatory compliance notice:

> **REGULATORY COMPLIANCE NOTICE & CREDIT RISK DISCLAIMER**
> 
> *This report is generated for credit risk assessment purposes and represents a statistical analysis of financial statements. It does not constitute investment advice or an official endorsement of creditworthiness. Calculations (including the Altman Z-Score and Isolation Forest anomaly index) depend on the accuracy of the source documents. Financial institutions must combine this analysis with qualitative underwriting controls, KYC checks, and manual verification.*
