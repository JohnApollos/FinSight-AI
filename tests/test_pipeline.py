import os
import sys
import pytest

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.ingestion import validate_financial_equations
from backend.app.services.analysis import compute_ratios_from_statement

def test_financial_validation():
    # Healthy balanced statement
    stmt_balanced = {
        "assets": 1000000.0,
        "liabilities": 400000.0,
        "equity": 600000.0,
        "revenue": 500000.0,
        "expenses": 450000.0,
        "net_income": 50000.0
    }
    is_valid, warnings = validate_financial_equations(stmt_balanced)
    assert is_valid is True
    assert len(warnings) == 0

    # Unbalanced Balance Sheet
    stmt_unbalanced_bs = {
        "assets": 1000000.0,
        "liabilities": 400000.0,
        "equity": 500000.0, # 100k missing
        "revenue": 500000.0,
        "expenses": 450000.0,
        "net_income": 50000.0
    }
    is_valid, warnings = validate_financial_equations(stmt_unbalanced_bs)
    assert is_valid is False
    assert any("Balance Sheet Discrepancy" in w for w in warnings)

    # Unbalanced Income Statement
    stmt_unbalanced_is = {
        "assets": 1000000.0,
        "liabilities": 400000.0,
        "equity": 600000.0,
        "revenue": 500000.0,
        "expenses": 450000.0,
        "net_income": 80000.0 # should be 50k
    }
    is_valid, warnings = validate_financial_equations(stmt_unbalanced_is)
    assert is_valid is False
    assert any("Income Statement Discrepancy" in w for w in warnings)

def test_ratio_computations():
    stmt = {
        "assets": 1000000.0,
        "liabilities": 400000.0,
        "equity": 600000.0,
        "revenue": 500000.0,
        "expenses": 420000.0,
        "net_income": 80000.0,
        "working_capital": 200000.0,
        "retained_earnings": 300000.0,
        "cash": 100000.0,
        "accounts_receivable": 50000.0,
        "inventory": 50000.0,
        "interest_expense": 10000.0,
        "sic": "Manufacturing"
    }
    
    ratios = compute_ratios_from_statement(stmt)
    
    # Assert specific mathematical metrics
    # ROA = Net Income / Assets = 80,000 / 1,000,000 = 0.08
    assert ratios["roa"] == pytest.approx(0.08)
    
    # ROE = Net Income / Equity = 80,000 / 600,000 = 0.1333
    assert ratios["roe"] == pytest.approx(0.133333, rel=1e-3)
    
    # Net Margin = Net Income / Revenue = 80,000 / 500,000 = 0.16
    assert ratios["net_margin"] == pytest.approx(0.16)
    
    # Debt-to-Assets = Liabilities / Assets = 400,000 / 1,000,000 = 0.40
    assert ratios["debt_to_assets"] == pytest.approx(0.40)
    
    # Debt-to-Equity = Liabilities / Equity = 400,000 / 600,000 = 0.6666
    assert ratios["debt_to_equity"] == pytest.approx(0.666666, rel=1e-3)
    
    # Asset Turnover = Revenue / Assets = 500,000 / 1,000,000 = 0.50
    assert ratios["asset_turnover"] == pytest.approx(0.50)
    
    # Altman Classic solvency scores (Manufacturing)
    # X1 = WC / Assets = 200k / 1M = 0.2
    # X2 = RE / Assets = 300k / 1M = 0.3
    # X3 = EBIT / Assets = (80k + 10k) / 1M = 0.09
    # X4 = Equity / Liabilities = 600k / 400k = 1.5
    # X5 = Revenue / Assets = 500k / 1M = 0.5
    # Z = 1.2 * 0.2 + 1.4 * 0.3 + 3.3 * 0.09 + 0.6 * 1.5 + 0.999 * 0.5 = 0.24 + 0.42 + 0.297 + 0.90 + 0.4995 = 2.3565
    assert ratios["altman_z_classic"] == pytest.approx(2.3565, rel=1e-3)
