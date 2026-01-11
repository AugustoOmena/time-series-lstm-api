# tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def dataframe_yfinance_sujo():
    data = {
        ('Close', 'PETR4.SA'): [30.0, 31.0],
        ('Open', 'PETR4.SA'): [29.0, 30.5]
    }
    return pd.DataFrame(data)
