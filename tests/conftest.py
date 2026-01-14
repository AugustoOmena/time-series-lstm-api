# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def dataframe_yfinance_sujo():
    data = {
        ('Close', 'PETR4.SA'): [30.0, 31.0],
        ('Open', 'PETR4.SA'): [29.0, 30.5]
    }
    return pd.DataFrame(data)


@pytest.fixture
def dados_numpy_simples():
    """
    Cria um array (10 linhas, 2 colunas) simples para testar janelas.
    Coluna 0: 0, 1, 2, ..., 9 (Usaremos isso como Target)
    Coluna 1: 10, 11, 12, ..., 19 (Feature extra)
    """
    # np.arange(10) cria [0, 1, 2...9]
    col1 = np.arange(10).reshape(-1, 1) 
    col2 = np.arange(10, 20).reshape(-1, 1)
    
    # Junta as colunas. Shape final: (10, 2)
    data = np.hstack((col1, col2))
    return data


@pytest.fixture
def dataframe_ohlc_completo():
    """
    Simula um DataFrame completo vindo do Yahoo Finance com várias colunas.
    """
    data = {
        'Open': [10.0, 11.0],
        'High': [10.5, 11.5],
        'Low': [9.5, 10.5],
        'Close': [10.2, 11.2],
        'Volume': [1000, 2000],
        'Adj Close': [10.2, 11.2]
    }
    return pd.DataFrame(data)