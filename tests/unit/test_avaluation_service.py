from unittest.mock import patch
import numpy as np
import pytest
import pandas as pd
from app.domain.services.avaluation_model_service import obtemDadosHistoricos
from app.domain.services.avaluation_model_service import create_sequences_multivariate
from app.domain.services.avaluation_model_service import build_features_estrategia2


@patch('app.domain.services.avaluation_model_service.yf.download')
def test_obtem_dados_limpa_colunas(mock_yf, dataframe_yfinance_sujo):
    
    mock_yf.return_value = dataframe_yfinance_sujo.copy()
    
    resultado = obtemDadosHistoricos("PETR4", "2023-01-01", "2023-01-02")
    
    assert "Close" in resultado.columns
    mock_yf.assert_called_once()


def test_create_sequences_deve_criar_janelas_e_targets_corretos(dados_numpy_simples):

    seq_length = 3
    total_linhas = len(dados_numpy_simples) # 10

    X, y = create_sequences_multivariate(dados_numpy_simples, seq_length)

    qtd_esperada = total_linhas - seq_length
    
    assert X.shape == (qtd_esperada, seq_length, 2)
    
    assert y.shape == (qtd_esperada,)

    np.testing.assert_array_equal(X[0], dados_numpy_simples[0:3])

    assert y[0] == 3 
    assert y[0] == dados_numpy_simples[3, 0]

    assert y[-1] == 9


def test_build_features_deve_retornar_apenas_coluna_close(dataframe_ohlc_completo):

    resultado = build_features_estrategia2(dataframe_ohlc_completo)

    assert len(resultado.columns) == 1
    
    assert list(resultado.columns) == ['Close']
  
    assert 'Volume' not in resultado.columns
    assert 'Open' not in resultado.columns

    assert resultado.iloc[0]['Close'] == 10.2


def test_build_features_deve_falhar_se_coluna_close_nao_existir():

    df_invalido = pd.DataFrame({'Open': [10, 11], 'Volume': [100, 200]})

    with pytest.raises(KeyError) as excinfo:
        build_features_estrategia2(df_invalido)

    assert "'Close'" in str(excinfo.value)