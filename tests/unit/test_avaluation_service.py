from unittest.mock import patch
from app.domain.services.avaluation_model_service import obtemDadosHistoricos

@patch('app.domain.services.avaluation_model_service.yf.download')
def test_obtem_dados_limpa_colunas(mock_yf, dataframe_yfinance_sujo):
    
    mock_yf.return_value = dataframe_yfinance_sujo.copy()
    
    resultado = obtemDadosHistoricos("PETR4", "2023-01-01", "2023-01-02")
    
    assert "Close" in resultado.columns
    mock_yf.assert_called_once()