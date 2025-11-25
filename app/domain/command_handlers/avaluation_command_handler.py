from app.domain.services.avaluation_model_service import executar_previsao, obtemX_para_um_dia, obtemX_testeY_teste_Janela_Deslisante
from app.schemas.ticker_request import TickerRequestBetweenDates, TickerRequest


def process_ticker(command: TickerRequestBetweenDates, model) -> dict:

    X_test, y_test, scaler = obtemX_testeY_teste_Janela_Deslisante(command.ticker, command.init_date, command.end_date)

    test_predictions_norm, error = executar_previsao(model, X_test)

    if error: return error
    
    # Inverte a normalização para obter os preços reais usando o mesmo
    # scaler que foi usado na função de preparação dos dados.
    # É CRÍTICO que o y_test esteja em [Amostras, 1] para o scaler
    y_test_pronto = y_test.view(-1, 1)

    test_predictions = scaler.inverse_transform(test_predictions_norm)
    actual_prices = scaler.inverse_transform(y_test_pronto.cpu().numpy())

    return {"message": f"Test predictions = {test_predictions}, Actual Prices = {actual_prices}"}


def process_ticker_single_day(command: TickerRequest, model) -> dict:
    
    X_test, scaler = obtemX_para_um_dia(command.ticker, command.target_date)

    test_predictions_norm, error = executar_previsao(model, X_test)

    if error: return error

    # Inverte transformação
    pred = scaler.inverse_transform(test_predictions_norm)

    # Retorna valor previsto (primeiro e único)
    return {"predicted": float(pred.reshape(-1)[0])}
