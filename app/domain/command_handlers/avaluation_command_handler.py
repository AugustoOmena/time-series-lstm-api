from app.domain.services.avaluation_model_service import executar_previsao, obtemX_para_um_dia, obtemX_testeY_teste_Janela_Deslisante
from app.schemas.ticker_request import TickerRequestBetweenDates, TickerRequest
import pandas as pd


def process_ticker(command: TickerRequestBetweenDates, model) -> dict:

    X_test, y_test, scaler, dates = obtemX_testeY_teste_Janela_Deslisante(command)

    test_predictions_norm, error = executar_previsao(model, X_test)

    if error: return error
    
    y_test_pronto = y_test.view(-1, 1)

    test_predictions = scaler.inverse_transform(test_predictions_norm)
    actual_prices = scaler.inverse_transform(y_test_pronto.cpu().numpy())

    result_data = []
    
    for pred, actual, date_val in zip(test_predictions, actual_prices, dates):
        
        p_val = round(float(pred), 2)
        a_val = round(float(actual), 2)
        diff_val = round(p_val - a_val, 2)
        
        data_formatada = pd.to_datetime(date_val).strftime('%Y-%m-%d')
        
        result_data.append({
            "date": data_formatada,
            "prediction": p_val,
            "actual": a_val,
            "diff": diff_val
        })

    response = {
        "ticker": command.ticker,
        "metadata": {
            "model_version": "lstm_39", 
            "period": "janela_deslizante",
            "count": len(result_data)
        },
        "data": result_data
    }

    return response


def process_ticker_single_day(command: TickerRequest, model) -> dict:

    X_test, scaler, actual_price, error = obtemX_para_um_dia(command)

    if error: return error

    test_predictions_norm, error_inf = executar_previsao(model, X_test)

    if error_inf: return error_inf

    # Inverte transformação
    pred = scaler.inverse_transform(test_predictions_norm)
    predicted_val = float(pred.reshape(-1)[0])

    # Preparação dos valores finais
    p_val = round(predicted_val, 2)
    
    item_data = {
        "date": command.target_date,
        "prediction": p_val,
        "actual": None,
        "diff": None
    }

    # Se tivermos o valor real, preenchemos a diferença
    if actual_price is not None:
        a_val = round(actual_price, 2)
        diff_val = round(p_val - a_val, 2)
        
        item_data["actual"] = a_val
        item_data["diff"] = diff_val

    # Retorno Padronizado
    return {
        "ticker": command.ticker,
        "metadata": {
            "model_version": "lstm_39",
            "period": "single_day",
            "type": "forecast" if actual_price is None else "backtest"
        },
        "data": [item_data]
    }
