from app.domain.services.avaluation_model_service import run_forecast, generate_recursive_forecast, obtemX_para_um_dia, getX_testY_test_Sliding_Window
from app.schemas.ticker_request import TickerRequestBetweenDates, TickerRequest
from app.domain.results.prediction_response_builder import PredictionResponseBuilder
from app.config.settings import get_settings

settings = get_settings()

def process_ticker(command: TickerRequestBetweenDates, model) -> dict:
    
    X_test, y_test, scaler, hist_dates = getX_testY_test_Sliding_Window(command)

    test_preds_norm, error = run_forecast(model, X_test)

    if error: return error
    
    hist_preds = scaler.inverse_transform(test_preds_norm).flatten().tolist()
    
    hist_actuals = []

    if y_test is not None and len(y_test) > 0:
        hist_actuals = scaler.inverse_transform(y_test.view(-1, 1).cpu().numpy()).flatten().tolist()

    last_val_norm = test_preds_norm[-1].item()

    fut_dates, fut_preds = generate_recursive_forecast(
        model=model,
        scaler=scaler,
        last_window_tensor=X_test[-1],
        last_val_norm=last_val_norm,
        last_date=hist_dates[-1],
        target_end_date=command.end_date
    )

    return (PredictionResponseBuilder()
            .set_ticker(command.ticker)
            .set_metadata(model_version=settings.MODEL_VERSION, period_type="janela_deslizante")
            .add_batch_predictions(hist_dates, hist_preds, hist_actuals)
            .add_batch_predictions(fut_dates, fut_preds, []) 
            .build())


def process_ticker_single_day(command: TickerRequest, model) -> dict:

    X_test, scaler, actual_price, error = obtemX_para_um_dia(command)

    if error: return error

    test_predictions_norm, error_inf = run_forecast(model, X_test)

    if error_inf: return error_inf

    # Inverte transformação
    pred = scaler.inverse_transform(test_predictions_norm)
    predicted_val = float(pred.reshape(-1)[0])

    return (PredictionResponseBuilder()
            .set_ticker(command.ticker)
            .set_metadata(model_version=settings.MODEL_VERSION, period_type="single_day")
            .add_prediction(date=command.target_date, prediction=predicted_val, actual=actual_price)
            .build())