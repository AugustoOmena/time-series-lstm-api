from fastapi import APIRouter, Depends
from app.config.dependencies import get_model
from app.schemas.ticker_request import TickerRequestBetweenDates, TickerRequest
from app.domain.commands.avaluation_prices_commands import handle_ticker_info_specific_date, handle_ticker_info_between_dates

router = APIRouter()


@router.get("/v1/previsao-entre-datas", response_model=dict, summary="Previsão de preços por ticker")
def ticker_info_range(payload: TickerRequestBetweenDates = Depends(), model = Depends(get_model)):
    """
    Recebe data inicial, data final e ticker, retornando a previsão de preços da bolsa para esse período.
    """
    return handle_ticker_info_between_dates(payload, model)

@router.get("/v1/previsao-dia", response_model=dict, summary="Previsão de preço por ticker em um dia específico")
def ticker_info_day(payload: TickerRequest = Depends(), model = Depends(get_model)):
    """
    Recebe data e ticker, retornando a previsão de preço da bolsa e se houver, o preço real.
    """
    return handle_ticker_info_specific_date(payload, model)