from app.schemas.ticker_request import TickerRequestBetweenDates, TickerRequest
from app.domain.command_handlers.avaluation_command_handler import process_ticker, process_ticker_single_day
from app.domain.validators.ticker_service_validator import validate_ticker_exists, validate_date_rangefunc, validate_has_date

"""
Camada de serviço/command handler que orquestra a chamada ao domínio.
Mantemos-a simples: possível local para validações adicionais antes
de delegar à lógica de negócio.
"""

@validate_ticker_exists
@validate_date_rangefunc
def handle_ticker_info_between_dates(req: TickerRequestBetweenDates, model):
    return process_ticker(req, model)

@validate_ticker_exists
@validate_has_date
def handle_ticker_info_specific_date(req: TickerRequest, model):
    return process_ticker_single_day(req, model)
