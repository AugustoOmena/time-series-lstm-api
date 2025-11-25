from datetime import timedelta
import yfinance as yf
from functools import wraps, lru_cache
from fastapi import HTTPException
from datetime import datetime, date

# 1. Função auxiliar com CACHE. 
# O Python lembrará dos últimos 1024 tickers verificados para não travar a API.
@lru_cache(maxsize=1024)
def _check_ticker_on_yahoo(symbol: str) -> bool:
    """
    Tenta buscar o histórico de 1 dia. 
    Retorna True se vierem dados, False se estiver vazio/inválido.
    """
    try:
        # period="1d" é a request mais leve possível que confirma existência
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1d")
        return not history.empty
    except Exception:
        return False

# 2. O Decorator Principal
def validate_ticker_exists(func):
    @wraps(func)
    def wrapper(req, *args, **kwargs):
        # Extrai o ticker do objeto request (assumindo que ele tem o atributo .ticker)
        ticker_symbol = getattr(req, "ticker", None)

        if not ticker_symbol:
             raise HTTPException(status_code=400, detail="Ticker não fornecido no payload.")

        ticker_symbol = ticker_symbol.upper().strip()

        if not ticker_symbol.endswith(".SA") and len(ticker_symbol) <= 5: 
            pass 

        is_valid = _check_ticker_on_yahoo(ticker_symbol)

        if not is_valid:
            raise HTTPException(
                status_code=404, 
                detail=f"O ticker '{ticker_symbol}' não foi encontrado ou não possui dados ativos no Yahoo Finance."
            )

        # Se passou, injetamos o ticker (talvez normalizado) de volta no req e prosseguimos
        req.ticker = ticker_symbol
        return func(req, *args, **kwargs)
    return wrapper

def validate_date_rangefunc(func):
    @wraps(func)
    def wrapper(req, *args, **kwargs):

        start = getattr(req, 'init_date', None)
        end = getattr(req, 'end_date', None)

        if end <= start:
            raise HTTPException(status_code=400, detail="A data final deve ser posterior à data inicial.")
        if (end - start) < timedelta(days=60):
            raise HTTPException(status_code=400, detail="Período inválido: a previsão entre datas exige pelo menos 2 meses (60 dias).")

        return func(req, *args, **kwargs)
    return wrapper


def validate_has_date(func):
    @wraps(func)
    def wrapper(req, *args, **kwargs):

        date_value = getattr(req, 'target_date', None)

        if not date_value:
            raise HTTPException(
                status_code=400, 
                detail="O campo 'target_date' é obrigatório e não pode ser nulo."
            )

        if isinstance(date_value, (date, datetime)):
            return func(req, *args, **kwargs)

        # Se for String, validamos o formato YYYY-MM-DD
        if isinstance(date_value, str):
            try:
                # O strptime lança ValueError se a data for inválida (ex: 2025-02-30)
                datetime.strptime(date_value, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Data inválida ou formato incorreto: '{date_value}'. Use AAAA-MM-DD."
                )
        else:
            # Caso venha um int ou outro tipo
            raise HTTPException(status_code=400, detail="O campo 'date' deve ser uma data válida.")

        return func(req, *args, **kwargs)

    return wrapper