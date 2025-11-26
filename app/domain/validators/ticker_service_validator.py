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
        # 1. Extração dos dados
        start = getattr(req, 'init_date', None)
        end = getattr(req, 'end_date', None)
        ticker = getattr(req, 'ticker', None)
        
        hoje = date.today()
        limite_futuro = hoje + timedelta(days=60)

        # 2. Validações de Lógica Temporal
        if end <= start:
            raise HTTPException(status_code=400, detail="A data final deve ser posterior à data inicial.")
        
        if (end - start) < timedelta(days=60):
            raise HTTPException(status_code=400, detail="Período de intervalo muito curto: a previsão exige pelo menos 60 dias entre inicio e fim.")

        if end > limite_futuro:
            raise HTTPException(
                status_code=400, 
                detail=f"Data final muito distante. O modelo só permite previsões até 60 dias a partir de hoje ({limite_futuro})."
            )

        # 3. Validação de Histórico Mínimo (Que fizemos antes)
        # Verifica se existe histórico antes do 'start' para alimentar o LSTM
        lookback_date = start - timedelta(days=90)
        try:
            hist_check = yf.download(ticker, start=lookback_date, end=start, progress=False, auto_adjust=True)
            if len(hist_check) < 30:
                first_valid = hist_check.index[0].date() if not hist_check.empty else "desconhecida"
                raise HTTPException(
                    status_code=400, 
                    detail=f"Data inicial inválida para {ticker}. Histórico insuficiente antes de {start}. Tente a partir de {first_valid}."
                )
        except Exception as e:
            if isinstance(e, HTTPException): raise e
            # Em produção, logar o erro do yfinance mas talvez não bloquear o usuário
            print(f"Aviso: Não foi possível validar histórico no YF: {e}")

        return func(req, *args, **kwargs)
    return wrapper

def validate_has_date(func):
    @wraps(func)
    def wrapper(req, *args, **kwargs):

        target_date = getattr(req, 'date', getattr(req, 'target_date', None))
        ticker = getattr(req, 'ticker', None)

        if not target_date:
            raise HTTPException(status_code=400, detail="Data alvo não fornecida.")

        hoje = date.today()
        limite_futuro = hoje + timedelta(days=60)

        if target_date > limite_futuro:
            raise HTTPException(
                status_code=400, 
                detail=f"Data muito distante. O modelo limita previsões a no máximo 60 dias futuros ({limite_futuro})."
            )

        lookback_date = target_date - timedelta(days=60)
        try:
            hist_check = yf.download(ticker, start=lookback_date, end=target_date, progress=False, auto_adjust=True)
            if len(hist_check) < 30:
                 raise HTTPException(
                    status_code=400, 
                    detail=f"Sem histórico suficiente para prever o dia {target_date}. O ticker {ticker} parece não ter dados suficientes neste período passado."
                )
        except Exception as e:
             if isinstance(e, HTTPException): raise e
             print(f"Aviso YF: {e}")

        return func(req, *args, **kwargs)
    return wrapper