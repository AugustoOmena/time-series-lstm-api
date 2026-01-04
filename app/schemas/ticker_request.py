from pydantic import BaseModel, Field
from datetime import date as Date

"""Payloads da aplicação"""
class TickerRequestBetweenDates(BaseModel):
    init_date: Date = Field(default="2025-11-01", description="Data inicial")
    end_date: Date = Field(default="2026-02-01", description="Data final")
    ticker: str = Field(default="ITUB4.SA", description="Ticker")

class TickerRequest(BaseModel):
    target_date: Date = Field(default="2025-12-24", description="Data alvo")
    ticker: str = Field(default="ITUB4.SA", description="Ticker")
