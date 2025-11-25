from pydantic import BaseModel, Field
from datetime import date as Date

"""Payloads da aplicação"""
class TickerRequestBetweenDates(BaseModel):
    init_date: Date = Field(..., example="2025-06-01")
    end_date: Date = Field(..., example="2025-08-01")
    ticker: str = Field(..., example="ITUB4.SA")

class TickerRequest(BaseModel):
    target_date: Date = Field(..., example="2025-06-01")
    ticker: str = Field(..., example="ITUB4.SA")
