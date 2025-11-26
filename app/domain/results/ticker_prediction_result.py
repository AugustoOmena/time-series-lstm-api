# Se estiver usando Pydantic (recomendado para APIs como FastAPI)
from pydantic import BaseModel
from typing import List

class PredictionItem(BaseModel):
    prediction: float
    actual: float
    diff: float

class PredictionMetadata(BaseModel):
    model_version: str
    period: str

class TickerPredictionResult(BaseModel):
    ticker: str
    metadata: PredictionMetadata
    data: List[PredictionItem]