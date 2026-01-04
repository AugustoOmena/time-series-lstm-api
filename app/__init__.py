from app.domain.services.ml_handler.ml_handler import load_global_model
from contextlib import asynccontextmanager
from app.routers import api as api_router
from fastapi import FastAPI
from app.config.settings import get_settings
from app.config.logging import configure_logging
import logging

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging early
    configure_logging()

    # --- STARTUP ---
    logger.info("[Startup] Loading LSTM model into memory...")

    model_path = settings.MODEL_PATH

    model = load_global_model(model_path)

    app.state.model = model

    if model:
        logger.info("[Startup] Model loaded and ready.")
    else:
        logger.error("[Startup] Failed to load model.")

    yield 

app = FastAPI(lifespan=lifespan, title="Tech Challenge 4")

# Incluir rotas
app.include_router(api_router.router, prefix="/api", tags=["Predictions"])
