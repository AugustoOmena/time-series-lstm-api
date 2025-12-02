from app.domain.services.ml_handler.ml_handler import carregar_modelo_global
from contextlib import asynccontextmanager
from app.routers import api as api_router
from fastapi import FastAPI
from app.config.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 [Startup] Carregando modelo LSTM na memória...")
    
    caminho_modelo = settings.MODEL_PATH 
    
    modelo = carregar_modelo_global(caminho_modelo)
    
    app.state.model = modelo
    
    if modelo:
        print("✅ [Startup] Modelo carregado e pronto.")
    else:
        print("❌ [Startup] Falha ao carregar modelo.")

    yield 

app = FastAPI(lifespan=lifespan, title="Tech Challenge 4")

# Incluir rotas
app.include_router(api_router.router, prefix="/api", tags=["Predictions"])
