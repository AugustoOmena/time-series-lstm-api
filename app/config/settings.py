from pathlib import Path
from functools import lru_cache

class Settings():
    APP_NAME: str = "ML Microservice"
    
    # Variáveis do Modelo
    SEQ_LENGTH: int = 30
    DEVICE: str = "cpu"
    MODEL_VERSION: str = "lstm_39"
    
    # Caminhos Base (Dinâmicos para funcionar em qualquer OS/Container)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CONFIG_DIR: Path = BASE_DIR / "config"
    MODEL_PATH: Path = BASE_DIR / "models" / "modelo_lstm_39.pkl"

# Padrão Singleton via lru_cache:
# Garante que as configurações sejam lidas/instanciadas apenas uma vez
@lru_cache()
def get_settings() -> Settings:
    return Settings()