import io
import pickle
import torch
import logging
from app.domain.services.avaluation_model_service import SimpleLSTM

logger = logging.getLogger(__name__)

# Sua classe customizada para converter de GPU para CPU
class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        if module == '__main__' and name == 'SimpleLSTM':
            return SimpleLSTM
        
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')

        return super().find_class(module, name)

# Global variable that will hold the loaded model
_loaded_model = None

def load_global_model(model_path: str = None):
    """
    Load the model file into a global variable.
    Intended to be called once at application startup.
    """
    global _loaded_model
    model_file = model_path

    try:
        logger.info("Loading model %s into memory...", model_file)
        with open(model_file, 'rb') as f:
            _loaded_model = CpuUnpickler(f).load()

        if hasattr(_loaded_model, 'eval'):
            _loaded_model.eval()

        logger.info("Model loaded successfully!")
        return _loaded_model

    except Exception as e:
        logger.exception("FATAL: Error loading model: %s", e)
        return None

def get_model():
    """
    Return the already-loaded model instance.
    If not present, attempt lazy-loading via `load_global_model()`.
    """
    global _loaded_model
    if _loaded_model is None:
        return load_global_model()
    return _loaded_model