from fastapi import Request, HTTPException

def get_model(request: Request):
    """
    Recupera o modelo armazenado no state da aplicação.
    """
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de IA não está disponível (falha no carregamento).")
    return model