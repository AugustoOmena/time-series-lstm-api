"""
Entrypoint da aplicação - redireciona para o pacote `app` onde a FastAPI
realmente vive.
"""

# Expor a instância FastAPI definida em app/__init__.py
from app import app