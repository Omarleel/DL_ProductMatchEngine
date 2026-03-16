from fastapi import FastAPI

from app.api.v1.datasets import router as datasets_router
from app.api.v1.clasificador import router as clasificador_router
from app.api.v1.homologador import router as homologador_router
from app.core.config import CONFIG_SEDES, get_settings
from app.services.inferencia_clasificador_service import _load_model, _load_maestro
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Iniciando Lifespan (Warm-up) ---")
    try:
        print("Cargando modelo clasificador...")
        _load_model()
        
        print("Cargando maestro de productos...")
        _load_maestro()
        
        print("--- Warm-up completado con éxito ---")
    except Exception as e:
        print(f"--- Error durante el warm-up: {e} ---")
    
    yield
    
    print("--- Limpiando recursos antes de apagar ---")

settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "env": settings.app_env,
        "sedes_configuradas": sorted(CONFIG_SEDES.keys()),
    }


app.include_router(datasets_router, prefix=settings.api_v1_prefix)
app.include_router(clasificador_router, prefix=settings.api_v1_prefix)
app.include_router(homologador_router, prefix=settings.api_v1_prefix)