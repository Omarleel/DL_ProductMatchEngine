from fastapi import FastAPI

from app.api.v1.datasets import router as datasets_router
from app.api.v1.clasificador import router as clasificador_router
from app.api.v1.homologador import router as homologador_router
from app.core.config import CONFIG_SEDES, get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)


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