from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI

from app.api.v1.datasets import router as datasets_router
from app.api.v1.clasificador import router as clasificador_router
from app.api.v1.homologador import router as homologador_router
from app.core.config import CONFIG_SEDES, get_settings
from app.services.inferencia_clasificador_service import _get_resolvers, _load_model as _load_clasificador_model
from app.services.inferencia_homologacion_service import _load_homologador_context, _load_homologador_model
from app.services.maestro_service import cargar_maestro
from app.services.tenant_service import (
    get_tenant_artifacts_dir,
    get_tenant_raw_data_dir,
    list_known_tenants,
)


def _warmup_if_available(nombre: str, loader: Callable[[], object], required_path: Optional[Path] = None) -> bool:
    if required_path is not None and not required_path.exists():
        print(f"Warm-up omitido para {nombre}: no existe {required_path}")
        return False

    try:
        loader()
        print(f"Warm-up OK: {nombre}")
        return True
    except FileNotFoundError as exc:
        print(f"Warm-up omitido para {nombre}: {exc}")
        return False
    except Exception as exc:
        print(f"Warm-up con error en {nombre}: {exc}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    print("--- Iniciando Lifespan (Warm-up multitenant) ---")

    tenants = list_known_tenants()
    if not tenants:
        print("Warm-up omitido: no se detectaron tenants con datasets o artefactos.")

    for tenant in tenants:
        print(f"--- Warm-up tenant={tenant} ---")
        maestro_path = get_tenant_raw_data_dir(tenant) / "maestro.csv"
        artifacts_dir = get_tenant_artifacts_dir(tenant)
        clasificador_dir = artifacts_dir / settings.clasificador_model_name
        homologador_dir = artifacts_dir / settings.homologador_model_name

        maestro_disponible = _warmup_if_available(
            f"maestro.csv tenant={tenant}",
            lambda tenant=tenant: cargar_maestro(tenant),
            maestro_path,
        )

        clasificador_disponible = _warmup_if_available(
            f"modelo clasificador tenant={tenant}",
            lambda tenant=tenant: _load_clasificador_model(tenant),
            clasificador_dir,
        )

        if maestro_disponible and clasificador_disponible:
            _warmup_if_available(
                f"resolvers del clasificador tenant={tenant}",
                lambda tenant=tenant: _get_resolvers(tenant),
            )

        homologador_disponible = _warmup_if_available(
            f"modelo homologador tenant={tenant}",
            lambda tenant=tenant: _load_homologador_model(tenant),
            homologador_dir,
        )

        if maestro_disponible and homologador_disponible:
            _warmup_if_available(
                f"contexto del homologador tenant={tenant}",
                lambda tenant=tenant: _load_homologador_context(tenant),
            )

    print("--- Warm-up finalizado ---")

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
        "tenants_detectados": list_known_tenants(),
    }


app.include_router(datasets_router, prefix=settings.api_v1_prefix)
app.include_router(clasificador_router, prefix=settings.api_v1_prefix)
app.include_router(homologador_router, prefix=settings.api_v1_prefix)
