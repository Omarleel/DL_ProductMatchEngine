from __future__ import annotations

from functools import lru_cache
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from app.services.tenant_service import (
    DEFAULT_TENANT,
    get_tenant_raw_data_dir,
    normalizar_tenant,
)
from ml_pipeline.utils.config import require_file
from ml_pipeline.utils.maestro import MAESTRO_FILENAME, leer_maestro_csv


FACTURAS_FILENAME = "productos_facturas.csv"


def get_maestro_path(tenant: Optional[str] = DEFAULT_TENANT) -> Path:
    """Ruta única del maestro usado por clasificador y homologador para un tenant."""
    tenant_norm = normalizar_tenant(tenant)
    return require_file(
        get_tenant_raw_data_dir(tenant_norm) / MAESTRO_FILENAME,
        f"{MAESTRO_FILENAME} del tenant '{tenant_norm}'",
    )


def get_facturas_path(tenant: Optional[str] = DEFAULT_TENANT) -> Path:
    """Ruta única del histórico/productos de facturas para un tenant."""
    tenant_norm = normalizar_tenant(tenant)
    return require_file(
        get_tenant_raw_data_dir(tenant_norm) / FACTURAS_FILENAME,
        f"{FACTURAS_FILENAME} del tenant '{tenant_norm}'",
    )


@lru_cache(maxsize=64)
def cargar_maestro(tenant: Optional[str] = DEFAULT_TENANT) -> pd.DataFrame:
    """Carga maestro.csv por tenant, cacheado e independiente entre sucursales."""
    tenant_norm = normalizar_tenant(tenant)
    return leer_maestro_csv(get_maestro_path(tenant_norm))


def limpiar_cache_maestro(tenant: Optional[str] = None) -> None:
    """
    Refresca el maestro y caches derivados.

    functools.lru_cache no permite invalidar una sola clave de forma pública, por eso se limpian
    todos los tenants. Es seguro y evita mezclar índices/modelos cacheados con datasets nuevos.
    """
    cargar_maestro.cache_clear()

    cacheados_derivados = {
        "app.services.inferencia_clasificador_service": ["_get_resolvers"],
        "app.services.inferencia_homologacion_service": ["_load_homologador_context"],
    }
    for module_name, function_names in cacheados_derivados.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for function_name in function_names:
            function = getattr(module, function_name, None)
            cache_clear = getattr(function, "cache_clear", None)
            if cache_clear is not None:
                cache_clear()
