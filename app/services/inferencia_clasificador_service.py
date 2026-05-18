from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.core.config import get_settings
from app.services.maestro_service import cargar_maestro as _load_maestro
from app.services.tenant_service import DEFAULT_TENANT, get_tenant_artifacts_dir, normalizar_tenant
from ml_pipeline.clasificador import (
    inferir_atributos_producto,
    ModeloClasificadorProductos,
)
from ml_pipeline.clasificador.factor_resolver import MaestroFactorResolver
from ml_pipeline.clasificador.weight_resolver import MaestroWeightResolver
from ml_pipeline.clasificador.brand_category_resolver import MaestroBrandCategoryResolver
from ml_pipeline.utils.config import require_file


@lru_cache(maxsize=64)
def _load_model(tenant: str = DEFAULT_TENANT) -> ModeloClasificadorProductos:
    tenant_norm = normalizar_tenant(tenant)
    settings = get_settings()
    model_dir = require_file(
        get_tenant_artifacts_dir(tenant_norm) / settings.clasificador_model_name,
        f"directorio del modelo clasificador del tenant '{tenant_norm}'",
    )
    return ModeloClasificadorProductos.cargar(model_dir)


@lru_cache(maxsize=64)
def _get_resolvers(tenant: str = DEFAULT_TENANT):
    """Crea e indexa los resolvers en memoria por tenant."""
    tenant_norm = normalizar_tenant(tenant)
    maestro = _load_maestro(tenant_norm)
    print(f"Indexando resolvers en memoria para tenant={tenant_norm}...")
    return {
        "factor": MaestroFactorResolver(maestro),
        "weight": MaestroWeightResolver(maestro),
        "brand_cat": MaestroBrandCategoryResolver(maestro, min_score=0.55),
    }


def predecir_desde_items(
    items: list[dict],
    include_factor_debug: bool = False,
    tenant: str = DEFAULT_TENANT,
) -> list[dict]:
    if not items:
        return []

    tenant_norm = normalizar_tenant(tenant)
    df = pd.DataFrame(items)

    resultado = inferir_atributos_producto(
        productos_facturas=df,
        modelo=_load_model(tenant_norm),
        resolvers=_get_resolvers(tenant_norm),
        batch_size=32,
        include_factor_debug=include_factor_debug,
        resolver_marca_categoria_desde_maestro=True,
    )
    return resultado.fillna("").to_dict(orient="records")
