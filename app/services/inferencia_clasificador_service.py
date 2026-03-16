from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.core.config import get_settings
from ml_pipeline.clasificador import (
    inferir_atributos_producto,
    ModeloClasificadorProductos
)
from ml_pipeline.clasificador.factor_resolver import MaestroFactorResolver
from ml_pipeline.clasificador.weight_resolver import MaestroWeightResolver
from ml_pipeline.clasificador.brand_category_resolver import MaestroBrandCategoryResolver
from ml_pipeline.utils.config import require_file


@lru_cache(maxsize=1)
def _load_maestro() -> pd.DataFrame:
    settings = get_settings()
    path = require_file(settings.raw_data_dir / "maestro.csv", "maestro.csv")
    return pd.read_csv(path, encoding="utf-8-sig", sep=',', engine="c", low_memory=False)

@lru_cache(maxsize=1)
def _load_model() -> ModeloClasificadorProductos:
    settings = get_settings()
    model_dir = require_file(settings.artifacts_dir / settings.claisifcador_model_name, "directorio del modelo")
    return ModeloClasificadorProductos.cargar(model_dir)

@lru_cache(maxsize=1)
def _get_resolvers():
    """Crea e indexa los resolvers en memoria una sola vez."""
    maestro = _load_maestro()
    print("Indexando Resolvers en memoria...")
    return {
        "factor": MaestroFactorResolver(maestro),
        "weight": MaestroWeightResolver(maestro),
        "brand_cat": MaestroBrandCategoryResolver(maestro, min_score=0.55)
    }


def predecir_desde_items(items: list[dict], include_factor_debug: bool = False) -> list[dict]:
    if not items:
        return []

    df = pd.DataFrame(items)
    
    resultado = inferir_atributos_producto(
        productos_facturas=df,
        modelo=_load_model(),
        resolvers=_get_resolvers(),
        batch_size=32,
        include_factor_debug=include_factor_debug,
        resolver_marca_categoria_desde_maestro=True,
    )
    return resultado.fillna("").to_dict(orient="records")