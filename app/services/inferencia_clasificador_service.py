from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.core.config import get_settings
from ml_pipeline.clasificador import ModeloClasificadorProductos, inferir_atributos_producto
from ml_pipeline.utils.config import require_file


@lru_cache(maxsize=1)
def _load_maestro() -> pd.DataFrame:
    settings = get_settings()
    path = require_file(settings.raw_data_dir / "maestro.csv", "maestro.csv")
    return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")


@lru_cache(maxsize=1)
def _load_model() -> ModeloClasificadorProductos:
    settings = get_settings()
    model_dir = require_file(settings.artifacts_dir / settings.claisifcador_model_name, "directorio del modelo")
    return ModeloClasificadorProductos.cargar(model_dir)


def predecir_desde_items(items: list[dict], include_factor_debug: bool = False) -> list[dict]:
    if not items:
        return []

    df = pd.DataFrame(items)

    resultado = inferir_atributos_producto(
        productos_facturas=df,
        modelo=_load_model(),
        maestro=_load_maestro(),
        batch_size=512,
        include_factor_debug=include_factor_debug,
        resolver_marca_categoria_desde_maestro=True,
    )
    return resultado.fillna("").to_dict(orient="records")