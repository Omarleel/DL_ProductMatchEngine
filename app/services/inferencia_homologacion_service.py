from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.core.config import get_settings
from ml_pipeline.homologador import (
    ModeloHomologadorProductos,
    inferir_codproducto_homologador,
)
from ml_pipeline.utils.config import require_file


@lru_cache(maxsize=1)
def _load_maestro() -> pd.DataFrame:
    settings = get_settings()
    path = require_file(settings.raw_data_dir / "maestro.csv", "maestro.csv")
    return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")


@lru_cache(maxsize=1)
def _load_homologador_model() -> ModeloHomologadorProductos:
    settings = get_settings()
    model_dir = require_file(
        settings.artifacts_dir / settings.homologador_model_name,
        "directorio del modelo homologador",
    )
    return ModeloHomologadorProductos.cargar(model_dir)


def homologar_items(
    items: list[dict],
    top_k: int = 5,
    umbral_match: float | None = None,
    top_n_candidates: int = 80,
) -> list[dict]:
    if not items:
        return []

    facturas = pd.DataFrame(items)

    resultado = inferir_codproducto_homologador(
        productos_facturas=facturas,
        maestro=_load_maestro(),
        modelo_match=_load_homologador_model(),
        top_k=top_k,
        umbral_match=umbral_match,
        top_n_candidates=top_n_candidates,
    )

    return resultado.fillna("").to_dict(orient="records")