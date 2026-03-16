from __future__ import annotations

import pandas as pd

from app.core.config import get_settings
from ml_pipeline.homologador.trainer import entrenar_y_promover_homologador
from ml_pipeline.utils import init_seeds


def entrenar_modelo_homologador(
    n_neg_por_pos: int = 4,
    epochs_warmup: int = 10,
    epochs_final: int = 16,
    batch_size: int = 256,
    top_n_candidates: int = 60,
    k_hard_per_positive: int = 2,
    min_model_score: float = 0.25,
    min_support: float = 0.08,
    force_replace: bool = False,
    run_id: str | None = None,
) -> dict:
    init_seeds()
    settings = get_settings()

    maestro = pd.read_csv(
        settings.raw_data_dir / "maestro.csv",
        encoding="utf-8-sig",
        sep=None,
        engine="python",
    )
    historial = pd.read_csv(
        settings.raw_data_dir / "productos_facturas.csv",
        encoding="utf-8-sig",
        sep=None,
        engine="python",
    )

    return entrenar_y_promover_homologador(
        maestro=maestro,
        historial=historial,
        artifacts_dir=settings.artifacts_dir,
        processed_data_dir=settings.processed_data_dir,
        model_name=settings.homologador_model_name,
        n_neg_por_pos=n_neg_por_pos,
        epochs_warmup=epochs_warmup,
        epochs_final=epochs_final,
        batch_size=batch_size,
        top_n_candidates=top_n_candidates,
        k_hard_per_positive=k_hard_per_positive,
        min_model_score=min_model_score,
        min_support=min_support,
        force_replace=force_replace,
        run_id=run_id,
    )