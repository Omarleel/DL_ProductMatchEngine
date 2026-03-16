from __future__ import annotations

import pandas as pd

from app.core.config import get_settings
from ml_pipeline.clasificador.trainer import entrenar_y_promover_clasificador
from ml_pipeline.utils import init_seeds


def entrenar_modelo(
    epochs: int = 24,
    batch_size: int = 256,
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

    return entrenar_y_promover_clasificador(
        maestro=maestro,
        historial=historial,
        artifacts_dir=settings.artifacts_dir,
        processed_data_dir=settings.processed_data_dir,
        model_name=settings.claisifcador_model_name,
        epochs=epochs,
        batch_size=batch_size,
        force_replace=force_replace,
        run_id=run_id,
    )