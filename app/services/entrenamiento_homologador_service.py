from __future__ import annotations

import pandas as pd

from app.core.config import get_settings
from app.services.maestro_service import cargar_maestro, get_facturas_path, get_maestro_path
from app.services.tenant_service import (
    DEFAULT_TENANT,
    get_tenant_artifacts_dir,
    get_tenant_processed_data_dir,
    normalizar_tenant,
)
from ml_pipeline.homologador.trainer import entrenar_y_promover_homologador
from ml_pipeline.utils import init_seeds
from ml_pipeline.utils.training_logger import TrainingLogger


logger = TrainingLogger("homologador")


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
    tenant: str = DEFAULT_TENANT,
) -> dict:
    tenant_norm = normalizar_tenant(tenant)
    logger.info(
        "Inicio entrenamiento tenant=%s run_id=%s epochs_warmup=%s epochs_final=%s batch_size=%s "
        "top_n_candidates=%s k_hard_per_positive=%s force_replace=%s",
        tenant_norm,
        run_id,
        epochs_warmup,
        epochs_final,
        batch_size,
        top_n_candidates,
        k_hard_per_positive,
        force_replace,
    )
    try:
        init_seeds()
        settings = get_settings()

        maestro_path = get_maestro_path(tenant_norm)
        historial_path = get_facturas_path(tenant_norm)

        logger.info("Leyendo maestro tenant=%s: %s", tenant_norm, maestro_path)
        maestro = cargar_maestro(tenant_norm)
        logger.info("Maestro cargado: rows=%s cols=%s", len(maestro), len(maestro.columns))

        logger.info("Leyendo historial tenant=%s: %s", tenant_norm, historial_path)
        historial = pd.read_csv(
            historial_path,
            encoding="utf-8-sig",
            sep=None,
            engine="python",
        )
        logger.info("Historial cargado: rows=%s cols=%s", len(historial), len(historial.columns))

        result = entrenar_y_promover_homologador(
            maestro=maestro,
            historial=historial,
            artifacts_dir=get_tenant_artifacts_dir(tenant_norm),
            processed_data_dir=get_tenant_processed_data_dir(tenant_norm),
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
        result["tenant"] = tenant_norm
        _limpiar_cache_inferencia_homologador()
        logger.info(
            "FIN OK tenant=%s run_id=%s promoted=%s artifact_dir=%s",
            tenant_norm,
            run_id,
            result.get("promoted"),
            result.get("artifact_dir"),
        )
        return result
    except Exception as exc:
        logger.exception("ERROR tenant=%s run_id=%s: %s: %s", tenant_norm, run_id, type(exc).__name__, exc)
        raise


def _limpiar_cache_inferencia_homologador() -> None:
    try:
        from app.services import inferencia_homologacion_service as inferencia

        inferencia._load_homologador_model.cache_clear()
        inferencia._load_homologador_context.cache_clear()
    except Exception:
        # El entrenamiento ya terminó; no conviene fallar solo por limpieza de cache.
        pass
