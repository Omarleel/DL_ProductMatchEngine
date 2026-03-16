from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_pipeline.utils.retraining import (
    MetricSpec,
    TrainingLock,
    build_run_id,
    compare_metric_reports,
    promote_candidate_if_needed,
    save_candidate_artifacts,
)

from . import ModeloClasificadorProductos, construir_dataset_clasificador


CLASIFICADOR_METRIC_SPECS: list[MetricSpec] = [
    MetricSpec("metrics_valid.target_categoria_f1_macro", "max", "categoria_f1_macro", min_delta=1e-4),
    MetricSpec("metrics_valid.target_marca_f1_macro", "max", "marca_f1_macro", min_delta=1e-4),
    MetricSpec("metrics_valid.target_categoria_acc", "max", "categoria_acc", min_delta=1e-4),
    MetricSpec("metrics_valid.target_marca_acc", "max", "marca_acc", min_delta=1e-4),
    MetricSpec("metrics_valid.target_factor_conversion_rmse", "min", "factor_conversion_rmse", min_delta=1e-4),
    MetricSpec("metrics_valid.target_factor_venta_rmse", "min", "factor_venta_rmse", min_delta=1e-4),
    MetricSpec("metrics_valid.target_peso_unitario_kg_rmse", "min", "peso_unitario_rmse", min_delta=1e-4),
    MetricSpec("metrics_valid.target_peso_caja_kg_rmse", "min", "peso_caja_rmse", min_delta=1e-4),
]


def entrenar_y_promover_clasificador(
    *,
    maestro: pd.DataFrame,
    historial: pd.DataFrame,
    artifacts_dir: Path,
    processed_data_dir: Path,
    model_name: str,
    epochs: int = 24,
    batch_size: int = 256,
    usar_maestro_como_ejemplos: bool = True,
    force_replace: bool = False,
    run_id: str | None = None,
) -> dict:
    artifacts_dir = Path(artifacts_dir)
    processed_data_dir = Path(processed_data_dir)
    target_dir = artifacts_dir / model_name
    run_id = run_id or build_run_id("retrain_clasificador")

    with TrainingLock(artifacts_dir / ".locks" / f"{model_name}.lock"):
        dataset, category_lexicon = construir_dataset_clasificador(
            maestro=maestro,
            productos_facturas=historial,
            usar_maestro_como_ejemplos=usar_maestro_como_ejemplos,
        )

        processed_data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = processed_data_dir / "dataset_clasificador_productos_v1.csv"
        dataset.to_csv(dataset_path, sep=";", index=False, encoding="utf-8-sig")

        split_helper = ModeloClasificadorProductos()
        train_df, valid_df = split_helper.split_train_valid(dataset, test_size=0.2, random_state=42)

        candidate_model = ModeloClasificadorProductos()
        candidate_report = candidate_model.fit_on_split(
            train_df=train_df,
            valid_df=valid_df,
            category_lexicon=category_lexicon,
            epochs=epochs,
            batch_size=batch_size,
        )

        incumbent_report_current = None
        if target_dir.exists():
            incumbent_model = ModeloClasificadorProductos.cargar(target_dir)
            incumbent_metrics = incumbent_model.evaluate(valid_df, batch_size=batch_size)
            incumbent_report_current = {"metrics_valid": incumbent_metrics}

        decision = compare_metric_reports(
            candidate_report=candidate_report,
            incumbent_report=incumbent_report_current,
            metric_specs=CLASIFICADOR_METRIC_SPECS,
            force_replace=force_replace,
        )

        enriched_report = {
            **candidate_report,
            "run_id": run_id,
            "model_name": model_name,
            "comparison_target": "validacion_actual",
            "incumbent_metrics_current_valid": incumbent_report_current,
            "promotion_decision": decision,
        }

        candidate_dir = save_candidate_artifacts(
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            run_id=run_id,
            save_model_fn=candidate_model.guardar,
            report=enriched_report,
        )

        promotion = promote_candidate_if_needed(
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            candidate_dir=candidate_dir,
            decision=decision,
        )

        return {
            "message": "Entrenamiento finalizado",
            "run_id": run_id,
            "dataset_path": str(dataset_path),
            "artifact_dir": promotion["champion_dir"] if promotion["promoted"] else promotion["candidate_dir"],
            "promoted": promotion["promoted"],
            "promotion_reason": promotion["reason"],
            "candidate_dir": promotion["candidate_dir"],
            "champion_dir": promotion["champion_dir"],
            "archived_previous_dir": promotion["archived_previous_dir"],
            "metrics_valid": candidate_report.get("metrics_valid", {}),
            "incumbent_metrics_current_valid": incumbent_report_current.get("metrics_valid", {}) if incumbent_report_current else None,
            "comparison": decision,
        }