from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_pipeline.utils import construir_dataset_entrenamiento
from ml_pipeline.utils.retraining import (
    MetricSpec,
    TrainingLock,
    build_run_id,
    compare_metric_reports,
    promote_candidate_if_needed,
    save_candidate_artifacts,
)

from . import mine_hard_negatives
from .model import ModeloHomologadorProductos


HOMOLOGADOR_METRIC_SPECS: list[MetricSpec] = [
    MetricSpec("ranking_metrics_valid.hit_at_1", "max", "hit_at_1", min_delta=1e-4),
    MetricSpec("pair_metrics_valid.pr_auc", "max", "pr_auc", min_delta=1e-4),
    MetricSpec("best_f1_valid", "max", "best_f1_valid", min_delta=1e-4),
    MetricSpec("ranking_metrics_valid.hit_at_3", "max", "hit_at_3", min_delta=1e-4),
    MetricSpec("pair_metrics_valid.roc_auc", "max", "roc_auc", min_delta=1e-4),
]


def entrenar_y_promover_homologador(
    *,
    maestro: pd.DataFrame,
    historial: pd.DataFrame,
    artifacts_dir: Path,
    processed_data_dir: Path,
    model_name: str,
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
    artifacts_dir = Path(artifacts_dir)
    processed_data_dir = Path(processed_data_dir)
    target_dir = artifacts_dir / model_name
    run_id = run_id or build_run_id("retrain_homologador")

    with TrainingLock(artifacts_dir / ".locks" / f"{model_name}.lock"):
        pares_base = construir_dataset_entrenamiento(
            maestro=maestro,
            productos_facturas=historial,
            n_neg_por_pos=n_neg_por_pos,
        )

        processed_data_dir.mkdir(parents=True, exist_ok=True)
        pares_base_path = processed_data_dir / "pares_entrenamiento_homologador.csv"
        pares_base.to_csv(pares_base_path, sep=";", index=False, encoding="utf-8-sig")

        split_helper = ModeloHomologadorProductos()
        train_base, valid_base = split_helper.split_train_valid(
            pares_base,
            test_size=0.2,
            random_state=42,
        )

        warmup_model = ModeloHomologadorProductos()
        warmup_report = warmup_model.fit_on_split(
            train_df=train_base,
            valid_df=valid_base,
            epochs=epochs_warmup,
            batch_size=batch_size,
        )

        pares_hard = mine_hard_negatives(
            modelo=warmup_model,
            maestro=maestro,
            pares_base=train_base,
            top_n_candidates=top_n_candidates,
            k_hard_per_positive=k_hard_per_positive,
            min_model_score=min_model_score,
            min_support=min_support,
            random_state=42,
        )

        pares_hard_path = None
        if not pares_hard.empty:
            pares_hard_path = processed_data_dir / "pares_hard_negatives_homologador.csv"
            pares_hard.to_csv(pares_hard_path, sep=";", index=False, encoding="utf-8-sig")
            train_final = pd.concat([train_base, pares_hard], ignore_index=True)
            train_final = train_final.drop_duplicates(
                subset=["RucProveedor", "fact_cod", "master_cod", "label"],
                keep="first",
            ).reset_index(drop=True)
        else:
            train_final = train_base.copy()

        train_final_path = processed_data_dir / "pares_entrenamiento_homologador_final.csv"
        train_final.to_csv(train_final_path, sep=";", index=False, encoding="utf-8-sig")

        final_model = ModeloHomologadorProductos()
        final_report = final_model.fit_on_split(
            train_df=train_final,
            valid_df=valid_base,
            epochs=epochs_final,
            batch_size=batch_size,
        )

        incumbent_report_current = None
        if target_dir.exists():
            incumbent_model = ModeloHomologadorProductos.cargar(target_dir)
            incumbent_pair_metrics = incumbent_model.evaluate_pairs(valid_base, batch_size=batch_size)
            incumbent_ranking_metrics = incumbent_model.evaluate_ranking(valid_base, batch_size=batch_size)
            incumbent_report_current = {
                "best_threshold": float(incumbent_model.best_threshold),
                "best_f1_valid": float(incumbent_pair_metrics.get("best_f1_eval", 0.0)),
                "pair_metrics_valid": incumbent_pair_metrics,
                "ranking_metrics_valid": incumbent_ranking_metrics,
            }

        decision = compare_metric_reports(
            candidate_report=final_report,
            incumbent_report=incumbent_report_current,
            metric_specs=HOMOLOGADOR_METRIC_SPECS,
            force_replace=force_replace,
        )

        report = {
            "run_id": run_id,
            "model_name": model_name,
            "comparison_target": "validacion_actual",
            "base_rows": int(len(pares_base)),
            "train_base_rows": int(len(train_base)),
            "valid_base_rows": int(len(valid_base)),
            "hard_negative_rows": int(len(pares_hard)),
            "train_final_rows": int(len(train_final)),
            "warmup_report": warmup_report,
            "final_report": final_report,
            "incumbent_metrics_current_valid": incumbent_report_current,
            "promotion_decision": decision,
        }

        candidate_dir = save_candidate_artifacts(
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            run_id=run_id,
            save_model_fn=final_model.guardar,
            report=report,
        )

        promotion = promote_candidate_if_needed(
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            candidate_dir=candidate_dir,
            decision=decision,
        )

        return {
            "message": "Entrenamiento Homologador finalizado con éxito.",
            "run_id": run_id,
            "artifact_dir": promotion["champion_dir"] if promotion["promoted"] else promotion["candidate_dir"],
            "promoted": promotion["promoted"],
            "promotion_reason": promotion["reason"],
            "candidate_dir": promotion["candidate_dir"],
            "champion_dir": promotion["champion_dir"],
            "archived_previous_dir": promotion["archived_previous_dir"],
            "pairs_base_path": str(pares_base_path),
            "pairs_hard_path": str(pares_hard_path) if pares_hard_path else None,
            "pairs_final_path": str(train_final_path),
            "final_report": final_report,
            "incumbent_metrics_current_valid": incumbent_report_current,
            "comparison": decision,
        }