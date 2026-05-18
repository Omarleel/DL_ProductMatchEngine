from __future__ import annotations

import argparse
import json

from app.services.entrenamiento_homologador_service import entrenar_modelo_homologador
from app.services.tenant_service import DEFAULT_TENANT, normalizar_tenant
from ml_pipeline.utils.training_logger import TrainingLogger


logger = TrainingLogger("homologador")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reentrena el homologador para un tenant y solo promueve el candidato si mejora al campeón."
    )
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant/sucursal a entrenar.")
    parser.add_argument("--n-neg-por-pos", type=int, default=4)
    parser.add_argument(
        "--auto_match",
        action="store_true",
        help="Usa compras/CPE históricos para construir positivos cuando el código CPE no está en el maestro.",
    )
    parser.add_argument("--epochs-warmup", type=int, default=10)
    parser.add_argument("--epochs-final", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-n-candidates", type=int, default=60)
    parser.add_argument("--k-hard-per-positive", type=int, default=2)
    parser.add_argument("--min-model-score", type=float, default=0.25)
    parser.add_argument("--min-support", type=float, default=0.08)
    parser.add_argument(
        "--force-replace",
        action="store_true",
        help="Promueve el candidato aunque no mejore al campeón.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tenant = normalizar_tenant(args.tenant)

    result = entrenar_modelo_homologador(
        tenant=tenant,
        n_neg_por_pos=args.n_neg_por_pos,
        auto_match=args.auto_match,
        epochs_warmup=args.epochs_warmup,
        epochs_final=args.epochs_final,
        batch_size=args.batch_size,
        top_n_candidates=args.top_n_candidates,
        k_hard_per_positive=args.k_hard_per_positive,
        min_model_score=args.min_model_score,
        min_support=args.min_support,
        force_replace=args.force_replace,
    )

    logger.info("REENTRENAMIENTO HOMOLOGADOR FINALIZADO")
    logger.info("Resultado:\n%s", json.dumps(result, ensure_ascii=False, indent=2))
    if result["promoted"]:
        logger.info("Nuevo campeón promovido para tenant=%s: %s", tenant, result["champion_dir"])
    else:
        logger.info("El campeón anterior se conserva para tenant=%s. Candidato archivado en: %s", tenant, result["candidate_dir"])


if __name__ == "__main__":
    main()
