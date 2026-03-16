from __future__ import annotations

import argparse
import json

import pandas as pd

from ml_pipeline.clasificador.trainer import entrenar_y_promover_clasificador
from ml_pipeline.utils import init_seeds
from ml_pipeline.utils.config import (
    dataset_path,
    ensure_project_dirs,
    require_file,
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
)


ARCHIVO_MAESTRO = "maestro.csv"
ARCHIVO_HISTORIAL_FACTURAS = "productos_facturas.csv"
MODELO_NOMBRE = "clasificador_productos_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reentrena el clasificador y solo promueve el candidato si mejora al campeón."
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--force-replace",
        action="store_true",
        help="Promueve el candidato aunque no mejore al campeón.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    init_seeds()
    ensure_project_dirs()

    maestro_path = require_file(dataset_path(ARCHIVO_MAESTRO), "dataset maestro")
    historial_path = require_file(dataset_path(ARCHIVO_HISTORIAL_FACTURAS), "dataset productos_facturas")

    maestro = pd.read_csv(maestro_path, encoding="utf-8-sig", sep=None, engine="python")
    historial = pd.read_csv(historial_path, encoding="utf-8-sig", sep=None, engine="python")

    result = entrenar_y_promover_clasificador(
        maestro=maestro,
        historial=historial,
        artifacts_dir=ARTIFACTS_DIR,
        processed_data_dir=PROCESSED_DATA_DIR,
        model_name=MODELO_NOMBRE,
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_replace=args.force_replace,
    )

    print("\n--- REENTRENAMIENTO CLASIFICADOR FINALIZADO ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["promoted"]:
        print(f"✅ Nuevo campeón promovido: {result['champion_dir']}")
    else:
        print(f"ℹ️ El campeón anterior se conserva. Candidato archivado en: {result['candidate_dir']}")


if __name__ == "__main__":
    main()