from __future__ import annotations

import argparse
import json

from app.services.entrenamiento_clasificador_service import entrenar_modelo
from app.services.tenant_service import DEFAULT_TENANT, normalizar_tenant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reentrena el clasificador para un tenant y solo promueve el candidato si mejora al campeón."
    )
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant/sucursal a entrenar.")
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
    tenant = normalizar_tenant(args.tenant)

    result = entrenar_modelo(
        tenant=tenant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_replace=args.force_replace,
    )

    print("\n--- REENTRENAMIENTO CLASIFICADOR FINALIZADO ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["promoted"]:
        print(f"✅ Nuevo campeón promovido para tenant={tenant}: {result['champion_dir']}")
    else:
        print(f"ℹ️ El campeón anterior se conserva para tenant={tenant}. Candidato archivado en: {result['candidate_dir']}")


if __name__ == "__main__":
    main()
