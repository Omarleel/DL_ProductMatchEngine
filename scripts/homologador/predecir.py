from __future__ import annotations

import argparse

import pandas as pd

from app.services.inferencia_homologacion_service import homologar_items
from app.services.maestro_service import get_facturas_path
from app.services.tenant_service import DEFAULT_TENANT, normalizar_tenant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Homologa productos de facturas contra el maestro de un tenant.")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant/sucursal a usar.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-n-candidates", type=int, default=80)
    parser.add_argument("--umbral-match", type=float, default=None)
    parser.add_argument("--guardar-csv", action="store_true", help="Guarda el CSV de salida. Por defecto no guarda.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tenant = normalizar_tenant(args.tenant)

    nuevas_path = get_facturas_path(tenant)
    nuevas = pd.read_csv(nuevas_path, encoding="utf-8-sig", sep=None, engine="python")

    resultados, output_csv = homologar_items(
        items=nuevas.to_dict(orient="records"),
        tenant=tenant,
        top_k=args.top_k,
        umbral_match=args.umbral_match,
        top_n_candidates=args.top_n_candidates,
        guardar_resultado=args.guardar_csv,
    )
    resultado = pd.DataFrame(resultados).fillna("")

    print("\n--- PREDICCIÓN MODELO HOMOLOGADOR FINALIZADA ---")
    print(f"Tenant: {tenant}")
    if output_csv:
        print(f"Archivo generado: '{output_csv}'")
    else:
        print("CSV no guardado. Usa --guardar-csv para persistir resultados.")
    print(resultado.head(10))


if __name__ == "__main__":
    main()
