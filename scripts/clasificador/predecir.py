from __future__ import annotations

import argparse

import pandas as pd

from app.services.inferencia_clasificador_service import predecir_desde_items
from app.services.maestro_service import get_facturas_path
from app.services.tenant_service import DEFAULT_TENANT, get_tenant_results_data_dir, normalizar_tenant


ARCHIVO_RESULTADO = "predichos_modelo_clasificador_productos_v1.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predice atributos del clasificador para un tenant.")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant/sucursal a usar.")
    parser.add_argument("--guardar-csv", action="store_true", help="Guarda el CSV de salida. Por defecto no guarda.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tenant = normalizar_tenant(args.tenant)

    nuevas_path = get_facturas_path(tenant)
    nuevas = pd.read_csv(nuevas_path, encoding="utf-8-sig", sep=None, engine="python")

    resultados = predecir_desde_items(
        items=nuevas.to_dict(orient="records"),
        tenant=tenant,
        include_factor_debug=False,
    )
    resultado = pd.DataFrame(resultados).fillna("")

    print("\n--- PREDICCIÓN DE ATRIBUTOS V2 FINALIZADA ---")
    print(f"Tenant: {tenant}")
    if args.guardar_csv:
        out_path = get_tenant_results_data_dir(tenant) / ARCHIVO_RESULTADO
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
        print(f"Archivo generado: '{out_path}'")
    else:
        print("CSV no guardado. Usa --guardar-csv para persistir resultados.")
    print(resultado.head(10))


if __name__ == "__main__":
    main()
