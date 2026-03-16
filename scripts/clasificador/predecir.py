from __future__ import annotations

import pandas as pd

from ml_pipeline.clasificador import ModeloClasificadorProductos, inferir_atributos_producto
from ml_pipeline.utils import init_seeds
from ml_pipeline.utils.config import dataset_path, ensure_project_dirs, model_path, require_file, result_path


MODELO_NOMBRE = "clasificador_productos_v1"
ARCHIVO_MAESTRO = "maestro.csv"
ARCHIVO_FACTURAS_NUEVAS = "productos_facturas.csv"
ARCHIVO_RESULTADO = "predichos_modelo_clasificador_productos_v1.csv"


def main() -> None:
    init_seeds()
    ensure_project_dirs()

    maestro_path = require_file(dataset_path(ARCHIVO_MAESTRO), "dataset maestro")
    nuevas_path = require_file(dataset_path(ARCHIVO_FACTURAS_NUEVAS), "dataset productos_facturas")
    ruta_modelo = require_file(model_path(MODELO_NOMBRE), "directorio del modelo")

    maestro = pd.read_csv(maestro_path, encoding="utf-8-sig", sep=None, engine="python")
    nuevas = pd.read_csv(nuevas_path, encoding="utf-8-sig", sep=None, engine="python")
    modelo = ModeloClasificadorProductos.cargar(ruta_modelo)

    resultado = inferir_atributos_producto(
        productos_facturas=nuevas,
        modelo=modelo,
        maestro=maestro,
        batch_size=512,
        include_factor_debug=False,
    )
    out_path = result_path(ARCHIVO_RESULTADO)
    resultado.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")

    print("\n--- PREDICCIÓN DE ATRIBUTOS V2 FINALIZADA ---")
    print(f"Archivo generado: '{out_path}'")
    print(resultado.head(10))


if __name__ == "__main__":
    main()
