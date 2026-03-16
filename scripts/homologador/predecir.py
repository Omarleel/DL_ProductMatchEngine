from __future__ import annotations

import pandas as pd

from ml_pipeline.homologador.model import ModeloHomologadorProductos
from ml_pipeline.homologador import inferir_codproducto_homologador
from ml_pipeline.utils import init_seeds
from ml_pipeline.utils.config import dataset_path, ensure_project_dirs, model_path, require_file, result_path


MODELO_NOMBRE = "homologador_productos_v1"
ARCHIVO_MAESTRO = "maestro.csv"
ARCHIVO_FACTURAS_NUEVAS = "productos_facturas.csv"

ARCHIVO_RESULTADO_TECNICO = "resultado_inferencia_homologador.csv"
ARCHIVO_RESULTADO_RESUMIDO = "match_final_resumido_homologador.csv"


def main() -> None:
    init_seeds()
    ensure_project_dirs()

    maestro_path = require_file(dataset_path(ARCHIVO_MAESTRO), "dataset maestro")
    nuevas_path = require_file(dataset_path(ARCHIVO_FACTURAS_NUEVAS), "dataset productos_facturas")
    ruta_modelo = require_file(model_path(MODELO_NOMBRE), "directorio del modelo")

    maestro = pd.read_csv(maestro_path, encoding="utf-8-sig", sep=None, engine="python")
    nuevas = pd.read_csv(nuevas_path, encoding="utf-8-sig", sep=None, engine="python")

    modelo = ModeloHomologadorProductos.cargar(ruta_modelo)
    resultado = inferir_codproducto_homologador(
        productos_facturas=nuevas,
        maestro=maestro,
        modelo_match=modelo,
        top_k=5,
        umbral_match=modelo.best_threshold,
        top_n_candidates=80,
    )

    resultado_path = result_path(ARCHIVO_RESULTADO_TECNICO)
    resultado.to_csv(resultado_path, sep=";", index=False, encoding="utf-8-sig")

    resumido = resultado[resultado["Rank"] == 1].copy()
    columnas_finales = [
        "RucProveedor",
        "CodFactura",
        "ProductoFactura",
        "UnidadFactura",
        "CodProducto",
        "Producto",
        "TipoResultado",
        "Score",
    ]
    resumido = resumido[columnas_finales]
    resumido.columns = [
        "RUC_PROVEEDOR",
        "COD_PRODUCTO_FACTURA",
        "NOMBRE_PRODUCTO_FACTURA",
        "UM_FACTURA",
        "COD_PRODUCTO_SISTEMA",
        "NOMBRE_PRODUCTO_SISTEMA",
        "ESTADO_MATCH",
        "SCORE",
    ]

    resumido_path = result_path(ARCHIVO_RESULTADO_RESUMIDO)
    resumido.to_csv(resumido_path, sep=";", index=False, encoding="utf-8-sig")

    print("\n--- PREDICCIÓN MODELO HOMOLOGADOR FINALIZADA ---")
    print(f"Archivo técnico: '{resultado_path}'")
    print(f"Archivo resumen: '{resumido_path}'")
    print(resumido.head(10))


if __name__ == "__main__":
    main()