from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from io import BytesIO
from typing import Optional, Tuple, List
from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.schemas.homologador import HomologacionItemRequest
from ml_pipeline.homologador import (
    ModeloHomologadorProductos,
    inferir_codproducto_homologador,
)
from ml_pipeline.utils.config import require_file


COLUMNAS_RESULTADO_CSV = [
    "RucProveedor",
    "CodFactura",
    "ProductoFactura",
    "UnidadFactura",
    "CostoFactura",
    "CodProducto",
    "Producto",
    "Marca",
    "Categoria",
    "UnidadMedidaCompra",
    "FactorVenta",
    "TipoResultado",
    "ScoreFinal",
    "Rank",
]


@lru_cache(maxsize=1)
def _load_maestro() -> pd.DataFrame:
    settings = get_settings()
    path = require_file(settings.raw_data_dir / "maestro.csv", "maestro.csv")
    return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")


@lru_cache(maxsize=1)
def _load_homologador_model() -> ModeloHomologadorProductos:
    settings = get_settings()
    model_dir = require_file(
        settings.artifacts_dir / settings.homologador_model_name,
        "directorio del modelo homologador",
    )
    return ModeloHomologadorProductos.cargar(model_dir)


def _filtrar_columnas_resultado(resultado: pd.DataFrame) -> pd.DataFrame:
    columnas_existentes = [col for col in COLUMNAS_RESULTADO_CSV if col in resultado.columns]
    if not columnas_existentes:
        return resultado.copy()
    return resultado[columnas_existentes].copy()


def cargar_items_desde_csv(csv_bytes: bytes) -> List[dict]:
    try:
        df = pd.read_csv(
            BytesIO(csv_bytes),
            encoding="utf-8-sig",
            sep=None,
            engine="python",
            dtype={
                "RucProveedor": "string",
                "CodProducto": "string",
                "Producto": "string",
                "UnidadMedidaCompra": "string",
            },
        )
    except Exception as exc:
        raise ValueError(f"No se pudo leer el archivo CSV: {exc}") from exc

    if df.empty:
        return []
    
    df.columns = df.columns.astype(str).str.strip()

    required_fields = list(HomologacionItemRequest.model_fields.keys())
    missing = [field for field in required_fields if field not in df.columns]
    if missing:
        raise ValueError(
            "El CSV no tiene la estructura esperada. "
            f"Faltan las columnas: {missing}. "
            f"Columnas encontradas: {list(df.columns)}. "
            f"Columnas esperadas: {required_fields}"
        )

    df["RucProveedor"] = df["RucProveedor"].fillna("").astype(str).str.strip()
    df["CodProducto"] = df["CodProducto"].fillna("").astype(str).str.strip()
    df["Producto"] = df["Producto"].fillna("").astype(str).str.strip()
    df["UnidadMedidaCompra"] = df["UnidadMedidaCompra"].fillna("").astype(str).str.strip()
    df["CostoCaja"] = pd.to_numeric(df["CostoCaja"], errors="coerce").fillna(0.0)

    items: List[dict] = []
    for row in df[required_fields].to_dict(orient="records"):
        item = HomologacionItemRequest.model_validate(row)
        items.append(item.model_dump())

    return items

def homologar_items(
    items: List[dict],
    top_k: int = 5,
    umbral_match: Optional[float] = None,
    top_n_candidates: int = 80,
    guardar_resultado: bool = True,
) -> Tuple[List[dict], Optional[str]]:
    if not items:
        return [], None

    facturas = pd.DataFrame(items)
    resultados_partes = []
    batch_size = 50

    output_path = _build_resultados_csv_path() if guardar_resultado else None

    try:
        for inicio in range(0, len(facturas), batch_size):
            lote = facturas.iloc[inicio:inicio + batch_size].copy()

            resultado_lote = inferir_codproducto_homologador(
                productos_facturas=lote,
                maestro=_load_maestro(),
                modelo_match=_load_homologador_model(),
                top_k=top_k,
                umbral_match=umbral_match,
                top_n_candidates=top_n_candidates,
            )

            resultado_lote = resultado_lote.fillna("")
            resultados_partes.append(resultado_lote)

            if output_path is not None:
                _append_resultados_csv(resultado_lote, output_path)

    except Exception as exc:
        if output_path is not None:
            raise RuntimeError(
                "La homologación falló parcialmente. "
                f"Se guardaron resultados parciales en: {output_path}. "
                f"Detalle: {exc}"
            ) from exc
        raise

    resultado = pd.concat(resultados_partes, ignore_index=True).fillna("")
    return resultado.to_dict(orient="records"), (str(output_path) if output_path else None)

def _build_resultados_csv_path() -> Path:
    settings = get_settings()
    settings.results_data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return settings.results_data_dir / f"homologacion_{timestamp}.csv"


def _append_resultados_csv(resultado_lote: pd.DataFrame, output_path: Path) -> None:
    resultado_csv = _filtrar_columnas_resultado(resultado_lote).fillna("")

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, mode="a", encoding="utf-8-sig", newline="") as f:
        resultado_csv.to_csv(
            f,
            sep=";",
            index=False,
            header=write_header,
        )
        f.flush()