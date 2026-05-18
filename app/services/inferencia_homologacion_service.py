from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from io import BytesIO
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.services.maestro_service import cargar_maestro as _load_maestro
from app.schemas.homologador import HomologacionItemRequest
from app.services.tenant_service import (
    DEFAULT_TENANT,
    get_tenant_artifacts_dir,
    get_tenant_results_data_dir,
    normalizar_tenant,
)
from ml_pipeline.homologador import (
    ModeloHomologadorProductos,
    inferir_codproducto_homologador,
)
from ml_pipeline.utils.config import require_file
from ml_pipeline.utils.matching import construir_indice_codigos
from ml_pipeline.utils.preparacion import preparar_maestro


COLUMNAS_RESULTADO_CSV = [
    "RucProveedor",
    "CodFactura",
    "ProductoFactura",
    "UnidadFactura",
    "CostoFactura",
    "ValorTotalFactura",
    "CantidadCompraFactura",
    "UnidadCompraCantidadFactura",
    "CantidadUnidadesFactura",
    "ContenidoTotalLineaFactura",
    "PesoTotalKgFactura",
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


@lru_cache(maxsize=64)
def _load_homologador_model(tenant: str = DEFAULT_TENANT) -> ModeloHomologadorProductos:
    tenant_norm = normalizar_tenant(tenant)
    settings = get_settings()
    model_dir = require_file(
        get_tenant_artifacts_dir(tenant_norm) / settings.homologador_model_name,
        f"directorio del modelo homologador del tenant '{tenant_norm}'",
    )
    return ModeloHomologadorProductos.cargar(model_dir)


@lru_cache(maxsize=64)
def _load_homologador_context(tenant: str = DEFAULT_TENANT) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Prepara y cachea maestro + índice + embeddings por tenant para inferencia."""
    tenant_norm = normalizar_tenant(tenant)
    modelo = _load_homologador_model(tenant_norm)
    maestro_p = preparar_maestro(_load_maestro(tenant_norm)).copy()
    idx = construir_indice_codigos(maestro_p)
    maestro_emb = modelo.encode_prepared_items(maestro_p)
    return maestro_p, idx, maestro_emb


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

    # Mantener compatibilidad con CSVs antiguos: ValorTotal es opcional.
    required_fields = [
        "RucProveedor",
        "CodProducto",
        "Producto",
        "UnidadMedidaCompra",
        "CostoCaja",
    ]
    optional_fields = [
        field for field in HomologacionItemRequest.model_fields.keys()
        if field not in required_fields
    ]

    missing = [field for field in required_fields if field not in df.columns]
    if missing:
        raise ValueError(
            "El CSV no tiene la estructura esperada. "
            f"Faltan las columnas: {missing}. "
            f"Columnas encontradas: {list(df.columns)}. "
            f"Columnas requeridas: {required_fields}. "
            f"Columnas opcionales: {optional_fields}"
        )

    for field in optional_fields:
        if field not in df.columns:
            df[field] = 0.0

    df["RucProveedor"] = df["RucProveedor"].fillna("").astype(str).str.strip()
    df["CodProducto"] = df["CodProducto"].fillna("").astype(str).str.strip()
    df["Producto"] = df["Producto"].fillna("").astype(str).str.strip()
    df["UnidadMedidaCompra"] = df["UnidadMedidaCompra"].fillna("").astype(str).str.strip()
    df["CostoCaja"] = pd.to_numeric(df["CostoCaja"], errors="coerce").fillna(0.0)
    df["ValorTotal"] = pd.to_numeric(df["ValorTotal"], errors="coerce").fillna(0.0)

    output_fields = required_fields + optional_fields

    items: List[dict] = []
    for row in df[output_fields].to_dict(orient="records"):
        item = HomologacionItemRequest.model_validate(row)
        items.append(item.model_dump())

    return items


def homologar_items(
    items: List[dict],
    top_k: int = 5,
    umbral_match: Optional[float] = None,
    top_n_candidates: int = 80,
    guardar_resultado: bool = False,
    tenant: str = DEFAULT_TENANT,
) -> Tuple[List[dict], Optional[str]]:
    if not items:
        return [], None

    tenant_norm = normalizar_tenant(tenant)
    facturas = pd.DataFrame(items)
    resultados_partes = []
    batch_size = 50

    output_path = _build_resultados_csv_path(tenant_norm) if guardar_resultado else None

    try:
        maestro_p, idx, maestro_emb = _load_homologador_context(tenant_norm)
        modelo_homologador = _load_homologador_model(tenant_norm)

        for inicio in range(0, len(facturas), batch_size):
            lote = facturas.iloc[inicio:inicio + batch_size].copy()

            resultado_lote = inferir_codproducto_homologador(
                productos_facturas=lote,
                maestro_p=maestro_p,
                idx=idx,
                maestro_emb=maestro_emb,
                modelo_match=modelo_homologador,
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


def _build_resultados_csv_path(tenant: str = DEFAULT_TENANT) -> Path:
    tenant_norm = normalizar_tenant(tenant)
    results_dir = get_tenant_results_data_dir(tenant_norm)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"homologacion_{tenant_norm}_{timestamp}.csv"


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
