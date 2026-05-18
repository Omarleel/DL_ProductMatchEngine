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
    get_tenant_processed_data_dir,
    get_tenant_raw_data_dir,
    get_tenant_results_data_dir,
    normalizar_tenant,
)
from ml_pipeline.homologador import (
    ModeloHomologadorProductos,
    inferir_codproducto_homologador,
)
from ml_pipeline.utils.config import require_file
from ml_pipeline.utils.limpieza import normalizar_codigo, normalizar_unidad
from ml_pipeline.utils.matching import construir_indice_codigos
from ml_pipeline.utils.preparacion import preparar_maestro


COLUMNAS_RESULTADO_CSV = [
    "RucProveedor",
    "CodFactura",
    "ProductoFactura",
    "UnidadFactura",
    "CostoFactura",
    "ValorTotalFactura",
    "CantidadFactura",
    "CantidadCompraFactura",
    "UnidadCompraCantidadFactura",
    "CantidadUnidadesFactura",
    "FactorConversionCantidadUsado",
    "ConversionCantidadEncontrada",
    "ConversionCantidadNivel",
    "ConversionCantidadMuestras",
    "ConversionCantidadConfianza",
    "UsoFallbackValorTotal",
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


def _norm_ruc_alias(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _norm_cod_alias(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return normalizar_codigo(text)


@lru_cache(maxsize=64)
def _load_positive_aliases(tenant: str = DEFAULT_TENANT) -> tuple[tuple[str, str, str], ...]:
    """
    Carga equivalencias positivas generadas durante el entrenamiento.
    (Versión Optimizada)
    """
    tenant_norm = normalizar_tenant(tenant)
    processed_dir = get_tenant_processed_data_dir(tenant_norm)
    candidate_paths = [
        processed_dir / "pares_entrenamiento_homologador.csv",
        processed_dir / "pares_entrenamiento_homologador_final.csv",
    ]

    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        return tuple()

    try:
        pares = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=["RucProveedor", "fact_cod", "master_cod", "label"],
            dtype={
                "RucProveedor": "string",
                "fact_cod": "string",
                "master_cod": "string",
            },
            sep=";",
            low_memory=False
        )
    except Exception:
        return tuple()

    required = {"RucProveedor", "fact_cod", "master_cod", "label"}
    if not required.issubset(set(pares.columns)):
        return tuple()

    label_num = pd.to_numeric(pares["label"], errors="coerce").fillna(0).astype(int)
    pos = pares.loc[label_num == 1, ["RucProveedor", "fact_cod", "master_cod"]].copy()
    
    if pos.empty:
        return tuple()

    pos["_ruc"] = pos["RucProveedor"].map(_norm_ruc_alias)
    pos["_fact"] = pos["fact_cod"].map(_norm_cod_alias)
    pos["_master"] = pos["master_cod"].map(_norm_cod_alias)
    
    pos = pos[(pos["_fact"] != "") & (pos["_master"] != "")].drop_duplicates(
        subset=["_ruc", "_fact", "_master"]
    )

    if pos.empty:
        return tuple()

    counts = pos.groupby(["_ruc", "_fact"], dropna=False)["_master"].transform("nunique")
    pos = pos[counts == 1]

    return tuple(pos[["_ruc", "_fact", "_master"]].itertuples(index=False, name=None))

def _norm_unit_alias(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return normalizar_unidad(str(value).strip())


def _safe_float_conversion(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _prefer_conversion_row(current: Optional[dict], candidate: dict) -> dict:
    if current is None:
        return candidate

    current_key = (
        _safe_float_conversion(current.get("MuestrasModa", 0.0)),
        _safe_float_conversion(current.get("ConfianzaModa", 0.0)),
        _safe_float_conversion(current.get("Muestras", 0.0)),
    )
    candidate_key = (
        _safe_float_conversion(candidate.get("MuestrasModa", 0.0)),
        _safe_float_conversion(candidate.get("ConfianzaModa", 0.0)),
        _safe_float_conversion(candidate.get("Muestras", 0.0)),
    )
    return candidate if candidate_key > current_key else current


@lru_cache(maxsize=64)
def _load_quantity_conversion_lookup(tenant: str = DEFAULT_TENANT) -> dict:
    """Carga el diccionario histórico CantidadCPE -> CantidadCompra por tenant."""
    tenant_norm = normalizar_tenant(tenant)
    candidate_paths = [
        get_tenant_raw_data_dir(tenant_norm) / "diccionario_conversion_unidades.csv",
        get_tenant_processed_data_dir(tenant_norm) / "diccionario_conversion_unidades.csv",
    ]
    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        return {}

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            sep=None,
            engine="python",
            dtype={
                "RucProveedor": "string",
                "CodProductoMaestro": "string",
                "CodProductoCpe": "string",
                "UnidadMedidaCpe": "string",
                "UnidadMedidaCompra": "string",
            },
        )
    except Exception:
        return {}

    required = {
        "RucProveedor",
        "CodProductoMaestro",
        "CodProductoCpe",
        "UnidadMedidaCpe",
        "UnidadMedidaCompra",
        "FactorCantidadCompra",
    }
    if df.empty or not required.issubset(set(df.columns)):
        return {}

    indexes = {
        "exact": {},
        "ruc_product": {},
        "global_exact": {},
        "global_product": {},
    }

    for _, raw in df.iterrows():
        factor = _safe_float_conversion(raw.get("FactorCantidadCompra", 0.0))
        if factor <= 0.0:
            continue

        row = {
            "RucProveedor": _norm_ruc_alias(raw.get("RucProveedor", "")),
            "CodProductoMaestro": _norm_cod_alias(raw.get("CodProductoMaestro", "")),
            "CodProductoCpe": _norm_cod_alias(raw.get("CodProductoCpe", "")),
            "UnidadMedidaCpe": _norm_unit_alias(raw.get("UnidadMedidaCpe", "")),
            "UnidadMedidaCompra": _norm_unit_alias(raw.get("UnidadMedidaCompra", "")),
            "FactorCantidadCompra": factor,
            "Muestras": _safe_float_conversion(raw.get("Muestras", 0.0)),
            "MuestrasModa": _safe_float_conversion(raw.get("MuestrasModa", 0.0)),
            "ConfianzaModa": _safe_float_conversion(raw.get("ConfianzaModa", 0.0)),
        }

        if not row["CodProductoMaestro"] or not row["CodProductoCpe"]:
            continue

        exact_key = (
            row["RucProveedor"],
            row["CodProductoMaestro"],
            row["CodProductoCpe"],
            row["UnidadMedidaCpe"],
            row["UnidadMedidaCompra"],
        )
        ruc_product_key = (
            row["RucProveedor"],
            row["CodProductoMaestro"],
            row["CodProductoCpe"],
        )
        global_exact_key = (
            row["CodProductoMaestro"],
            row["CodProductoCpe"],
            row["UnidadMedidaCpe"],
            row["UnidadMedidaCompra"],
        )
        global_product_key = (row["CodProductoMaestro"], row["CodProductoCpe"])

        indexes["exact"][exact_key] = _prefer_conversion_row(indexes["exact"].get(exact_key), row)
        indexes["ruc_product"][ruc_product_key] = _prefer_conversion_row(
            indexes["ruc_product"].get(ruc_product_key), row
        )
        indexes["global_exact"][global_exact_key] = _prefer_conversion_row(
            indexes["global_exact"].get(global_exact_key), row
        )
        indexes["global_product"][global_product_key] = _prefer_conversion_row(
            indexes["global_product"].get(global_product_key), row
        )

    return indexes


def _build_learned_alias_index(
    maestro_p: pd.DataFrame,
    aliases: tuple[tuple[str, str, str], ...],
) -> dict[tuple[str, str], int]:
    if not aliases:
        return {}

    maestro = maestro_p.copy()
    maestro["_ruc_alias"] = maestro["RucProveedor"].map(_norm_ruc_alias)
    maestro["_cod_alias"] = maestro["CodProducto"].map(_norm_cod_alias)

    by_ruc_cod = {
        (str(row["_ruc_alias"]), str(row["_cod_alias"])): idx
        for idx, row in maestro.iterrows()
        if str(row["_cod_alias"])
    }

    code_counts = maestro.groupby("_cod_alias").size()
    unique_codes = set(code_counts[code_counts.eq(1)].index.astype(str))
    by_unique_cod = {
        str(row["_cod_alias"]): idx
        for idx, row in maestro.iterrows()
        if str(row["_cod_alias"]) in unique_codes
    }

    out: dict[tuple[str, str], int] = {}
    for ruc, fact_cod, master_cod in aliases:
        idx = by_ruc_cod.get((ruc, master_cod))
        if idx is None:
            idx = by_unique_cod.get(master_cod)
        if idx is not None:
            out[(ruc, fact_cod)] = idx

    return out


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

    alias_columnas = {
        "cantidad": "Cantidad",
        "cantidadcpe": "Cantidad",
        "cantidad_cpe": "Cantidad",
        "cantidad_factura": "Cantidad",
        "valortotal": "ValorTotal",
        "valor_total": "ValorTotal",
    }
    renombres = {}
    existentes_lower = {str(c).lower().strip(): c for c in df.columns}
    for alias, canonica in alias_columnas.items():
        if canonica not in df.columns and alias in existentes_lower:
            renombres[existentes_lower[alias]] = canonica
    if renombres:
        df = df.rename(columns=renombres)

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
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0.0)

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
        learned_alias_idx = _build_learned_alias_index(
            maestro_p,
            _load_positive_aliases(tenant_norm),
        )
        quantity_conversion_lookup = _load_quantity_conversion_lookup(tenant_norm)

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
                learned_alias_idx=learned_alias_idx,
                quantity_conversion_lookup=quantity_conversion_lookup,
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
