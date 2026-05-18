from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.utils.preparacion import preparar_facturas, preparar_maestro

from .categories import CategoryLexicon
from .feature_engineering import add_item_aux_features
from .labels import preparar_targets_desde_maestro


TARGET_COLUMNS = [
    "target_factor_venta",
    "target_factor_conversion",
    "target_peso_unitario_kg",
    "target_peso_caja_kg",
    "target_factor_venta_log",
    "target_factor_conversion_log",
    "target_peso_unitario_kg_log",
    "target_peso_caja_kg_log",
    "target_marca",
    "target_categoria",
    "mask_factor_venta",
    "mask_factor_conversion",
    "mask_peso_unitario",
    "mask_peso_caja",
    "mask_marca",
    "mask_categoria",
]

NUMERIC_SOURCE_COLUMNS = {
    "cost": "Costo_log",
    "factor": "Factor_log",
    "content": "ContenidoUnidad_log",
    "total": "ContenidoTotal_log",
    "peso": "PesoUnitario",
}

NUMERIC_TARGET_COLUMNS = [
    "target_factor_venta",
    "target_factor_conversion",
    "target_peso_unitario_kg",
    "target_peso_caja_kg",
    "target_factor_venta_log",
    "target_factor_conversion_log",
    "target_peso_unitario_kg_log",
    "target_peso_caja_kg_log",
    "mask_factor_venta",
    "mask_factor_conversion",
    "mask_peso_unitario",
    "mask_peso_caja",
    "mask_marca",
    "mask_categoria",
]

LABEL_TARGET_DEFAULTS = {
    "target_marca": "SIN_MARCA",
    "target_categoria": "SIN_CATEGORIA",
}


def _timed(logger: Any | None, message: str):
    if logger is not None and hasattr(logger, "timed"):
        return logger.timed(message)
    return nullcontext()


def _normalizar_ruc_series(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _normalizar_codigo_series(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .replace("nan", "")
        .str.strip()
    )


def _series(df: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _text_series(df: pd.DataFrame, column: str, fallback: str | None = None) -> pd.Series:
    if column in df.columns:
        return df[column].fillna("").astype(str)
    if fallback and fallback in df.columns:
        return df[fallback].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype="object")


def _num_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _build_codigo_index(maestro_t: pd.DataFrame) -> pd.DataFrame:
    cod_cols = [c for c in ("CodProducto", "CodProducto2", "CodProducto3") if c in maestro_t.columns]
    if not cod_cols:
        return pd.DataFrame(columns=["_ruc_key", "_codigo_key", "_mrow"])

    key_frames: list[pd.DataFrame] = []
    base = maestro_t[["RucProveedor"]].copy()
    base["_mrow"] = np.arange(len(maestro_t), dtype=np.int64)
    base["_ruc_key"] = _normalizar_ruc_series(base["RucProveedor"])

    for cod_col in cod_cols:
        frame = base[["_ruc_key", "_mrow"]].copy()
        frame["_codigo_key"] = _normalizar_codigo_series(maestro_t[cod_col])
        frame = frame[frame["_codigo_key"] != ""]
        if not frame.empty:
            key_frames.append(frame)

    if not key_frames:
        return pd.DataFrame(columns=["_ruc_key", "_codigo_key", "_mrow"])

    keys = pd.concat(key_frames, ignore_index=True)

    # Equivale al dict anterior: ante claves repetidas, gana la última ocurrencia.
    keys = keys.drop_duplicates(subset=["_ruc_key", "_codigo_key"], keep="last")
    return keys


def _project_rows(source: pd.DataFrame, *, source_type: str, sample_weight_base: float) -> pd.DataFrame:
    out = pd.DataFrame(index=source.index)
    out["source_type"] = source_type
    out["sample_weight_base"] = sample_weight_base
    out["RucProveedor"] = _text_series(source, "RucProveedor")
    out["provider"] = out["RucProveedor"]
    out["fact_cod"] = _text_series(source, "CodProducto")
    out["text"] = _text_series(source, "Producto_norm", fallback="Producto")
    out["base_text"] = _text_series(source, "Producto_base_norm", fallback="Producto")
    out["Producto"] = _text_series(source, "Producto")
    out["Producto_base_norm"] = _text_series(source, "Producto_base_norm", fallback="Producto")
    out["unit"] = _text_series(source, "Unidad_norm")
    out["type"] = _text_series(source, "TipoContenido").replace("", "NONE")

    for dst, src in NUMERIC_SOURCE_COLUMNS.items():
        out[dst] = _num_series(source, src, 0.0).astype(float)

    for col in TARGET_COLUMNS:
        if col in LABEL_TARGET_DEFAULTS:
            out[col] = _text_series(source, col).replace("", LABEL_TARGET_DEFAULTS[col])
        elif col in NUMERIC_TARGET_COLUMNS:
            out[col] = _num_series(source, col, 0.0).astype(float)
        else:
            out[col] = _series(source, col)

    return out


def _build_historial_rows(historial_p: pd.DataFrame, maestro_t: pd.DataFrame, logger: Any | None) -> pd.DataFrame:
    with _timed(logger, "clasificador dataset | deduplicar historial"):
        before = len(historial_p)
        dedup_cols = [c for c in ["RucProveedor", "CodProducto", "Producto_norm"] if c in historial_p.columns]
        if dedup_cols:
            historial_p = historial_p.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        if logger is not None:
            logger.info("Historial deduplicado: rows=%s -> rows_unicas=%s", before, len(historial_p))

    with _timed(logger, "clasificador dataset | construir índice de códigos maestro"):
        keys = _build_codigo_index(maestro_t)
        if logger is not None:
            logger.info("Índice códigos maestro: keys=%s", len(keys))

    with _timed(logger, "clasificador dataset | cruzar historial vs maestro"):
        hist = historial_p.copy()
        hist["_ruc_key"] = _normalizar_ruc_series(hist["RucProveedor"])
        hist["_codigo_key"] = _normalizar_codigo_series(hist["CodProducto"])

        matched = hist.merge(keys, on=["_ruc_key", "_codigo_key"], how="inner", sort=False)
        if matched.empty:
            return pd.DataFrame()

        targets = maestro_t.reset_index(drop=True)[TARGET_COLUMNS].copy()
        targets["_mrow"] = np.arange(len(targets), dtype=np.int64)
        matched = matched.merge(targets, on="_mrow", how="left", sort=False)

        if logger is not None:
            coverage = len(matched) / max(len(hist), 1)
            logger.info("Cruce historial/maestro: matches=%s coverage=%.2f%%", len(matched), coverage * 100)

    with _timed(logger, "clasificador dataset | proyectar filas historial"):
        return _project_rows(matched, source_type="historial", sample_weight_base=1.0)


def _build_maestro_rows(maestro_t: pd.DataFrame, logger: Any | None) -> pd.DataFrame:
    with _timed(logger, "clasificador dataset | proyectar filas maestro"):
        return _project_rows(maestro_t, source_type="maestro", sample_weight_base=0.25)


def construir_dataset_clasificador(
    maestro: pd.DataFrame,
    productos_facturas: pd.DataFrame,
    usar_maestro_como_ejemplos: bool = True,
    logger: Any | None = None,
) -> tuple[pd.DataFrame, CategoryLexicon]:
    with _timed(logger, "clasificador dataset | preparar maestro"):
        maestro_p = preparar_maestro(maestro)

    with _timed(logger, "clasificador dataset | preparar historial facturas"):
        historial_p = preparar_facturas(productos_facturas)

    with _timed(logger, "clasificador dataset | preparar targets maestro"):
        maestro_t = preparar_targets_desde_maestro(maestro_p)

    parts: list[pd.DataFrame] = []
    historial_rows = _build_historial_rows(historial_p, maestro_t, logger)
    if not historial_rows.empty:
        parts.append(historial_rows)

    if usar_maestro_como_ejemplos:
        maestro_rows = _build_maestro_rows(maestro_t, logger)
        if not maestro_rows.empty:
            parts.append(maestro_rows)

    with _timed(logger, "clasificador dataset | consolidar ejemplos"):
        dataset = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if dataset.empty:
            raise ValueError(
                "No se pudieron construir ejemplos supervisados. Verifica que productos_facturas comparta códigos con maestro."
            )

        before = len(dataset)
        dataset = dataset.drop_duplicates(
            subset=["RucProveedor", "fact_cod", "text", "target_marca", "target_categoria", "source_type"],
            keep="first",
        ).reset_index(drop=True)
        if logger is not None:
            logger.info("Dataset consolidado: rows=%s -> rows_unicas=%s", before, len(dataset))

    with _timed(logger, "clasificador dataset | construir lexicon categorías"):
        lexicon_source = dataset[dataset["mask_categoria"] > 0].copy()
        category_lexicon = CategoryLexicon.build(
            lexicon_source,
            text_col="base_text",
            label_col="target_categoria",
            min_support=4,
            top_k_per_category=10,
        )
        if logger is not None:
            logger.info("Lexicon categorías: categorias=%s", len(category_lexicon.category_to_terms))

    with _timed(logger, "clasificador dataset | agregar features auxiliares"):
        dataset = add_item_aux_features(dataset, category_lexicon)

    return dataset, category_lexicon
