from __future__ import annotations

import pandas as pd

from ml_pipeline.utils.matching import construir_indice_codigos
from ml_pipeline.utils.preparacion import preparar_facturas, preparar_maestro

from .categories import CategoryLexicon
from .feature_engineering import add_item_aux_features
from .labels import preparar_targets_desde_maestro


def _historial_row(f: pd.Series, target_row: pd.Series) -> dict:
    return {
        "source_type": "historial",
        "sample_weight_base": 1.0,
        "RucProveedor": f.get("RucProveedor", ""),
        "provider": f.get("RucProveedor", ""),
        "fact_cod": f.get("CodProducto", ""),
        "text": f.get("Producto_norm", f.get("Producto", "")),
        "base_text": f.get("Producto_base_norm", f.get("Producto", "")),
        "Producto": f.get("Producto", ""),
        "Producto_base_norm": f.get("Producto_base_norm", f.get("Producto", "")),
        "unit": f.get("Unidad_norm", ""),
        "type": f.get("TipoContenido", "NONE"),
        "cost": float(f.get("Costo_log", 0.0)),
        "factor": float(f.get("Factor_log", 0.0)),
        "content": float(f.get("ContenidoUnidad_log", 0.0)),
        "total": float(f.get("ContenidoTotal_log", 0.0)),
        "peso": float(f.get("PesoUnitario", 0.0)),
        "target_factor_venta": target_row.get("target_factor_venta"),
        "target_factor_conversion": target_row.get("target_factor_conversion"),
        "target_peso_unitario_kg": target_row.get("target_peso_unitario_kg"),
        "target_peso_caja_kg": target_row.get("target_peso_caja_kg"),
        "target_factor_venta_log": target_row.get("target_factor_venta_log"),
        "target_factor_conversion_log": target_row.get("target_factor_conversion_log"),
        "target_peso_unitario_kg_log": target_row.get("target_peso_unitario_kg_log"),
        "target_peso_caja_kg_log": target_row.get("target_peso_caja_kg_log"),
        "target_marca": str(target_row.get("target_marca", "SIN_MARCA")),
        "target_categoria": str(target_row.get("target_categoria", "SIN_CATEGORIA")),
        "mask_factor_venta": float(target_row.get("mask_factor_venta", 0.0)),
        "mask_factor_conversion": float(target_row.get("mask_factor_conversion", 0.0)),
        "mask_peso_unitario": float(target_row.get("mask_peso_unitario", 0.0)),
        "mask_peso_caja": float(target_row.get("mask_peso_caja", 0.0)),
        "mask_marca": float(target_row.get("mask_marca", 0.0)),
        "mask_categoria": float(target_row.get("mask_categoria", 0.0)),
    }


def _maestro_row(m: pd.Series) -> dict:
    return {
        "source_type": "maestro",
        "sample_weight_base": 0.25,
        "RucProveedor": m.get("RucProveedor", ""),
        "provider": m.get("RucProveedor", ""),
        "fact_cod": m.get("CodProducto", ""),
        "text": m.get("Producto_norm", m.get("Producto", "")),
        "base_text": m.get("Producto_base_norm", m.get("Producto", "")),
        "Producto": m.get("Producto", ""),
        "Producto_base_norm": m.get("Producto_base_norm", m.get("Producto", "")),
        "unit": m.get("Unidad_norm", ""),
        "type": m.get("TipoContenido", "NONE"),
        "cost": float(m.get("Costo_log", 0.0)),
        "factor": float(m.get("Factor_log", 0.0)),
        "content": float(m.get("ContenidoUnidad_log", 0.0)),
        "total": float(m.get("ContenidoTotal_log", 0.0)),
        "peso": float(m.get("PesoUnitario", 0.0)),
        "target_factor_venta": m.get("target_factor_venta"),
        "target_factor_conversion": m.get("target_factor_conversion"),
        "target_peso_unitario_kg": m.get("target_peso_unitario_kg"),
        "target_peso_caja_kg": m.get("target_peso_caja_kg"),
        "target_factor_venta_log": m.get("target_factor_venta_log"),
        "target_factor_conversion_log": m.get("target_factor_conversion_log"),
        "target_peso_unitario_kg_log": m.get("target_peso_unitario_kg_log"),
        "target_peso_caja_kg_log": m.get("target_peso_caja_kg_log"),
        "target_marca": str(m.get("target_marca", "SIN_MARCA")),
        "target_categoria": str(m.get("target_categoria", "SIN_CATEGORIA")),
        "mask_factor_venta": float(m.get("mask_factor_venta", 0.0)),
        "mask_factor_conversion": float(m.get("mask_factor_conversion", 0.0)),
        "mask_peso_unitario": float(m.get("mask_peso_unitario", 0.0)),
        "mask_peso_caja": float(m.get("mask_peso_caja", 0.0)),
        "mask_marca": float(m.get("mask_marca", 0.0)),
        "mask_categoria": float(m.get("mask_categoria", 0.0)),
    }


def construir_dataset_clasificador(
    maestro: pd.DataFrame,
    productos_facturas: pd.DataFrame,
    usar_maestro_como_ejemplos: bool = True,
) -> tuple[pd.DataFrame, CategoryLexicon]:
    maestro_p = preparar_maestro(maestro)
    historial_p = preparar_facturas(productos_facturas)
    maestro_t = preparar_targets_desde_maestro(maestro_p)

    idx = construir_indice_codigos(maestro_t)
    rows: list[dict] = []

    for f in historial_p.itertuples(index=False):
        key = (str(getattr(f, "RucProveedor", "")).strip(), getattr(f, "CodProducto", ""))
        m_idx = idx.get(key)
        if m_idx is None:
            continue
        target_row = maestro_t.loc[m_idx]
        rows.append(_historial_row(pd.Series(f._asdict()), target_row))

    if usar_maestro_como_ejemplos:
        for _, m in maestro_t.iterrows():
            rows.append(_maestro_row(m))

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError(
            "No se pudieron construir ejemplos supervisados. Verifica que productos_facturas comparta códigos con maestro."
        )

    dataset = dataset.drop_duplicates(
        subset=["RucProveedor", "fact_cod", "text", "target_marca", "target_categoria", "source_type"],
        keep="first",
    ).reset_index(drop=True)

    lexicon_source = dataset[dataset["mask_categoria"] > 0].copy()
    category_lexicon = CategoryLexicon.build(
        lexicon_source,
        text_col="base_text",
        label_col="target_categoria",
        min_support=4,
        top_k_per_category=10,
    )
    dataset = add_item_aux_features(dataset, category_lexicon)
    return dataset, category_lexicon
