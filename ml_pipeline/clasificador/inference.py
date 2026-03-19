from __future__ import annotations

import numpy as np
import pandas as pd

from ml_pipeline.utils.preparacion import preparar_facturas
from .model import ModeloClasificadorProductos


def _quantize_factor_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)
    s = s.round()
    s = s.where(s.isna(), s.clip(lower=1))
    return s


def _enforce_integer_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_factorVenta"] = _quantize_factor_series(df["pred_factorVenta"])
    df["pred_factorConversion"] = _quantize_factor_series(df["pred_factorConversion"])
    df["pred_factorVenta"] = df["pred_factorVenta"].fillna(1).astype(int)
    df["pred_factorConversion"] = df[["pred_factorConversion", "pred_factorVenta"]].max(axis=1).fillna(df["pred_factorVenta"]).astype(int)
    return df


def _build_inference_frame(productos_facturas: pd.DataFrame) -> pd.DataFrame:
    fact_p = preparar_facturas(productos_facturas)
    base = pd.DataFrame({
        "RucProveedor": fact_p.get("RucProveedor", ""),
        "provider": fact_p.get("RucProveedor", ""),
        "fact_cod": fact_p.get("CodProducto", ""),
        "Producto": fact_p.get("Producto", ""),
        "Producto_base_norm": fact_p.get("Producto_base_norm", fact_p.get("Producto", "")),
        "text": fact_p.get("Producto_norm", fact_p.get("Producto", "")),
        "base_text": fact_p.get("Producto_base_norm", fact_p.get("Producto", "")),
        "unit": fact_p.get("Unidad_norm", ""),
        "type": fact_p.get("TipoContenido", "NONE"),
        "cost": fact_p.get("Costo_log", 0.0),
        "factor": fact_p.get("Factor_log", 0.0),
        "content": fact_p.get("ContenidoUnidad_log", 0.0),
        "total": fact_p.get("ContenidoTotal_log", 0.0),
        "peso": fact_p.get("PesoUnitario", 0.0),
        "target_factor_venta": 0.0,
        "target_factor_conversion": 0.0,
        "target_peso_unitario_kg": 0.0,
        "target_peso_caja_kg": 0.0,
        "target_factor_venta_log": 0.0,
        "target_factor_conversion_log": 0.0,
        "target_peso_unitario_kg_log": 0.0,
        "target_peso_caja_kg_log": 0.0,
        "target_marca": "SIN_MARCA",
        "target_categoria": "SIN_CATEGORIA",
        "mask_factor_venta": 0.0,
        "mask_factor_conversion": 0.0,
        "mask_peso_unitario": 0.0,
        "mask_peso_caja": 0.0,
        "mask_marca": 0.0,
        "mask_categoria": 0.0,
        "sample_weight_base": 1.0,
        "source_type": "inferencia",
    })
    return base


def inferir_atributos_producto(
    productos_facturas: pd.DataFrame,
    modelo: ModeloClasificadorProductos,
    resolvers: dict | None = None,
    batch_size: int = 32,
    include_factor_debug: bool = False,
    resolver_marca_categoria_desde_maestro: bool = False,
    brand_category_min_score: float = 0.55,
) -> pd.DataFrame:
    
    # 1. Preparación de datos e Inferencia de Red Neuronal
    # _build_inference_frame debe ser eficiente (operaciones vectorizadas)
    inf_df = _build_inference_frame(productos_facturas)
    pred_df = modelo.predict(inf_df, batch_size=batch_size)

    salida = pred_df.copy()
    salida["RucProveedor"] = productos_facturas.get("RucProveedor", "")
    salida["CodProductoFactura"] = productos_facturas.get("CodProducto", "")
    salida["ProductoFactura"] = productos_facturas.get("Producto", "")
    salida["UnidadFactura"] = productos_facturas.get("UnidadMedidaCompra", "")

    # 2. Uso de Resolvers pre-indexados (Evitamos el LAG de inicialización)
    if resolvers and len(productos_facturas) > 0:
        # Resolver Factores
        factor_df = resolvers["factor"].resolve_many(productos_facturas)
        salida["pred_factorVenta"] = pd.to_numeric(factor_df["resolved_factorVenta"], errors="coerce")
        salida["pred_factorConversion"] = pd.to_numeric(factor_df["resolved_factorConversion"], errors="coerce")
        salida = _enforce_integer_factors(salida)

        # Resolver Pesos
        weight_df = resolvers["weight"].resolve_many(
            productos_facturas=productos_facturas,
            factor_venta=salida["pred_factorVenta"],
            factor_conversion=salida["pred_factorConversion"],
        )

        peso_res = pd.to_numeric(weight_df["resolved_pesoUnitarioKg"], errors="coerce")
        peso_caja_res = pd.to_numeric(weight_df["resolved_pesoCajaKg"], errors="coerce")
        peso_nn = pd.to_numeric(salida["pred_pesoUnitarioKg"], errors="coerce")

        salida["pred_pesoUnitarioKg"] = peso_res.where(peso_res.notna() & (peso_res >= 0), peso_nn)
        salida["pred_pesoCajaKg"] = peso_caja_res.where(
            peso_caja_res.notna() & (peso_caja_res >= 0),
            pd.to_numeric(salida["pred_pesoUnitarioKg"], errors="coerce").fillna(0.0)
            * pd.to_numeric(salida["pred_factorVenta"], errors="coerce").fillna(0.0),
        )

        # NUEVO: resolver marca/categoría desde maestro
        if resolver_marca_categoria_desde_maestro:
            bc_df = resolvers["brand_cat"].resolve_many(productos_facturas)

            marca_res = bc_df["resolved_marca"].astype(object)
            categoria_res = bc_df["resolved_categoria"].astype(object)

            salida["pred_marca"] = marca_res.where(
                marca_res.notna() & (marca_res.astype(str).str.strip() != "") & (marca_res.astype(str) != "SIN_MARCA"),
                salida["pred_marca"],
            )
            salida["pred_categoria"] = categoria_res.where(
                categoria_res.notna() & (categoria_res.astype(str).str.strip() != "") & (categoria_res.astype(str) != "SIN_CATEGORIA"),
                salida["pred_categoria"],
            )

            # Opcional: subir confianza cuando vino de maestro
            marca_from_maestro = bc_df["brand_category_source"].isin(["maestro_exact", "maestro_fuzzy"])
            salida["conf_marca"] = np.where(
                marca_from_maestro,
                np.maximum(pd.to_numeric(salida["conf_marca"], errors="coerce").fillna(0.0), bc_df["brand_category_match_score"].fillna(0.0)),
                pd.to_numeric(salida["conf_marca"], errors="coerce").fillna(0.0),
            )
            salida["conf_categoria"] = np.where(
                marca_from_maestro,
                np.maximum(pd.to_numeric(salida["conf_categoria"], errors="coerce").fillna(0.0), bc_df["brand_category_match_score"].fillna(0.0)),
                pd.to_numeric(salida["conf_categoria"], errors="coerce").fillna(0.0),
            )

        if include_factor_debug and factor_df is not None:
            for c in [
                "factor_source",
                "factor_match_score",
                "factor_match_cod",
                "factor_match_producto",
                "resolved_factorVenta",
                "resolved_factorConversion",
            ]:
                salida[c] = factor_df[c].values
            for c in [
                "weight_source",
                "weight_match_score",
                "weight_match_cod",
                "weight_match_producto",
                "resolved_pesoUnitarioKg",
                "resolved_pesoCajaKg",
            ]:
                salida[c] = weight_df[c].values

            if bc_df is not None:
                for c in [
                    "brand_category_source",
                    "brand_category_match_score",
                    "brand_category_match_cod",
                    "brand_category_match_producto",
                    "resolved_marca",
                    "resolved_categoria",
                ]:
                    salida[c] = bc_df[c].values
    else:
        salida = _enforce_integer_factors(salida)
        salida["pred_pesoCajaKg"] = (
            pd.to_numeric(salida["pred_pesoUnitarioKg"], errors="coerce").fillna(0.0)
            * pd.to_numeric(salida["pred_factorVenta"], errors="coerce").fillna(0.0)
        )

    salida["pred_pesoUnitarioKg"] = pd.to_numeric(salida["pred_pesoUnitarioKg"], errors="coerce").fillna(0.0).round(6)
    salida["pred_pesoCajaKg"] = pd.to_numeric(salida["pred_pesoCajaKg"], errors="coerce").fillna(0.0).round(6)

    if include_factor_debug:
        # Usamos una lista para recolectar DataFrames y concatenar una sola vez (más rápido)
        to_concat = []
        for df_source in [factor_df, weight_df, bc_df]:
            if df_source is not None:
                # Solo traemos columnas que no existan ya en 'salida' para evitar duplicados (.1, .2)
                cols_to_use = df_source.columns.difference(salida.columns)
                to_concat.append(df_source[cols_to_use])
        
        if to_concat:
            salida = pd.concat([salida, *to_concat], axis=1)

    # 4. Manejo de casos donde no hay Maestro (Valores por defecto para debug)
    if include_factor_debug and not resolvers:
        # Aseguramos que las columnas existan aunque no haya maestro para evitar KeyErrors
        debug_cols = ["factor_source", "weight_source", "brand_category_source"]
        for col in debug_cols:
            if col not in salida.columns:
                salida[col] = "modelo_nn_default"

    columnas = [
        "RucProveedor",
        "CodProductoFactura",
        "ProductoFactura",
        "UnidadFactura",
        "pred_factorVenta",
        "pred_factorConversion",
        "pred_pesoUnitarioKg",
        "pred_pesoCajaKg",
        "pred_marca",
        "pred_categoria",
        "conf_marca",
        "conf_categoria",
    ]
    if include_factor_debug:
        columnas.extend([
            "factor_source",
            "factor_match_score",
            "factor_match_cod",
            "factor_match_producto",
            "resolved_factorVenta",
            "resolved_factorConversion",
            "weight_source",
            "weight_match_score",
            "weight_match_cod",
            "weight_match_producto",
            "resolved_pesoUnitarioKg",
            "resolved_pesoCajaKg",
            "brand_category_source",
            "brand_category_match_score",
            "brand_category_match_cod",
            "brand_category_match_producto",
            "resolved_marca",
            "resolved_categoria",
        ])
    return salida[columnas].copy()