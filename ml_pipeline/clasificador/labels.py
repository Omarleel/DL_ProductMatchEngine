from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from ml_pipeline.utils.limpieza import log_seguro, renombrar_columnas_equivalentes


TARGET_EQUIVALENCIAS = {
    "Marca": ["marca", "brand", "brands", "nombre_marca"],
    "Categoria": [
        "categoria",
        "categoría",
        "category",
        "familia",
        "subcategoria",
        "subcategoría",
        "linea",
        "línea",
        "segmento",
    ],
    "FactorVenta": [
        "factorventa",
        "factor_venta",
        "factor venta",
        "factor de venta",
        "undxcaja",
        "unidadesxcaja",
        "udsxcaja",
        "unid_por_caja",
    ],
    "FactorConversionTarget": [
        "factorconversiontarget",
        "factor_conversion_target",
        "factor conversion target",
        "factorconversion",
        "factor_conversion",
        "factor de conversion",
        "factor de conversión",
    ],
    "PesoUnitarioKgTarget": [
        "pesounitariokgtarget",
        "peso_unitario_kg_target",
        "pesounitariokg",
        "peso_unitario_kg",
        "pesounitario",
        "peso_unitario",
        "peso unitario",
    ],
    "PesoCajaKg": [
        "pesocajakg",
        "peso_caja_kg",
        "peso caja kg",
        "pesocaja",
        "peso_caja",
        "peso caja",
        "pesototalkg",
        "peso_total_kg",
        "peso total kg",
    ],
}

MISSING_BRAND = "SIN_MARCA"
MISSING_CATEGORY = "SIN_CATEGORIA"


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_label(text: str, missing: str) -> str:
    if text is None:
        return missing
    s = _strip_accents(str(text)).upper().strip()
    s = re.sub(r"\s+", " ", s)
    return s if s and s != "NAN" else missing


def preparar_targets_desde_maestro(maestro: pd.DataFrame) -> pd.DataFrame:
    df = renombrar_columnas_equivalentes(maestro.copy(), TARGET_EQUIVALENCIAS)

    required = ["Marca", "Categoria"]
    faltantes = [c for c in required if c not in df.columns]
    if faltantes:
        raise KeyError(
            "Para el modelo v2, maestro.csv debe traer etiquetas curadas de Marca y Categoria. "
            f"Faltan columnas: {faltantes}."
        )

    if "FactorVenta" not in df.columns:
        df["FactorVenta"] = np.nan
    if "FactorConversionTarget" not in df.columns:
        df["FactorConversionTarget"] = df.get("FactorConversion", np.nan)
    if "PesoUnitarioKgTarget" not in df.columns:
        df["PesoUnitarioKgTarget"] = df.get("PesoUnitario", np.nan)
    if "PesoCajaKg" not in df.columns:
        df["PesoCajaKg"] = np.nan

    for c in ["FactorVenta", "FactorConversionTarget", "PesoUnitarioKgTarget", "PesoCajaKg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "FactorConversion" in df.columns:
        df["FactorConversionTarget"] = df["FactorConversionTarget"].fillna(df["FactorConversion"])
    if "PesoUnitario" in df.columns:
        df["PesoUnitarioKgTarget"] = df["PesoUnitarioKgTarget"].fillna(df["PesoUnitario"])

    df["Marca"] = df["Marca"].map(lambda x: normalize_label(x, MISSING_BRAND))
    df["Categoria"] = df["Categoria"].map(lambda x: normalize_label(x, MISSING_CATEGORY))

    # FactorVenta es obligatorio para pesoCaja cuando no venga directo.
    fill_factor = df["FactorConversionTarget"].where(df["FactorConversionTarget"] > 0)
    df["FactorVenta"] = df["FactorVenta"].where(df["FactorVenta"] > 0, fill_factor)

    multiplicador_peso = df["FactorVenta"].where(df["FactorVenta"] > 0, df["FactorConversionTarget"])
    multiplicador_peso = multiplicador_peso.fillna(1.0)
    df["PesoCajaKg"] = df["PesoCajaKg"].where(df["PesoCajaKg"] > 0, df["PesoUnitarioKgTarget"] * multiplicador_peso)

    # Se preservan NaN para poder enmascarar targets no disponibles.
    for c in ["FactorVenta", "FactorConversionTarget", "PesoUnitarioKgTarget", "PesoCajaKg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = np.nan

    df["target_factor_venta"] = df["FactorVenta"]
    df["target_factor_conversion"] = df["FactorConversionTarget"]
    df["target_peso_unitario_kg"] = df["PesoUnitarioKgTarget"]
    df["target_peso_caja_kg"] = df["PesoCajaKg"]
    df["target_marca"] = df["Marca"]
    df["target_categoria"] = df["Categoria"]

    for src, dst in [
        ("target_factor_venta", "target_factor_venta_log"),
        ("target_factor_conversion", "target_factor_conversion_log"),
        ("target_peso_unitario_kg", "target_peso_unitario_kg_log"),
        ("target_peso_caja_kg", "target_peso_caja_kg_log"),
    ]:
        df[dst] = df[src].map(lambda x: np.nan if pd.isna(x) else log_seguro(x))

    df["mask_factor_venta"] = df["target_factor_venta"].notna().astype(np.float32)
    df["mask_factor_conversion"] = df["target_factor_conversion"].notna().astype(np.float32)
    df["mask_peso_unitario"] = df["target_peso_unitario_kg"].notna().astype(np.float32)
    df["mask_peso_caja"] = df["target_peso_caja_kg"].notna().astype(np.float32)
    df["mask_marca"] = (df["target_marca"] != MISSING_BRAND).astype(np.float32)
    df["mask_categoria"] = (df["target_categoria"] != MISSING_CATEGORY).astype(np.float32)

    return df
