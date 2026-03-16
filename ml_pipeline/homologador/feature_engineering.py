from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.brands import extract_primary_brand, brand_set


AUX_FEATURE_COLUMNS = [
    "same_brand",
    "brand_conflict",
    "brand_overlap",
    "unit_equal",
    "type_equal_known",
    "factor_sim",
    "content_sim",
    "total_sim",
    "peso_sim",
    "cost_sim",
]


def _safe_str(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("")


def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32)


def _exp_sim(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _safe_float(a)
    b = _safe_float(b)
    return np.exp(-np.abs(a - b)).astype(np.float32)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def add_aux_pair_features(pares: pd.DataFrame) -> pd.DataFrame:
    df = pares.copy()

    fact_text = _safe_str(df["fact_text"])
    master_text = _safe_str(df["master_text"])

    fact_brand_primary = fact_text.map(extract_primary_brand)
    master_brand_primary = master_text.map(extract_primary_brand)

    fact_brand_set = fact_text.map(brand_set)
    master_brand_set = master_text.map(brand_set)

    df["same_brand"] = (
        (fact_brand_primary != "")
        & (master_brand_primary != "")
        & (fact_brand_primary == master_brand_primary)
    ).astype(np.float32)

    df["brand_conflict"] = (
        (fact_brand_primary != "")
        & (master_brand_primary != "")
        & (fact_brand_primary != master_brand_primary)
    ).astype(np.float32)

    df["brand_overlap"] = [
        _jaccard(a, b) for a, b in zip(fact_brand_set, master_brand_set)
    ]
    df["brand_overlap"] = df["brand_overlap"].astype(np.float32)

    df["unit_equal"] = (
        _safe_str(df["fact_unit"]).str.strip()
        == _safe_str(df["master_unit"]).str.strip()
    ).astype(np.float32)

    fact_type = _safe_str(df["fact_type"]).str.strip()
    master_type = _safe_str(df["master_type"]).str.strip()

    df["type_equal_known"] = (
        (fact_type != "NONE")
        & (master_type != "NONE")
        & (fact_type == master_type)
    ).astype(np.float32)

    df["factor_sim"] = _exp_sim(df["fact_factor"], df["master_factor"])
    df["content_sim"] = _exp_sim(df["fact_content"], df["master_content"])
    df["total_sim"] = _exp_sim(df["fact_total"], df["master_total"])
    df["peso_sim"] = _exp_sim(df["fact_peso"], df["master_peso"])
    df["cost_sim"] = _exp_sim(df["fact_cost"], df["master_cost"])

    for col in AUX_FEATURE_COLUMNS:
        df[col] = _safe_float(df[col])

    return df