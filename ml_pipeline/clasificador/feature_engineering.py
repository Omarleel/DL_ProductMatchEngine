from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.brands import extract_primary_brand
from ml_pipeline.utils.limpieza import MEASURE_TOKENS, PACK_TOKENS, normalizar_texto

from .categories import CategoryLexicon
from .schema import AttributeSchemaV2


def _safe_str_series(s) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.fillna("").astype(str)
    return pd.Series(s).fillna("").astype(str)


def add_item_aux_features(df: pd.DataFrame, category_lexicon: CategoryLexicon | None) -> pd.DataFrame:
    work = df.copy()
    raw_text = _safe_str_series(work.get("Producto", work.get("text", "")))
    base_text = _safe_str_series(work.get("Producto_base_norm", work.get("base_text", raw_text)))
    norm_text = raw_text.map(normalizar_texto)
    tokens = norm_text.str.split()

    work["brand_hint"] = _safe_str_series(work.get("brand_hint", raw_text.map(extract_primary_brand)))
    if category_lexicon is None:
        work["category_hint"] = _safe_str_series(work.get("category_hint", ""))
    else:
        work["category_hint"] = _safe_str_series(work.get("category_hint", base_text.map(category_lexicon.primary)))

    work["provider"] = _safe_str_series(work.get("RucProveedor", work.get("provider", "")))
    work["brand_hint_present"] = (work["brand_hint"].str.strip() != "").astype(np.float32)
    work["category_hint_present"] = (work["category_hint"].str.strip() != "").astype(np.float32)
    work["n_tokens"] = tokens.map(len).astype(np.float32)
    work["n_chars"] = norm_text.str.len().astype(np.float32)
    work["has_digits"] = norm_text.str.contains(r"\d", regex=True).astype(np.float32)
    work["has_pack_hint"] = tokens.map(lambda xs: float(any(t in PACK_TOKENS for t in xs))).astype(np.float32)
    work["has_measure_hint"] = tokens.map(lambda xs: float(any(t in MEASURE_TOKENS for t in xs))).astype(np.float32)
    work["pack_count_hint"] = tokens.map(lambda xs: float(sum(1 for t in xs if t in PACK_TOKENS))).astype(np.float32)

    def _measure_value(xs: list[str]) -> float:
        vals = []
        for i in range(len(xs) - 1):
            if xs[i].replace('.', '', 1).isdigit() and xs[i + 1] in MEASURE_TOKENS:
                try:
                    vals.append(float(xs[i]))
                except Exception:
                    pass
        return float(max(vals) if vals else 0.0)

    work["measure_value_hint"] = tokens.map(_measure_value).astype(np.float32)

    for c in AttributeSchemaV2.NUMERIC_COLUMNS:
        work[c] = pd.to_numeric(work.get(c, 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    for c in AttributeSchemaV2.AUX_COLUMNS:
        work[c] = pd.to_numeric(work.get(c, 0.0), errors="coerce").fillna(0.0).astype(np.float32)

    return work
