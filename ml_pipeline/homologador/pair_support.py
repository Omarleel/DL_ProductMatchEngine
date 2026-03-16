from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from ml_pipeline.utils.limpieza import normalizar_texto


GENERIC_TOKENS = {
    "X", "UND", "UNIDAD", "UNIDADES", "CAJA", "CJA", "CJ",
    "PAQUETE", "PQT", "PACK", "PCK", "PAQ", "BOL", "BOLSA",
    "BOT", "BOTELLA", "LT", "L", "ML", "CC", "KG", "GR", "G",
    "TIPO", "CONT", "FC", "PE", "DISPLAY", "DP", "DSP",
}


def _safe_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return 0.0
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return 0.0


def _core_tokens(texto: str) -> list[str]:
    toks = normalizar_texto(_safe_text(texto)).split()
    out = []
    for t in toks:
        if not t:
            continue
        if t in GENERIC_TOKENS:
            continue
        if t.isdigit():
            continue
        if len(t) <= 1:
            continue
        out.append(t)
    return out


def _jaccard_tokens(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _overlap_ratio(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(min(len(sa), len(sb)), 1)


def _char_ngrams(texto: str, n: int = 3) -> set[str]:
    s = normalizar_texto(_safe_text(texto)).replace(" ", "")
    if not s:
        return set()
    if len(s) <= n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def _char_jaccard(a: str, b: str, n: int = 3) -> float:
    return _jaccard_tokens(_char_ngrams(a, n=n), _char_ngrams(b, n=n))


def _rel_diff(a: float, b: float) -> float:
    a = _safe_float(a)
    b = _safe_float(b)
    den = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / den


def _sim_rel(a: float, b: float, alpha: float) -> float:
    if _safe_float(a) <= 0.0 or _safe_float(b) <= 0.0:
        return 0.5
    return float(math.exp(-alpha * _rel_diff(a, b)))


def _tipo_known(x: str) -> bool:
    s = _safe_text(x).strip().upper()
    return bool(s) and s != "NONE"


def compute_support_for_candidate(f: pd.Series, c: pd.Series) -> dict:
    fact_text = _safe_text(f.get("Producto_norm", ""))
    fact_base = _safe_text(f.get("Producto_base_norm", ""))
    cand_text = _safe_text(c.get("Producto_norm", ""))
    cand_base = _safe_text(c.get("Producto_base_norm", ""))

    tf = _core_tokens(fact_text)
    tm = _core_tokens(cand_text)
    bf = _core_tokens(fact_base)
    bm = _core_tokens(cand_base)

    lex_j_text = _jaccard_tokens(tf, tm)
    lex_j_base = _jaccard_tokens(bf, bm)
    lex_o_text = _overlap_ratio(tf, tm)
    lex_o_base = _overlap_ratio(bf, bm)
    char_j_text = _char_jaccard(fact_text, cand_text, n=3)
    char_j_base = _char_jaccard(fact_base, cand_base, n=3)

    lexical_support = (
        0.18 * lex_j_text
        + 0.28 * lex_j_base
        + 0.18 * lex_o_text
        + 0.22 * lex_o_base
        + 0.05 * char_j_text
        + 0.09 * char_j_base
    )

    unit_equal = float(_safe_text(f.get("Unidad_norm", "")) == _safe_text(c.get("Unidad_norm", "")))
    type_equal = float(_safe_text(f.get("TipoContenido", "")) == _safe_text(c.get("TipoContenido", "")))
    type_known = float(_tipo_known(f.get("TipoContenido", "")) and _tipo_known(c.get("TipoContenido", "")))

    factor_sim = _sim_rel(f.get("FactorConversion", 0), c.get("FactorConversion", 0), 6.0)
    content_sim = _sim_rel(f.get("ContenidoUnidad", 0), c.get("ContenidoUnidad", 0), 7.0)
    total_sim = _sim_rel(f.get("ContenidoTotal", 0), c.get("ContenidoTotal", 0), 7.0)
    peso_sim = _sim_rel(f.get("PesoUnitario", 0), c.get("PesoUnitario", 0), 7.0)
    cost_sim = _sim_rel(f.get("CostoCaja", 0), c.get("CostoCaja", 0), 2.5)

    structure_support = (
        0.10 * unit_equal
        + 0.08 * (type_equal if type_known else 0.5)
        + 0.24 * factor_sim
        + 0.22 * content_sim
        + 0.22 * total_sim
        + 0.08 * peso_sim
        + 0.06 * cost_sim
    )

    support = 0.58 * lexical_support + 0.42 * structure_support

    return {
        "LexicalSupport": float(lexical_support),
        "StructureSupport": float(structure_support),
        "Support": float(support),
        "UnitEqual": float(unit_equal),
        "TypeEqualKnown": float(type_equal if type_known else 0.0),
        "FactorSim": float(factor_sim),
        "ContentSim": float(content_sim),
        "TotalSim": float(total_sim),
        "PesoSim": float(peso_sim),
        "CostSim": float(cost_sim),
    }


def attach_support_features(f: pd.Series, cand: pd.DataFrame) -> pd.DataFrame:
    rows = [compute_support_for_candidate(f, c) for _, c in cand.iterrows()]
    feats = pd.DataFrame(rows, index=cand.index)
    return pd.concat([cand.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)