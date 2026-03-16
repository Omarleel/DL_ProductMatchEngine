from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import pandas as pd

from ml_pipeline.utils.limpieza import (
    DIMENSION_TOKENS,
    MEASURE_TOKENS,
    PACK_TOKENS,
    extraer_atributos_producto,
    normalizar_texto,
)
from ml_pipeline.utils.preparacion import preparar_facturas, preparar_maestro

from .labels import preparar_targets_desde_maestro


@dataclass
class FactorResolution:
    factor_venta: Optional[float]
    factor_conversion: Optional[float]
    source: str
    score: float
    matched_cod: str = ""
    matched_producto: str = ""


_TOKEN_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")
_HYPHEN_PAIR_RE = re.compile(r"^(\d+)-(\d+)$")


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


def _norm_ruc(x) -> str:
    s = "" if x is None else str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def _norm_code_robust(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if not s:
        return ""
    s = s.replace(" ", "")
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".", 1)[0]
    if re.search(r"E[+-]?\d+", s):
        try:
            f = float(s)
            if math.isfinite(f) and float(f).is_integer():
                return str(int(round(f)))
        except Exception:
            pass
    return s


def _split_tokens(text: str) -> list[str]:
    return [t for t in normalizar_texto(text).split() if t]


def _is_number(tok: str) -> bool:
    return bool(_TOKEN_NUM_RE.fullmatch(tok))


def _num(tok: str) -> Optional[int]:
    if not _is_number(tok):
        return None
    try:
        v = float(tok)
        return int(v) if float(v).is_integer() else None
    except Exception:
        return None


def _rel_diff(a: float, b: float) -> float:
    a = float(a or 0.0)
    b = float(b or 0.0)
    if a <= 0.0 or b <= 0.0:
        return 1.0
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def _sim_rel(a: float, b: float, scale: float = 5.0) -> float:
    if float(a or 0.0) <= 0.0 or float(b or 0.0) <= 0.0:
        return 0.5
    return float(math.exp(-scale * _rel_diff(a, b)))


def _jaccard(a: str, b: str) -> float:
    sa = set(_split_tokens(a))
    sb = set(_split_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalizar_texto(a), normalizar_texto(b)).ratio()


_STOP_TOKENS = {
    "CAJA",
    "CJA",
    "CJ",
    "UNIDAD",
    "UND",
    "PQT",
    "PCK",
    "PACK",
    "BOL",
    "BOT",
    "DP",
    "DISPLAY",
    "DISPLEY",
    "PE",
    "EX",
    "X",
}


def _anchor_tokens(text: str, n: int = 3) -> tuple[str, ...]:
    toks = [t for t in _split_tokens(text) if not _is_number(t) and t not in _STOP_TOKENS and len(t) > 1]
    return tuple(toks[:n])


def _parse_hyphen_pair(tok: str) -> tuple[int, int] | None:
    m = _HYPHEN_PAIR_RE.fullmatch(str(tok).strip())
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2))
    if a > 1 and b > 1:
        return a, b
    return None


def _quantize_factor(x) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    if not math.isfinite(v) or v <= 0:
        return 0
    return int(max(1, round(v)))


def _normalize_factor_pair(factor_venta, factor_conversion) -> tuple[int, int]:
    fv = _quantize_factor(factor_venta)
    fc = _quantize_factor(factor_conversion)

    if fc > 0 and fv <= 0:
        fv = fc
    if fv > 0 and fc <= 0:
        fc = fv
    if fv > 0 and fc > 0:
        fc = max(fc, fv)

    return int(fv), int(fc)


def _units_per_sale(factor_venta: int, factor_conversion: int) -> int:
    fv, fc = _normalize_factor_pair(factor_venta, factor_conversion)
    if fv <= 0 or fc <= 0 or fc < fv:
        return 1
    if fc % fv != 0:
        return 1
    ups = fc // fv
    return int(max(1, min(ups, 1000)))


@dataclass
class ParsedPackSignals:
    factor_venta: int
    factor_conversion: int
    strength: float
    explicit: bool = False
    kind: str = ""


class PackSignalExtractor:
    @staticmethod
    def extract(text: str) -> ParsedPackSignals:
        toks = _split_tokens(text)
        if not toks:
            return ParsedPackSignals(0, 0, 0.0, False, "")

        attrs = extraer_atributos_producto(text)
        factor_conversion = _quantize_factor(attrs.get("FactorConversion", 0))

        measure_idx = -1
        for i in range(len(toks) - 1):
            if _is_number(toks[i]) and toks[i + 1] in MEASURE_TOKENS:
                measure_idx = i
                break

        explicit_counts: list[int] = []
        after_measure_counts: list[int] = []
        after_measure_hyphen: tuple[int, int] | None = None
        pre_measure_count = 0
        generic_chain: list[int] = []
        has_dimension = any(t in DIMENSION_TOKENS for t in toks)

        i = 0
        while i < len(toks) - 1:
            n = _num(toks[i])
            nxt = toks[i + 1]

            if n is not None and n > 1 and nxt in PACK_TOKENS:
                explicit_counts.append(int(n))
                i += 2
                continue

            pair = _parse_hyphen_pair(toks[i])
            if pair and nxt in PACK_TOKENS:
                explicit_counts.extend([pair[0], pair[1]])
                i += 2
                continue

            if toks[i] in PACK_TOKENS and toks[i + 1] == "X" and i + 2 < len(toks):
                n2 = _num(toks[i + 2])
                if n2 is not None and n2 > 1:
                    explicit_counts.append(int(n2))
                    i += 3
                    continue
            i += 1

        for i in range(len(toks) - 3):
            a = _num(toks[i])
            b = _num(toks[i + 2]) if toks[i + 1] == "X" else None
            if a is None or b is None or a <= 1:
                continue
            if toks[i + 3] in MEASURE_TOKENS:
                pre_measure_count = int(a)
                break

        if measure_idx >= 0:
            j = measure_idx + 2
            if j < len(toks) and toks[j] == "X" and j + 1 < len(toks):
                pair = _parse_hyphen_pair(toks[j + 1])
                if pair is not None:
                    after_measure_hyphen = pair
                else:
                    while j < len(toks) - 1:
                        if toks[j] != "X":
                            j += 1
                            continue
                        n = _num(toks[j + 1])
                        if n is None or n <= 1:
                            j += 1
                            continue
                        nxt = toks[j + 2] if j + 2 < len(toks) else ""
                        if nxt in MEASURE_TOKENS or nxt in DIMENSION_TOKENS:
                            break
                        after_measure_counts.append(int(n))
                        j += 2
            elif j < len(toks):
                n0 = _num(toks[j])
                if n0 is not None and n0 > 1:
                    after_measure_counts.append(int(n0))

        if not has_dimension and measure_idx < 0:
            for i in range(len(toks) - 2):
                a = _num(toks[i])
                b = _num(toks[i + 2]) if toks[i + 1] == "X" else None
                if a is None or b is None or a <= 1:
                    continue
                if float(b) <= 1:
                    generic_chain = [int(a)]
                    break

                generic_chain = [int(a), int(b)]
                k = i + 3
                while k + 1 < len(toks) and toks[k] == "X":
                    nk = _num(toks[k + 1])
                    if nk is None or nk <= 1:
                        break
                    generic_chain.append(int(nk))
                    k += 2
                break

        factor_venta = 0
        strength = 0.0
        explicit = False
        kind = ""

        if after_measure_hyphen is not None:
            factor_venta = int(after_measure_hyphen[0])
            strength = 0.97
            explicit = True
            kind = "after_measure_hyphen"
            if factor_conversion <= 1:
                factor_conversion = int(after_measure_hyphen[0] * after_measure_hyphen[1])

        elif len(after_measure_counts) >= 2:
            factor_venta = int(after_measure_counts[-1])
            strength = 0.96
            explicit = True
            kind = "after_measure_chain"
            if factor_conversion <= 1:
                fc = 1
                for n in after_measure_counts:
                    if n > 1:
                        fc *= int(n)
                factor_conversion = int(max(fc, factor_venta))

        elif explicit_counts:
            factor_venta = int(explicit_counts[-1])
            explicit = True
            kind = "explicit_pack_tokens"
            strength = 0.93 if len(explicit_counts) >= 2 else 0.88
            if factor_conversion <= 1:
                fc = 1
                for n in explicit_counts:
                    if n > 1:
                        fc *= int(n)
                factor_conversion = int(max(fc, factor_venta))

        elif len(after_measure_counts) == 1:
            factor_venta = int(after_measure_counts[0])
            strength = 0.93
            explicit = True
            kind = "after_measure_single"
            if factor_conversion <= 1:
                factor_conversion = int(factor_venta)

        elif pre_measure_count > 1:
            factor_venta = int(pre_measure_count)
            strength = 0.90
            explicit = True
            kind = "pre_measure_count"
            if factor_conversion <= 1:
                factor_conversion = int(factor_venta)

        elif generic_chain:
            factor_venta = int(generic_chain[0])
            explicit = True
            kind = "generic_x_chain"

            if len(generic_chain) == 1:
                strength = 0.84
                if factor_conversion <= 1:
                    factor_conversion = int(factor_venta)
            elif len(generic_chain) == 2:
                strength = 0.91
                if factor_conversion <= 1:
                    factor_conversion = int(generic_chain[0] * generic_chain[1])
            else:
                strength = 0.89
                if factor_conversion <= 1:
                    fc = 1
                    for n in generic_chain:
                        if n > 1:
                            fc *= int(n)
                    factor_conversion = int(max(fc, factor_venta))

        factor_venta, factor_conversion = _normalize_factor_pair(factor_venta, factor_conversion)

        return ParsedPackSignals(
            int(factor_venta),
            int(factor_conversion),
            float(strength),
            bool(explicit),
            str(kind),
        )


def _parser_signature(text: str, parsed: ParsedPackSignals) -> dict:
    attrs = extraer_atributos_producto(text)
    return {
        "factor_venta": int(parsed.factor_venta),
        "factor_conversion": int(parsed.factor_conversion),
        "strength": float(parsed.strength),
        "explicit": bool(parsed.explicit),
        "kind": str(parsed.kind),
        "tipo": str(attrs.get("TipoContenido", "NONE")).upper(),
        "contenido_unidad": _safe_float(attrs.get("ContenidoUnidad", 0.0)),
        "contenido_total": _safe_float(attrs.get("ContenidoTotal", 0.0)),
        "ups": _units_per_sale(parsed.factor_venta, parsed.factor_conversion),
    }


def _master_signature(row: pd.Series) -> dict:
    fv = _quantize_factor(row.get("target_factor_venta", np.nan))
    fc = _quantize_factor(row.get("target_factor_conversion", np.nan))
    return {
        "factor_venta": int(fv),
        "factor_conversion": int(fc),
        "tipo": str(row.get("TipoContenido", "NONE")).upper(),
        "contenido_unidad": _safe_float(row.get("ContenidoUnidad", 0.0)),
        "contenido_total": _safe_float(row.get("ContenidoTotal", 0.0)),
        "ups": _units_per_sale(fv, fc),
    }


def _parser_pair(parsed: ParsedPackSignals) -> tuple[int, int]:
    fv = parsed.factor_venta if parsed.factor_venta > 1 else parsed.factor_conversion
    fc = parsed.factor_conversion if parsed.factor_conversion > 1 else fv
    return _normalize_factor_pair(fv, fc)


def _master_pair(row: pd.Series) -> tuple[int, int]:
    return _normalize_factor_pair(
        row.get("target_factor_venta", np.nan),
        row.get("target_factor_conversion", np.nan),
    )


def _content_conflict(parser_sig: dict, master_sig: dict) -> bool:
    p_tipo = str(parser_sig.get("tipo", "NONE"))
    m_tipo = str(master_sig.get("tipo", "NONE"))
    p_cu = float(parser_sig.get("contenido_unidad", 0.0) or 0.0)
    m_cu = float(master_sig.get("contenido_unidad", 0.0) or 0.0)

    if p_tipo in {"MASS", "VOLUME"} and m_tipo in {"MASS", "VOLUME"} and p_tipo == m_tipo and p_cu > 0 and m_cu > 0:
        return _rel_diff(p_cu, m_cu) > 0.10
    return False


def _factor_conflict(parser_sig: dict, master_sig: dict) -> bool:
    pfv = int(parser_sig.get("factor_venta", 0) or 0)
    pfc = int(parser_sig.get("factor_conversion", 0) or 0)
    mfv = int(master_sig.get("factor_venta", 0) or 0)
    mfc = int(master_sig.get("factor_conversion", 0) or 0)

    if pfv > 1 and mfv > 1 and _rel_diff(pfv, mfv) > 0.08:
        return True
    if pfc > 1 and mfc > 1 and _rel_diff(pfc, mfc) > 0.08:
        return True
    if parser_sig.get("ups", 1) > 1 and master_sig.get("ups", 1) > 1:
        if _rel_diff(parser_sig["ups"], master_sig["ups"]) > 0.12:
            return True
    return False


def _should_prefer_parser_over_master(
    parser_sig: dict,
    master_sig: dict,
    match_score: float,
    is_exact: bool,
) -> bool:
    if not parser_sig.get("explicit", False):
        return False

    pfv = int(parser_sig.get("factor_venta", 0) or 0)
    pfc = int(parser_sig.get("factor_conversion", 0) or 0)

    if pfv <= 1 and pfc <= 1:
        return False

    mfv = int(master_sig.get("factor_venta", 0) or 0)
    mfc = int(master_sig.get("factor_conversion", 0) or 0)

    if mfv <= 0 and pfv > 1:
        return True
    if mfc <= 0 and pfc > 1:
        return True

    conflict_factor = _factor_conflict(parser_sig, master_sig)
    conflict_content = _content_conflict(parser_sig, master_sig)
    parser_strength = float(parser_sig.get("strength", 0.0) or 0.0)

    if is_exact:
        if parser_strength >= 0.95 and (conflict_factor or conflict_content):
            return True
        if parser_strength >= 0.90 and conflict_factor and conflict_content:
            return True
        return False

    if parser_strength >= 0.90 and conflict_factor:
        return True
    if parser_strength >= 0.88 and conflict_factor and match_score < 0.92:
        return True
    if parser_strength >= 0.85 and conflict_factor and conflict_content:
        return True
    if conflict_factor and match_score < 0.85:
        return True

    return False


class MaestroFactorResolver:
    def __init__(self, maestro: pd.DataFrame):
        maestro_p = preparar_maestro(maestro)
        maestro_t = preparar_targets_desde_maestro(maestro_p)

        work = maestro_t.copy().reset_index(drop=True)
        work["_ruc_key"] = work["RucProveedor"].map(_norm_ruc)
        work["_cod_key"] = work["CodProducto"].map(_norm_code_robust)
        work["_base_anchor"] = work["Producto_base_norm"].map(_anchor_tokens)
        work["_producto_norm"] = work["Producto_norm"].fillna("").astype(str)
        work["_base_norm"] = work["Producto_base_norm"].fillna("").astype(str)

        keep = (work["target_factor_venta"].notna()) | (work["target_factor_conversion"].notna())
        self.master = work.loc[keep].reset_index(drop=True)

        self.exact_index: dict[tuple[str, str], int] = {}
        for i, row in self.master.iterrows():
            ruc = str(row["_ruc_key"])
            for col in ["CodProducto", "CodProducto2", "CodProducto3"]:
                if col in row.index:
                    code = _norm_code_robust(row.get(col, ""))
                    if code:
                        self.exact_index[(ruc, code)] = int(i)

    def _candidate_pool(self, row: pd.Series) -> tuple[pd.DataFrame, str]:
        ruc = _norm_ruc(row.get("RucProveedor", ""))
        local = self.master[self.master["_ruc_key"] == ruc]
        if not local.empty:
            return local, "same_ruc"
        return self.master, "global"

    def _score_candidates(self, row: pd.Series, pool: pd.DataFrame, parsed: ParsedPackSignals) -> pd.DataFrame:
        if pool.empty:
            return pool

        fact_base = str(row.get("Producto_base_norm", row.get("Producto", "")))
        fact_norm = str(row.get("Producto_norm", row.get("Producto", "")))
        fact_anchor = _anchor_tokens(fact_base)
        fact_unit = str(row.get("Unidad_norm", ""))
        fact_type = str(row.get("TipoContenido", ""))
        fact_content = _safe_float(row.get("ContenidoUnidad", 0.0))
        fact_total = _safe_float(row.get("ContenidoTotal", 0.0))

        parsed_ups = _units_per_sale(parsed.factor_venta, parsed.factor_conversion)

        cand = pool.copy()
        cand["sim_base_jaccard"] = cand["_base_norm"].map(lambda x: _jaccard(fact_base, x))
        cand["sim_text_ratio"] = cand["_producto_norm"].map(lambda x: _seq_ratio(fact_norm, x))
        cand["same_unit"] = (cand["Unidad_norm"].fillna("").astype(str) == fact_unit).astype(np.float32)
        cand["same_type"] = (cand["TipoContenido"].fillna("").astype(str) == fact_type).astype(np.float32)
        cand["sim_content"] = cand["ContenidoUnidad"].map(lambda x: _sim_rel(fact_content, _safe_float(x), scale=7.0))
        cand["sim_total"] = cand["ContenidoTotal"].map(lambda x: _sim_rel(fact_total, _safe_float(x), scale=6.0))

        cand["sim_factor_parse_fv"] = cand["target_factor_venta"].map(
            lambda x: _sim_rel(parsed.factor_venta, _safe_float(x), scale=4.0)
        )
        cand["sim_factor_parse_fc"] = cand["target_factor_conversion"].map(
            lambda x: _sim_rel(parsed.factor_conversion, _safe_float(x), scale=4.0)
        )
        cand["sim_units_per_sale"] = cand.apply(
            lambda r: _sim_rel(
                parsed_ups,
                _units_per_sale(r.get("target_factor_venta", 1), r.get("target_factor_conversion", 1)),
                scale=5.0,
            ),
            axis=1,
        )
        cand["anchor_bonus"] = cand["_base_anchor"].map(
            lambda x: 1.0 if fact_anchor and tuple(x)[:2] == fact_anchor[:2] else 0.0
        )

        cand["score_factor_match"] = (
            0.24 * cand["sim_base_jaccard"]
            + 0.20 * cand["sim_text_ratio"]
            + 0.10 * cand["same_unit"]
            + 0.08 * cand["same_type"]
            + 0.10 * cand["sim_content"]
            + 0.08 * cand["sim_total"]
            + 0.08 * cand["sim_factor_parse_fv"]
            + 0.08 * cand["sim_factor_parse_fc"]
            + 0.02 * cand["sim_units_per_sale"]
            + 0.02 * cand["anchor_bonus"]
        )
        return cand.sort_values(["score_factor_match", "sim_text_ratio", "sim_base_jaccard"], ascending=False)

    def _exact_match(self, row: pd.Series) -> Optional[pd.Series]:
        key = (_norm_ruc(row.get("RucProveedor", "")), _norm_code_robust(row.get("CodProducto", "")))
        idx = self.exact_index.get(key)
        if idx is None:
            return None
        return self.master.iloc[int(idx)]

    def _resolution_from_row(
        self,
        row: pd.Series,
        parsed: ParsedPackSignals,
        source: str,
        score: float,
    ) -> FactorResolution:
        factor_venta, factor_conversion = _master_pair(row)

        if factor_venta <= 0 and parsed.factor_venta > 1:
            factor_venta = parsed.factor_venta
            source += "+parser_fv"
        if factor_conversion <= 0 and parsed.factor_conversion > 1:
            factor_conversion = parsed.factor_conversion
            source += "+parser_fc"

        factor_venta, factor_conversion = _normalize_factor_pair(factor_venta, factor_conversion)

        return FactorResolution(
            factor_venta=float(factor_venta) if factor_venta > 0 else None,
            factor_conversion=float(factor_conversion) if factor_conversion > 0 else None,
            source=source,
            score=float(score),
            matched_cod=str(row.get("CodProducto", "")),
            matched_producto=str(row.get("Producto", "")),
        )

    def _resolution_from_parser(self, parsed: ParsedPackSignals, source: str, score: float) -> FactorResolution:
        factor_venta, factor_conversion = _parser_pair(parsed)
        return FactorResolution(
            factor_venta=float(factor_venta) if factor_venta > 0 else None,
            factor_conversion=float(factor_conversion) if factor_conversion > 0 else None,
            source=source,
            score=float(score),
        )

    def resolve_one(self, factura_preparada: pd.Series) -> FactorResolution:
        text = str(factura_preparada.get("Producto", ""))
        parsed = PackSignalExtractor.extract(text)
        parser_sig = _parser_signature(text, parsed)

        exact = self._exact_match(factura_preparada)
        if exact is not None:
            exact_res = self._resolution_from_row(exact, parsed, "exacto_maestro", 1.0)
            if exact_res.factor_venta is not None or exact_res.factor_conversion is not None:
                if _should_prefer_parser_over_master(
                    parser_sig=parser_sig,
                    master_sig=_master_signature(exact),
                    match_score=1.0,
                    is_exact=True,
                ):
                    return self._resolution_from_parser(parsed, "parser_descripcion>exacto_maestro", 0.98)
                return exact_res

        pool, scope = self._candidate_pool(factura_preparada)
        ranked = self._score_candidates(factura_preparada, pool, parsed)

        if not ranked.empty:
            best = ranked.iloc[0]
            best_score = float(best.get("score_factor_match", 0.0))
            min_score = 0.78 if scope == "same_ruc" else 0.86
            min_text = 0.64 if scope == "same_ruc" else 0.74

            if best_score >= min_score and float(best.get("sim_text_ratio", 0.0)) >= min_text:
                best_res = self._resolution_from_row(best, parsed, f"similar_maestro_{scope}", best_score)
                if best_res.factor_venta is not None or best_res.factor_conversion is not None:
                    if _should_prefer_parser_over_master(
                        parser_sig=parser_sig,
                        master_sig=_master_signature(best),
                        match_score=best_score,
                        is_exact=False,
                    ):
                        return self._resolution_from_parser(
                            parsed,
                            f"parser_descripcion>similar_maestro_{scope}",
                            max(0.90, min(0.97, parsed.strength)),
                        )
                    return best_res

        if parsed.factor_conversion > 1 or parsed.factor_venta > 1:
            return self._resolution_from_parser(parsed, "parser_descripcion", float(parsed.strength))

        return FactorResolution(None, None, "modelo_nn", 0.0)

    def resolve_many(self, productos_facturas: pd.DataFrame) -> pd.DataFrame:
        fact_p = preparar_facturas(productos_facturas)
        rows = []
        for row in fact_p.itertuples(index=False):
            r = self.resolve_one(pd.Series(row._asdict()))
            rows.append(
                {
                    "resolved_factorVenta": r.factor_venta,
                    "resolved_factorConversion": r.factor_conversion,
                    "factor_source": r.source,
                    "factor_match_score": r.score,
                    "factor_match_cod": r.matched_cod,
                    "factor_match_producto": r.matched_producto,
                }
            )
        return pd.DataFrame(rows)