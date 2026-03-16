from __future__ import annotations

import math
from dataclasses import dataclass

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

from .factor_resolver import (
    _anchor_tokens,
    _jaccard,
    _norm_code_robust,
    _norm_ruc,
    _quantize_factor,
    _rel_diff,
    _safe_float,
    _seq_ratio,
    _sim_rel,
)
from .labels import preparar_targets_desde_maestro


@dataclass
class WeightResolution:
    peso_unitario_kg: float | None
    peso_caja_kg: float | None
    source: str
    score: float
    matched_cod: str = ""
    matched_producto: str = ""


_NON_PHYSICAL_TERMS = {
    "ANTICIPO",
    "SERVICIO",
    "BONIFICACION",
    "BONIFICACIONES",
    "MATERIAL POP",
    "PROMOTIONAL MATERIAL",
    "PPG",
    "MUESTRA",
    "OBSEQUIO",
}

_FAMILY_STOP = {
    "CAJA",
    "CJA",
    "CJ",
    "UNIDAD",
    "UND",
    "UN",
    "PQT",
    "PCK",
    "PACK",
    "BOL",
    "BOT",
    "BOTELLA",
    "BOTELLA",
    "DP",
    "DISPLAY",
    "DISPLEY",
    "PE",
    "EX",
    "X",
    "CON",
    "SIN",
    "PARA",
    "DE",
    "DEL",
    "LA",
    "EL",
    "LOS",
    "LAS",
    "Y",
    "EN",
    "POR",
    "TM",
    "TM.",
    "TLLA",
    "H",
}


def _round_kg(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v) or v < 0:
        return None
    return round(v, 6)


def _normalize_factor_pair(factor_venta, factor_conversion) -> tuple[int, int]:
    fv = int(_quantize_factor(factor_venta))
    fc = int(_quantize_factor(factor_conversion))

    if fc > 0 and fv <= 0:
        fv = fc
    if fv > 0 and fc <= 0:
        fc = fv
    if fv > 0 and fc > 0:
        fc = max(fc, fv)

    return fv, fc


def _units_per_sale(factor_venta: int, factor_conversion: int) -> int:
    fv, fc = _normalize_factor_pair(factor_venta, factor_conversion)
    if fv <= 0 or fc <= 0 or fc < fv:
        return 1
    if fc % fv != 0:
        return 1
    ups = fc // fv
    if ups < 1:
        return 1
    return int(min(ups, 500))


def _is_non_physical(text: str, unit: str = "") -> bool:
    text_u = normalizar_texto(text)
    unit_u = normalizar_texto(unit)
    if unit_u == "SERVICIO":
        return True
    return any(term in text_u for term in _NON_PHYSICAL_TERMS)


def _density_from_text(text: str) -> float | None:
    # Regla pedida: todo volumen se trata como equivalente en kg
    # 1L = 1kg, 1ML = 0.001kg, 1CC = 0.001kg
    return 1.0

def _family_tokens(text: str, limit: int = 8) -> tuple[str, ...]:
    toks = []
    extra_stop = set(MEASURE_TOKENS) | set(PACK_TOKENS) | set(DIMENSION_TOKENS) | _FAMILY_STOP
    for tok in normalizar_texto(text).split():
        if not tok:
            continue
        if tok.isdigit():
            continue
        if tok in extra_stop:
            continue
        if len(tok) <= 1:
            continue
        toks.append(tok)
    return tuple(toks[:limit])


def _family_overlap(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _family_prefix_match(a: tuple[str, ...], b: tuple[str, ...], n: int = 2) -> float:
    if len(a) < n or len(b) < n:
        return 0.0
    return 1.0 if tuple(a[:n]) == tuple(b[:n]) else 0.0


def _description_signature(text: str, factor_venta: int, factor_conversion: int) -> dict:
    attrs = extraer_atributos_producto(text)
    tipo = str(attrs.get("TipoContenido", "NONE")).upper()
    contenido_unidad = float(attrs.get("ContenidoUnidad", 0.0) or 0.0)
    contenido_total = float(attrs.get("ContenidoTotal", 0.0) or 0.0)
    ups = _units_per_sale(factor_venta, factor_conversion)
    family = _family_tokens(text)

    explicit = tipo in {"MASS", "VOLUME"} and contenido_unidad > 0
    density = 1.0 if tipo == "VOLUME" else None

    peso_unit = None
    if explicit:
        if tipo == "MASS":
            peso_unit = _round_kg((contenido_unidad / 1000.0) * ups)
        elif tipo == "VOLUME":
            peso_unit = _round_kg((contenido_unidad / 1000.0) * ups)

    return {
        "tipo": tipo,
        "contenido_unidad": contenido_unidad,
        "contenido_total": contenido_total,
        "ups": ups,
        "explicit": explicit,
        "density": density,
        "peso_unitario_kg": peso_unit,
        "family": family,
        "count_only": (not explicit),
    }

def _master_signature(row: pd.Series) -> dict:
    tipo = str(row.get("TipoContenido", "NONE")).upper()
    contenido_unidad = _safe_float(row.get("ContenidoUnidad", 0.0))
    contenido_total = _safe_float(row.get("ContenidoTotal", 0.0))
    fv = _quantize_factor(row.get("target_factor_venta", np.nan))
    fc = _quantize_factor(row.get("target_factor_conversion", np.nan))
    ups = _units_per_sale(fv, fc)
    peso_unit = _safe_float(row.get("target_peso_unitario_kg", np.nan))
    peso_caja = _safe_float(row.get("target_peso_caja_kg", np.nan))
    family = _family_tokens(str(row.get("Producto", "")))

    return {
        "tipo": tipo,
        "contenido_unidad": contenido_unidad,
        "contenido_total": contenido_total,
        "factor_venta": fv,
        "factor_conversion": fc,
        "ups": ups,
        "peso_unitario_kg": peso_unit if peso_unit > 0 else None,
        "peso_caja_kg": peso_caja if peso_caja > 0 else None,
        "family": family,
    }


def _should_prefer_description(
    desc_sig: dict,
    master_sig: dict,
    resolved_master_unit: float | None,
    master_score: float,
    is_exact: bool,
) -> bool:
    desc_weight = desc_sig.get("peso_unitario_kg")
    if not desc_sig.get("explicit") or desc_weight is None or desc_weight <= 0:
        return False

    if resolved_master_unit is None or resolved_master_unit <= 0:
        return True

    desc_tipo = str(desc_sig.get("tipo", "NONE"))
    master_tipo = str(master_sig.get("tipo", "NONE"))

    if desc_tipo in {"MASS", "VOLUME"} and master_tipo not in {"NONE", desc_tipo}:
        return True

    desc_content = float(desc_sig.get("contenido_unidad", 0.0) or 0.0)
    master_content = float(master_sig.get("contenido_unidad", 0.0) or 0.0)

    if desc_content > 0 and master_content > 0 and _rel_diff(desc_content, master_content) > 0.10:
        return True

    diff_weight = _rel_diff(desc_weight, resolved_master_unit)

    if is_exact:
        return diff_weight > 0.18

    if master_score < 0.90 and diff_weight > 0.12:
        return True

    return diff_weight > 0.20


class MaestroWeightResolver:
    def __init__(self, maestro: pd.DataFrame):
        maestro_p = preparar_maestro(maestro)
        maestro_t = preparar_targets_desde_maestro(maestro_p)

        work = maestro_t.copy().reset_index(drop=True)
        work["_ruc_key"] = work["RucProveedor"].map(_norm_ruc)
        work["_cod_key"] = work["CodProducto"].map(_norm_code_robust)
        work["_producto_norm"] = work["Producto_norm"].fillna("").astype(str)
        work["_base_norm"] = work["Producto_base_norm"].fillna("").astype(str)
        work["_base_anchor"] = work["Producto_base_norm"].map(_anchor_tokens)
        work["_family_tokens"] = work["Producto_norm"].fillna("").astype(str).map(_family_tokens)

        keep = (pd.to_numeric(work["target_peso_unitario_kg"], errors="coerce") > 0) | (
            pd.to_numeric(work["target_peso_caja_kg"], errors="coerce") > 0
        )
        self.master = work.loc[keep].reset_index(drop=True)

        self.exact_index: dict[tuple[str, str], int] = {}
        for i, row in self.master.iterrows():
            ruc = str(row["_ruc_key"])
            for col in ["CodProducto", "CodProducto2", "CodProducto3"]:
                code = _norm_code_robust(row.get(col, ""))
                if code:
                    self.exact_index[(ruc, code)] = int(i)

    def _exact_match(self, row: pd.Series) -> pd.Series | None:
        key = (_norm_ruc(row.get("RucProveedor", "")), _norm_code_robust(row.get("CodProducto", "")))
        idx = self.exact_index.get(key)
        if idx is None:
            return None
        return self.master.iloc[int(idx)]

    def _candidate_pool(self, row: pd.Series) -> tuple[pd.DataFrame, str]:
        ruc = _norm_ruc(row.get("RucProveedor", ""))
        local = self.master[self.master["_ruc_key"] == ruc]
        if not local.empty:
            return local, "same_ruc"
        return self.master, "global"

    def _score_candidates(
        self,
        row: pd.Series,
        pool: pd.DataFrame,
        factor_venta: int,
        factor_conversion: int,
        desc_sig: dict,
    ) -> pd.DataFrame:
        if pool.empty:
            return pool

        fact_base = str(row.get("Producto_base_norm", row.get("Producto", "")))
        fact_norm = str(row.get("Producto_norm", row.get("Producto", "")))
        fact_anchor = _anchor_tokens(fact_base)
        fact_family = desc_sig.get("family", _family_tokens(fact_norm))
        fact_unit = str(row.get("Unidad_norm", ""))
        fact_type = str(row.get("TipoContenido", ""))
        fact_content = _safe_float(row.get("ContenidoUnidad", 0.0))
        fact_total = _safe_float(row.get("ContenidoTotal", 0.0))
        ups_cur = _units_per_sale(factor_venta, factor_conversion)

        cand = pool.copy()
        cand["sim_base_jaccard"] = cand["_base_norm"].map(lambda x: _jaccard(fact_base, x))
        cand["sim_text_ratio"] = cand["_producto_norm"].map(lambda x: _seq_ratio(fact_norm, x))
        cand["sim_family"] = cand["_family_tokens"].map(lambda x: _family_overlap(fact_family, tuple(x)))
        cand["family_prefix"] = cand["_family_tokens"].map(lambda x: _family_prefix_match(fact_family, tuple(x), 2))
        cand["same_unit"] = (cand["Unidad_norm"].fillna("").astype(str) == fact_unit).astype(np.float32)
        cand["same_type"] = (cand["TipoContenido"].fillna("").astype(str) == fact_type).astype(np.float32)
        cand["sim_content"] = cand["ContenidoUnidad"].map(lambda x: _sim_rel(fact_content, _safe_float(x), scale=7.0))
        cand["sim_total"] = cand["ContenidoTotal"].map(lambda x: _sim_rel(fact_total, _safe_float(x), scale=6.0))
        cand["sim_factor_venta"] = cand["target_factor_venta"].map(lambda x: _sim_rel(factor_venta, _safe_float(x), scale=4.0))
        cand["sim_factor_conversion"] = cand["target_factor_conversion"].map(
            lambda x: _sim_rel(factor_conversion, _safe_float(x), scale=4.0)
        )
        cand["sim_units_per_sale"] = cand.apply(
            lambda r: _sim_rel(
                ups_cur,
                _units_per_sale(r.get("target_factor_venta", 1), r.get("target_factor_conversion", 1)),
                scale=5.0,
            ),
            axis=1,
        )
        cand["same_factor_pair"] = (
            (pd.to_numeric(cand["target_factor_venta"], errors="coerce").fillna(0).round().astype(int) == int(factor_venta))
            & (pd.to_numeric(cand["target_factor_conversion"], errors="coerce").fillna(0).round().astype(int) == int(factor_conversion))
        ).astype(np.float32)
        cand["anchor_bonus"] = cand["_base_anchor"].map(
            lambda x: 1.0 if fact_anchor and tuple(x)[:2] == fact_anchor[:2] else 0.0
        )

        explicit_desc = bool(desc_sig.get("explicit", False))

        if explicit_desc:
            cand["score_weight_match"] = (
                0.18 * cand["sim_base_jaccard"]
                + 0.16 * cand["sim_text_ratio"]
                + 0.12 * cand["sim_family"]
                + 0.08 * cand["same_unit"]
                + 0.06 * cand["same_type"]
                + 0.14 * cand["sim_content"]
                + 0.10 * cand["sim_total"]
                + 0.06 * cand["sim_factor_venta"]
                + 0.04 * cand["sim_factor_conversion"]
                + 0.03 * cand["sim_units_per_sale"]
                + 0.02 * cand["family_prefix"]
                + 0.01 * cand["anchor_bonus"]
            )
        else:
            cand["score_weight_match"] = (
                0.16 * cand["sim_base_jaccard"]
                + 0.16 * cand["sim_text_ratio"]
                + 0.18 * cand["sim_family"]
                + 0.10 * cand["same_unit"]
                + 0.06 * cand["same_type"]
                + 0.05 * cand["sim_content"]
                + 0.04 * cand["sim_total"]
                + 0.09 * cand["sim_factor_venta"]
                + 0.07 * cand["sim_factor_conversion"]
                + 0.05 * cand["sim_units_per_sale"]
                + 0.03 * cand["same_factor_pair"]
                + 0.01 * cand["family_prefix"]
            )

        return cand.sort_values(
            ["score_weight_match", "sim_family", "sim_text_ratio", "sim_base_jaccard"],
            ascending=False,
        )

    def _accept_similar_candidate(self, best: pd.Series, scope: str, desc_sig: dict) -> bool:
        best_score = float(best.get("score_weight_match", 0.0))
        sim_text = float(best.get("sim_text_ratio", 0.0))
        sim_family = float(best.get("sim_family", 0.0))
        same_pair = float(best.get("same_factor_pair", 0.0))
        sim_ups = float(best.get("sim_units_per_sale", 0.0))
        explicit_desc = bool(desc_sig.get("explicit", False))

        if explicit_desc:
            min_score = 0.80 if scope == "same_ruc" else 0.88
            min_text = 0.66 if scope == "same_ruc" else 0.76
            return best_score >= min_score and sim_text >= min_text

        if scope == "same_ruc":
            if same_pair >= 1.0 and sim_family >= 0.45 and sim_text >= 0.55 and best_score >= 0.70:
                return True
            if sim_family >= 0.60 and sim_text >= 0.58 and sim_ups >= 0.90 and best_score >= 0.72:
                return True
            return best_score >= 0.77 and sim_text >= 0.62 and sim_family >= 0.38

        if same_pair >= 1.0 and sim_family >= 0.60 and sim_text >= 0.68 and best_score >= 0.82:
            return True
        return best_score >= 0.86 and sim_text >= 0.74 and sim_family >= 0.45

    def _resolve_from_master_row(
        self,
        row: pd.Series,
        factor_venta: int,
        factor_conversion: int,
        source: str,
        score: float,
    ) -> WeightResolution:
        peso_unit = _safe_float(row.get("target_peso_unitario_kg", np.nan))
        peso_caja = _safe_float(row.get("target_peso_caja_kg", np.nan))

        current_ups = _units_per_sale(factor_venta, factor_conversion)
        row_fv = _quantize_factor(row.get("target_factor_venta", np.nan))
        row_fc = _quantize_factor(row.get("target_factor_conversion", np.nan))
        row_ups = _units_per_sale(row_fv, row_fc)

        if peso_unit > 0 and row_ups > 0 and current_ups > 0 and row_ups != current_ups:
            piece = peso_unit / float(row_ups)
            if piece > 0:
                peso_unit = piece * current_ups
                source = source + "+scaled_piece"

        if peso_unit <= 0 and peso_caja > 0 and factor_venta > 0:
            peso_unit = peso_caja / float(max(int(factor_venta), 1))

        if peso_unit <= 0:
            return WeightResolution(
                None,
                None,
                source,
                score,
                str(row.get("CodProducto", "")),
                str(row.get("Producto", "")),
            )

        if peso_caja <= 0:
            peso_caja = peso_unit * float(max(int(factor_venta), 1))

        return WeightResolution(
            peso_unitario_kg=_round_kg(peso_unit),
            peso_caja_kg=_round_kg(peso_caja),
            source=source,
            score=float(score),
            matched_cod=str(row.get("CodProducto", "")),
            matched_producto=str(row.get("Producto", "")),
        )

    def resolve_one(self, factura_preparada: pd.Series, factor_venta: int, factor_conversion: int) -> WeightResolution:
        text = str(factura_preparada.get("Producto", ""))
        unit = str(factura_preparada.get("UnidaMedidaCompra", ""))

        if _is_non_physical(text, unit):
            return WeightResolution(0.0, 0.0, "regla_no_fisico", 1.0)

        desc_sig = _description_signature(text, factor_venta, factor_conversion)
        parsed = desc_sig.get("peso_unitario_kg")

        exact = self._exact_match(factura_preparada)
        if exact is not None:
            exact_res = self._resolve_from_master_row(exact, factor_venta, factor_conversion, "exacto_maestro_peso", 1.0)
            if exact_res.peso_unitario_kg is not None:
                if _should_prefer_description(
                    desc_sig=desc_sig,
                    master_sig=_master_signature(exact),
                    resolved_master_unit=exact_res.peso_unitario_kg,
                    master_score=1.0,
                    is_exact=True,
                ):
                    return WeightResolution(
                        parsed,
                        _round_kg(parsed * float(max(int(factor_venta), 1))) if parsed is not None else None,
                        "parser_descripcion_peso>exacto_maestro",
                        0.97,
                    )
                return exact_res

        pool, scope = self._candidate_pool(factura_preparada)
        ranked = self._score_candidates(factura_preparada, pool, factor_venta, factor_conversion, desc_sig)

        if not ranked.empty:
            best = ranked.iloc[0]
            best_score = float(best.get("score_weight_match", 0.0))

            if self._accept_similar_candidate(best, scope, desc_sig):
                best_res = self._resolve_from_master_row(best, factor_venta, factor_conversion, f"similar_maestro_peso_{scope}", best_score)
                if best_res.peso_unitario_kg is not None:
                    if _should_prefer_description(
                        desc_sig=desc_sig,
                        master_sig=_master_signature(best),
                        resolved_master_unit=best_res.peso_unitario_kg,
                        master_score=best_score,
                        is_exact=False,
                    ):
                        return WeightResolution(
                            parsed,
                            _round_kg(parsed * float(max(int(factor_venta), 1))) if parsed is not None else None,
                            f"parser_descripcion_peso>similar_maestro_{scope}",
                            max(0.94, min(0.98, best_score)),
                            matched_cod=str(best.get("CodProducto", "")),
                            matched_producto=str(best.get("Producto", "")),
                        )
                    return best_res

        if parsed is not None:
            return WeightResolution(
                parsed,
                _round_kg(parsed * float(max(int(factor_venta), 1))),
                "parser_descripcion_peso",
                0.93,
            )

        return WeightResolution(None, None, "modelo_nn_peso", 0.0)

    def resolve_many(self, productos_facturas: pd.DataFrame, factor_venta: pd.Series, factor_conversion: pd.Series) -> pd.DataFrame:
        fact_p = preparar_facturas(productos_facturas)
        fv = pd.to_numeric(pd.Series(factor_venta).reset_index(drop=True), errors="coerce").fillna(1).astype(int)
        fc = pd.to_numeric(pd.Series(factor_conversion).reset_index(drop=True), errors="coerce").fillna(fv).astype(int)

        rows = []
        for i, row in enumerate(fact_p.itertuples(index=False)):
            r = self.resolve_one(pd.Series(row._asdict()), int(fv.iloc[i]), int(fc.iloc[i]))
            rows.append(
                {
                    "resolved_pesoUnitarioKg": r.peso_unitario_kg,
                    "resolved_pesoCajaKg": r.peso_caja_kg,
                    "weight_source": r.source,
                    "weight_match_score": r.score,
                    "weight_match_cod": r.matched_cod,
                    "weight_match_producto": r.matched_producto,
                }
            )
        return pd.DataFrame(rows)