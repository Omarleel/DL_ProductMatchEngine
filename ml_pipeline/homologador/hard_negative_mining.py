from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .model import ModeloHomologadorProductos
from ml_pipeline.utils.limpieza import normalizar_codigo, normalizar_texto
from ml_pipeline.utils.preparacion import preparar_maestro


_MAESTRO_TOKENS_CACHE: dict[str, list[str]] = {}


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


def _norm_code(x) -> str:
    s = _safe_text(x).strip()
    return normalizar_codigo(s) if s else ""


def _inv_log1p(x) -> float:
    v = _safe_float(x)
    if v <= 0.0:
        return 0.0
    return max(math.exp(v) - 1.0, 0.0)


def _core_tokens(texto: str) -> list[str]:
    stop = {
        "X", "UND", "UNIDAD", "UNIDADES", "CAJA", "CJA", "CJ",
        "PAQUETE", "PQT", "PACK", "PCK", "PAQ", "BOL", "BOLSA",
        "BOT", "BOTELLA", "LT", "L", "ML", "CC", "KG", "GR", "G",
        "TIPO", "CONT", "FC", "PE", "DISPLAY", "DP", "DSP",
    }
    toks = normalizar_texto(_safe_text(texto)).split()
    out = []
    for t in toks:
        if not t or t in stop or t.isdigit() or len(t) <= 1:
            continue
        out.append(t)
    return out


def _get_tokens_cached(text: str) -> list[str]:
    if text not in _MAESTRO_TOKENS_CACHE:
        _MAESTRO_TOKENS_CACHE[text] = _core_tokens(text)
    return _MAESTRO_TOKENS_CACHE[text]


def _token_set(tokens: list[str]) -> set[str]:
    return set(tokens)


def _rel_diff(a: float, b: float) -> float:
    a = _safe_float(a)
    b = _safe_float(b)
    den = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / den


def _sim_rel(a: float, b: float, alpha: float = 6.0) -> float:
    if _safe_float(a) <= 0.0 or _safe_float(b) <= 0.0:
        return 0.5
    return float(math.exp(-alpha * _rel_diff(a, b)))


def _build_pair_frame(f: pd.Series, cand: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "fact_cod": [f["CodProducto"]] * len(cand),
        "fact_text": [f["Producto_norm"]] * len(cand),
        "fact_base_text": [f["Producto_base_norm"]] * len(cand),
        "fact_unit": [f["Unidad_norm"]] * len(cand),
        "fact_type": [f["TipoContenido"]] * len(cand),
        "fact_cost": [f["Costo_log"]] * len(cand),
        "fact_peso": [f["PesoUnitario"]] * len(cand),
        "fact_factor": [f["Factor_log"]] * len(cand),
        "fact_content": [f["ContenidoUnidad_log"]] * len(cand),
        "fact_total": [f["ContenidoTotal_log"]] * len(cand),
        "master_cod": cand["CodProducto"].values,
        "master_text": cand["Producto_norm"].values,
        "master_base_text": cand["Producto_base_norm"].values,
        "master_unit": cand["Unidad_norm"].values,
        "master_type": cand["TipoContenido"].values,
        "master_cost": cand["Costo_log"].values,
        "master_peso": cand["PesoUnitario"].values,
        "master_factor": cand["Factor_log"].values,
        "master_content": cand["ContenidoUnidad_log"].values,
        "master_total": cand["ContenidoTotal_log"].values,
        "label": [0] * len(cand),
        "RucProveedor": [f["RucProveedor"]] * len(cand),
    })


def _build_negative_row(
    f: pd.Series,
    c: pd.Series,
    label: int = 0,
) -> dict:
    return {
        "fact_cod": f["CodProducto"],
        "fact_text": f["Producto_norm"],
        "fact_base_text": f["Producto_base_norm"],
        "fact_unit": f["Unidad_norm"],
        "fact_type": f["TipoContenido"],
        "fact_cost": f["Costo_log"],
        "fact_peso": f["PesoUnitario"],
        "fact_factor": f["Factor_log"],
        "fact_content": f["ContenidoUnidad_log"],
        "fact_total": f["ContenidoTotal_log"],
        "master_cod": c["CodProducto"],
        "master_text": c["Producto_norm"],
        "master_base_text": c["Producto_base_norm"],
        "master_unit": c["Unidad_norm"],
        "master_type": c["TipoContenido"],
        "master_cost": c["Costo_log"],
        "master_peso": c["PesoUnitario"],
        "master_factor": c["Factor_log"],
        "master_content": c["ContenidoUnidad_log"],
        "master_total": c["ContenidoTotal_log"],
        "label": int(label),
        "RucProveedor": f["RucProveedor"],
        "hard_negative_score_modelo": _safe_float(c.get("ScoreModelo", 0)),
        "hard_negative_support": _safe_float(c.get("Support", 0)),
        "hard_negative_embedding": _safe_float(c.get("EmbeddingScore", 0)),
        "hard_negative_origen": _safe_text(c.get("OrigenCandidato", "")),
    }


def _build_fact_frame_from_pairs(positivos: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "RucProveedor": positivos["RucProveedor"].apply(_safe_text),
        "CodProducto": positivos["fact_cod"].apply(_safe_text),
        "Producto_norm": positivos["fact_text"].apply(_safe_text),
        "Producto_base_norm": positivos["fact_base_text"].apply(_safe_text),
        "Unidad_norm": positivos["fact_unit"].apply(_safe_text),
        "TipoContenido": positivos["fact_type"].apply(_safe_text),
        "Costo_log": positivos["fact_cost"].apply(_safe_float),
        "PesoUnitario": positivos["fact_peso"].apply(_safe_float),
        "Factor_log": positivos["fact_factor"].apply(_safe_float),
        "ContenidoUnidad_log": positivos["fact_content"].apply(_safe_float),
        "ContenidoTotal_log": positivos["fact_total"].apply(_safe_float),
    })
    out["CostoCaja"] = out["Costo_log"].map(_inv_log1p)
    out["FactorConversion"] = out["Factor_log"].map(_inv_log1p)
    out["ContenidoUnidad"] = out["ContenidoUnidad_log"].map(_inv_log1p)
    out["ContenidoTotal"] = out["ContenidoTotal_log"].map(_inv_log1p)
    out["__cod_norm__"] = out["CodProducto"].map(_norm_code)
    out["_ruc_norm"] = out["RucProveedor"].astype(str).fillna("").str.strip()
    return out


def _build_master_arrays(maestro_p: pd.DataFrame) -> dict:
    token_text = [_get_tokens_cached(x) for x in maestro_p["Producto_norm"].astype(str).tolist()]
    token_base = [_get_tokens_cached(x) for x in maestro_p["Producto_base_norm"].astype(str).tolist()]
    token_text_set = [_token_set(x) for x in token_text]
    token_base_set = [_token_set(x) for x in token_base]

    rucs = maestro_p["_ruc_norm"].astype(str).to_numpy()
    by_ruc: dict[str, np.ndarray] = {}
    for idx, ruc in enumerate(rucs):
        by_ruc.setdefault(ruc, []).append(idx)
    by_ruc = {k: np.asarray(v, dtype=np.int32) for k, v in by_ruc.items()}

    return {
        "codes_norm": maestro_p["__cod_norm__"].astype(str).to_numpy(),
        "rucs": rucs,
        "by_ruc": by_ruc,
        "token_text": token_text,
        "token_base": token_base,
        "token_text_set": token_text_set,
        "token_base_set": token_base_set,
        "unidad": maestro_p["Unidad_norm"].astype(str).to_numpy(),
        "tipo": maestro_p["TipoContenido"].astype(str).to_numpy(),
        "factor": maestro_p["FactorConversion"].astype(np.float32).to_numpy(),
        "content": maestro_p["ContenidoUnidad"].astype(np.float32).to_numpy(),
        "total": maestro_p["ContenidoTotal"].astype(np.float32).to_numpy(),
        "peso": maestro_p["PesoUnitario"].astype(np.float32).to_numpy(),
        "costo": maestro_p["CostoCaja"].astype(np.float32).to_numpy(),
    }


def _vectorized_sim_rel_scalar(v: float, arr: np.ndarray, alpha: float) -> np.ndarray:
    if v <= 0.0:
        return np.full(arr.shape, 0.5, dtype=np.float32)
    out = np.full(arr.shape, 0.5, dtype=np.float32)
    valid = arr > 0.0
    if np.any(valid):
        den = np.maximum(np.maximum(np.abs(v), np.abs(arr[valid])), 1e-9)
        rel = np.abs(v - arr[valid]) / den
        out[valid] = np.exp(-alpha * rel).astype(np.float32)
    return out


def _compute_support_batch(f_row: pd.Series, cand_idx: np.ndarray, master_data: dict) -> pd.DataFrame:
    tf = _get_tokens_cached(f_row["Producto_norm"])
    bf = _get_tokens_cached(f_row["Producto_base_norm"])
    tf_set = _token_set(tf)
    bf_set = _token_set(bf)

    lexical = np.zeros(len(cand_idx), dtype=np.float32)
    for i, idx in enumerate(cand_idx):
        tm_set = master_data["token_text_set"][int(idx)]
        bm_set = master_data["token_base_set"][int(idx)]

        jt = (len(tf_set & tm_set) / max(len(tf_set | tm_set), 1)) if tf_set and tm_set else 0.0
        jb = (len(bf_set & bm_set) / max(len(bf_set | bm_set), 1)) if bf_set and bm_set else 0.0
        ot = (len(tf_set & tm_set) / max(min(len(tf_set), len(tm_set)), 1)) if tf_set and tm_set else 0.0
        ob = (len(bf_set & bm_set) / max(min(len(bf_set), len(bm_set)), 1)) if bf_set and bm_set else 0.0
        lexical[i] = 0.30 * jt + 0.30 * jb + 0.20 * ot + 0.20 * ob

    unit_equal = (master_data["unidad"][cand_idx] == str(f_row["Unidad_norm"])).astype(np.float32)
    type_equal = (master_data["tipo"][cand_idx] == str(f_row["TipoContenido"])).astype(np.float32)

    structure = (
        0.10 * unit_equal
        + 0.08 * type_equal
        + 0.24 * _vectorized_sim_rel_scalar(float(f_row["FactorConversion"]), master_data["factor"][cand_idx], 6.0)
        + 0.22 * _vectorized_sim_rel_scalar(float(f_row["ContenidoUnidad"]), master_data["content"][cand_idx], 7.0)
        + 0.22 * _vectorized_sim_rel_scalar(float(f_row["ContenidoTotal"]), master_data["total"][cand_idx], 7.0)
        + 0.08 * _vectorized_sim_rel_scalar(float(f_row["PesoUnitario"]), master_data["peso"][cand_idx], 7.0)
        + 0.06 * _vectorized_sim_rel_scalar(float(f_row["CostoCaja"]), master_data["costo"][cand_idx], 2.5)
    ).astype(np.float32)

    support = (0.58 * lexical + 0.42 * structure).astype(np.float32)
    return pd.DataFrame({
        "LexicalSupport": lexical,
        "StructureSupport": structure,
        "Support": support,
    })


def _topk_idx_from_similarity(sim: np.ndarray, k: int) -> np.ndarray:
    if sim.ndim != 1:
        raise ValueError("sim debe ser 1D")
    if len(sim) <= k:
        return np.argsort(-sim)
    part = np.argpartition(-sim, kth=k - 1)[:k]
    return part[np.argsort(-sim[part])]


def _shortlist_for_positive(
    f_row: pd.Series,
    f_emb: np.ndarray,
    maestro_p: pd.DataFrame,
    maestro_emb: np.ndarray,
    master_data: dict,
    top_n_candidates: int,
) -> pd.DataFrame:
    local_idx = master_data["by_ruc"].get(str(f_row["_ruc_norm"]), np.empty(0, dtype=np.int32))
    if len(local_idx) >= min(20, max(10, top_n_candidates // 2)):
        pool_idx = local_idx
        origin = "RUC"
    else:
        pool_idx = np.arange(len(maestro_p), dtype=np.int32)
        origin = "GLOBAL"

    if len(pool_idx) == 0:
        return maestro_p.iloc[[]].copy()

    sim = np.asarray(maestro_emb[pool_idx] @ f_emb, dtype=np.float32)
    keep_local = _topk_idx_from_similarity(sim, min(top_n_candidates * 3, len(pool_idx)))
    keep_idx = pool_idx[keep_local]

    true_code = str(f_row["__cod_norm__"])
    if true_code:
        keep_idx = keep_idx[master_data["codes_norm"][keep_idx] != true_code]
    if len(keep_idx) == 0:
        return maestro_p.iloc[[]].copy()

    supports = _compute_support_batch(f_row, keep_idx, master_data)
    cand = maestro_p.iloc[keep_idx].copy().reset_index(drop=True)
    cand["EmbeddingScore"] = sim[keep_local][:len(cand)]
    cand["OrigenCandidato"] = origin
    cand = pd.concat([cand, supports], axis=1)

    emb01 = (cand["EmbeddingScore"].astype(np.float32) + 1.0) / 2.0
    cand["RetrievalScore"] = 0.55 * cand["Support"].astype(np.float32) + 0.45 * emb01
    cand = (
        cand.sort_values(["RetrievalScore", "Support", "EmbeddingScore"], ascending=[False, False, False])
        .drop_duplicates(subset=["__cod_norm__"], keep="first")
        .head(top_n_candidates)
        .reset_index(drop=True)
    )
    return cand


def mine_hard_negatives(
    modelo: ModeloHomologadorProductos,
    maestro: pd.DataFrame,
    pares_base: pd.DataFrame,
    top_n_candidates: int = 60,
    k_hard_per_positive: int = 2,
    min_model_score: float = 0.25,
    min_support: float = 0.08,
    max_positives: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    _ = np.random.default_rng(random_state)
    if pares_base.empty:
        return pd.DataFrame(columns=list(pares_base.columns))

    positivos = pares_base[pares_base["label"].astype(int) == 1].drop_duplicates(
        subset=["RucProveedor", "fact_cod", "master_cod", "fact_text"]
    ).reset_index(drop=True)

    if max_positives and len(positivos) > max_positives:
        positivos = positivos.sample(n=max_positives, random_state=random_state).reset_index(drop=True)

    print("Pre-procesando maestro y tokens...")
    maestro_p = preparar_maestro(maestro).copy()
    maestro_p["__cod_norm__"] = maestro_p["CodProducto"].map(_norm_code)
    maestro_p["_ruc_norm"] = maestro_p["RucProveedor"].astype(str).fillna("").str.strip()

    master_data = _build_master_arrays(maestro_p)
    maestro_emb = np.asarray(modelo.encode_prepared_items(maestro_p), dtype=np.float32)

    print(f"Pre-calculando embeddings para {len(positivos)} positivos...")
    fact_rows = _build_fact_frame_from_pairs(positivos)
    pos_for_enc = fact_rows[[
        "Producto_norm", "Producto_base_norm", "Unidad_norm", "TipoContenido",
        "Costo_log", "PesoUnitario", "Factor_log", "ContenidoUnidad_log", "ContenidoTotal_log",
    ]].copy()
    positivos_emb = np.asarray(modelo.encode_prepared_items(pos_for_enc, batch_size=2048), dtype=np.float32)

    all_candidate_pairs: list[pd.DataFrame] = []
    metadata_map: list[tuple[pd.Series, pd.DataFrame]] = []

    print("Buscando candidatos (vectorizado por embeddings + rescoring)...")
    for i in range(len(fact_rows)):
        f_row = fact_rows.iloc[i]
        cand = _shortlist_for_positive(
            f_row=f_row,
            f_emb=positivos_emb[i],
            maestro_p=maestro_p,
            maestro_emb=maestro_emb,
            master_data=master_data,
            top_n_candidates=top_n_candidates,
        )
        if cand.empty:
            continue
        all_candidate_pairs.append(_build_pair_frame(f_row, cand))
        metadata_map.append((f_row, cand))

    if not all_candidate_pairs:
        return pd.DataFrame(columns=list(pares_base.columns))

    df_huge = pd.concat(all_candidate_pairs, ignore_index=True)
    all_scores = modelo.predict_pairs(df_huge, batch_size=2048)

    negatives = []
    current_idx = 0
    for f_row, cand in metadata_map:
        n_cand = len(cand)
        group_scores = all_scores[current_idx: current_idx + n_cand]
        current_idx += n_cand

        cand = cand.copy()
        cand["ScoreModelo"] = group_scores
        emb01 = (cand["EmbeddingScore"].astype(np.float32) + 1.0) / 2.0
        cand["Hardness"] = (
            0.72 * cand["ScoreModelo"].astype(np.float32)
            + 0.18 * cand["Support"].astype(np.float32)
            + 0.10 * emb01
        )
        cand = cand[
            (cand["ScoreModelo"].astype(np.float32) >= float(min_model_score))
            | (cand["Support"].astype(np.float32) >= float(min_support))
        ].copy()
        if cand.empty:
            continue

        cand = cand.sort_values(
            ["Hardness", "ScoreModelo", "Support", "EmbeddingScore"],
            ascending=[False, False, False, False],
        )

        taken = 0
        used_codes = set()
        for _, c in cand.iterrows():
            code = _norm_code(c.get("CodProducto", ""))
            if not code or code in used_codes:
                continue
            negatives.append(_build_negative_row(f_row, c, label=0))
            used_codes.add(code)
            taken += 1
            if taken >= int(k_hard_per_positive):
                break

    return pd.DataFrame(negatives) if negatives else pd.DataFrame(columns=list(pares_base.columns))
