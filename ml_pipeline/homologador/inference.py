from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ml_pipeline.utils.equivalents import RUC_EQUIVALENTES

from .model import ModeloHomologadorProductos
from .pair_support import attach_support_features
from ml_pipeline.utils.matching import (
    buscar_match_exacto,
    construir_indice_codigos,
    recuperar_candidatos,
)
from ml_pipeline.utils.preparacion import preparar_facturas, preparar_maestro
from .feature_engineering import add_aux_pair_features


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return default


def _first_positive(*values) -> float:
    for value in values:
        numeric = _safe_float(value)
        if numeric > 0.0:
            return numeric
    return 0.0


def _round_quantity(value: float, decimals: int = 6):
    value = _safe_float(value)
    if value <= 0.0:
        return 0.0

    nearest = round(value)
    if abs(value - nearest) <= 1e-3:
        return int(nearest)

    return round(value, decimals)


def _build_factura_output_fields(f: pd.Series, candidato: Optional[pd.Series] = None) -> dict:
    """
    Campos comunes de salida para la línea de factura.

    Si la factura trae ValorTotal y ya existe un candidato homologado,
    se infiere la cantidad de presentaciones de compra así:

        CantidadCompraFactura = ValorTotalFactura / CostoCaja del candidato

    Luego se infieren las unidades comerciales usando el factor del candidato
    (FactorVenta o FactorConversion) para evitar usar una presentación errónea
    extraída desde la unidad de la factura, por ejemplo PALETAS.
    """
    candidato = candidato if candidato is not None else pd.Series(dtype=object)

    valor_total = _safe_float(f.get("ValorTotal", 0.0))
    costo_candidato = _safe_float(candidato.get("CostoCaja", 0.0))

    factor_candidato = _first_positive(
        candidato.get("FactorVenta", 0.0),
        candidato.get("FactorConversion", 0.0),
    )
    factor_factura = _safe_float(f.get("FactorConversion", 0.0))
    factor_para_cantidad = _first_positive(factor_candidato, factor_factura)

    contenido_unidad = _first_positive(
        candidato.get("ContenidoUnidad", 0.0),
        f.get("ContenidoUnidad", 0.0),
    )
    peso_unitario = _first_positive(
        candidato.get("PesoUnitario", 0.0),
        f.get("PesoUnitario", 0.0),
        candidato.get("PesoExtraidoKg", 0.0),
        f.get("PesoExtraidoKg", 0.0),
    )

    cantidad_compra = 0.0
    cantidad_unidades = 0.0
    contenido_total_linea = 0.0
    peso_total_kg = 0.0
    cantidad_inferida_desde = ""

    if valor_total > 0.0 and costo_candidato > 0.0:
        cantidad_compra = _round_quantity(valor_total / costo_candidato)
        cantidad_compra_calc = _safe_float(cantidad_compra)
        cantidad_inferida_desde = "ValorTotal/CostoCajaMaestro"

        if factor_para_cantidad > 0.0:
            cantidad_unidades = _round_quantity(cantidad_compra_calc * factor_para_cantidad)

        cantidad_unidades_calc = _safe_float(cantidad_unidades)

        if contenido_unidad > 0.0 and cantidad_unidades_calc > 0.0:
            contenido_total_linea = cantidad_unidades_calc * contenido_unidad

        if peso_unitario > 0.0 and cantidad_unidades_calc > 0.0:
            peso_total_kg = cantidad_unidades_calc * peso_unitario

    return {
        "CodFactura": f["CodProducto"],
        "ProductoFactura": f["Producto"],
        "ProductoFacturaBase": f["Producto_base_norm"],
        "UnidadFactura": f["UnidadMedidaCompra"],
        "CostoFactura": f["CostoCaja"],
        "ValorTotalFactura": valor_total,
        "FactorFactura": factor_factura,
        "ContenidoFactura": f["ContenidoUnidad"],
        "ContenidoTotalFactura": f["ContenidoTotal"],
        "TipoContenidoFactura": f["TipoContenido"],
        "Producto_norm_factura": f["Producto_norm"],
        "Unidad_norm_factura": f["Unidad_norm"],
        "Costo_log_factura": f["Costo_log"],
        "CantidadCompraFactura": _round_quantity(cantidad_compra),
        "UnidadCompraCantidadFactura": candidato.get("UnidadMedidaCompra", ""),
        "CantidadUnidadesFactura": _round_quantity(cantidad_unidades),
        "FactorCantidadUsado": _round_quantity(factor_para_cantidad),
        "CostoCajaCantidadUsado": costo_candidato,
        "ContenidoTotalLineaFactura": _round_quantity(contenido_total_linea),
        "PesoTotalKgFactura": _round_quantity(peso_total_kg),
        "CantidadInferidaDesde": cantidad_inferida_desde,
    }


def _build_pair_frame(f: pd.Series, cand: pd.DataFrame) -> pd.DataFrame:
    pares = pd.DataFrame({
        "fact_text": [f["Producto_norm"]] * len(cand),
        "fact_base_text": [f["Producto_base_norm"]] * len(cand),
        "fact_unit": [f["Unidad_norm"]] * len(cand),
        "fact_type": [f["TipoContenido"]] * len(cand),
        "fact_cost": [f["Costo_log"]] * len(cand),
        "fact_peso": [f["PesoUnitario"]] * len(cand),
        "fact_factor": [f["Factor_log"]] * len(cand),
        "fact_content": [f["ContenidoUnidad_log"]] * len(cand),
        "fact_total": [f["ContenidoTotal_log"]] * len(cand),

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
    })

    pares = add_aux_pair_features(pares)
    return pares


def _embedding_shortlist(
    f: pd.Series,
    maestro_p: pd.DataFrame,
    maestro_emb: np.ndarray,
    modelo_match: ModeloHomologadorProductos,
    top_n: int,
) -> tuple[pd.DataFrame, str]:
    f_df = pd.DataFrame([f])
    fact_emb = modelo_match.encode_prepared_items(f_df)[0]

    ruc_key = str(f["RucProveedor"]).strip()
    local_mask = (maestro_p["_ruc_norm"] == ruc_key).values

    if int(local_mask.sum()) >= min(20, max(10, top_n // 2)):
        candidate_idx = np.where(local_mask)[0]
        origen = "RUC"
    else:
        candidate_idx = np.arange(len(maestro_p))
        origen = "GLOBAL"

    emb_subset = maestro_emb[candidate_idx]
    sim = emb_subset @ fact_emb

    take = min(top_n, len(candidate_idx))
    if take <= 0:
        return maestro_p.iloc[[]].copy(), origen

    top_local = np.argpartition(-sim, kth=max(take - 1, 0))[:take]
    top_local = top_local[np.argsort(-sim[top_local])]
    chosen_idx = candidate_idx[top_local]

    cand = maestro_p.iloc[chosen_idx].copy().reset_index(drop=True)
    cand["EmbeddingScore"] = sim[top_local]
    cand["OrigenCandidato"] = origen
    return cand, origen


def _hybrid_candidates(
    f: pd.Series,
    maestro_p: pd.DataFrame,
    maestro_emb: np.ndarray,
    modelo_match: ModeloHomologadorProductos,
    top_n_embedding: int = 80,
    top_n_lexical: int = 80,
) -> pd.DataFrame:
    cand_emb, origen = _embedding_shortlist(
        f=f,
        maestro_p=maestro_p,
        maestro_emb=maestro_emb,
        modelo_match=modelo_match,
        top_n=top_n_embedding,
    )

    cand_lex = recuperar_candidatos(
        fila_factura=f,
        maestro=maestro_p.drop(columns=["_row_idx", "_ruc_norm"], errors="ignore"),
        top_n=top_n_lexical,
    ).copy()

    if "EmbeddingScore" not in cand_lex.columns:
        cand_lex["EmbeddingScore"] = 0.0
    cand_lex["OrigenCandidato"] = cand_lex.get("OrigenCandidato", origen)

    cand = pd.concat([cand_emb, cand_lex], ignore_index=True)
    cand = cand.drop_duplicates(subset=["CodProducto"], keep="first").reset_index(drop=True)

    if cand.empty:
        return cand

    cand = attach_support_features(f, cand)

    emb01 = (cand["EmbeddingScore"].astype(float) + 1.0) / 2.0
    cand["RetrievalScore"] = (
        0.55 * cand["Support"].astype(float)
        + 0.45 * emb01
    )

    cand = cand.sort_values(
        ["RetrievalScore", "Support", "LexicalSupport", "EmbeddingScore"],
        ascending=[False, False, False, False],
    ).head(max(top_n_embedding, top_n_lexical))

    return cand.reset_index(drop=True)


def _compute_final_scores(cand: pd.DataFrame) -> pd.DataFrame:
    cand = cand.copy()

    model_score = cand["ScoreModelo"].astype(float)
    lexical = cand["LexicalSupport"].astype(float)
    struct = cand["StructureSupport"].astype(float)
    support = cand["Support"].astype(float)

    emb01 = (cand["EmbeddingScore"].astype(float) + 1.0) / 2.0

    emb_rank = np.where(
        support >= 0.72,
        np.maximum(emb01, 0.55),
        emb01,
    )

    close_match_bonus = np.where(
        (support >= 0.72)
        & (lexical >= 0.55)
        & (cand["FactorSim"].astype(float) >= 0.98)
        & (cand["ContentSim"].astype(float) >= 0.98)
        & (cand["TotalSim"].astype(float) >= 0.98),
        0.05,
        0.0,
    )

    model_spread = float(model_score.max() - model_score.min()) if len(model_score) > 1 else 0.0
    model_max = float(model_score.max()) if len(model_score) > 0 else 0.0
    model_collapse = (model_max < 0.15) or (model_spread < 0.03)

    if model_collapse:
        cand["ScoreFinal"] = (
            0.52 * support
            + 0.25 * lexical
            + 0.11 * struct
            + 0.08 * emb_rank
            + close_match_bonus
        )
    else:
        raw_score = (
            0.45 * model_score
            + 0.25 * support
            + 0.12 * lexical
            + 0.10 * struct
            + 0.08 * emb_rank
            + close_match_bonus
        )

        gate = np.clip(
            0.35 + 0.65 * (0.55 * lexical + 0.45 * struct),
            0.35,
            1.00,
        )

        cand["ScoreFinal"] = raw_score * gate

    cand["Score"] = cand["ScoreFinal"]
    return cand


def inferir_codproducto_homologador(
    productos_facturas: pd.DataFrame,
    maestro_p: Optional[pd.DataFrame] = None,
    idx: Optional[dict] = None,
    maestro_emb: Optional[np.ndarray] = None,
    modelo_match: Optional[ModeloHomologadorProductos] = None,
    top_k: int = 5,
    umbral_match: Optional[float] = None,
    top_n_candidates: int = 80,
    maestro: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Infiere el CodProducto usando un maestro preparado o un maestro crudo.
    """
    if modelo_match is None:
        raise TypeError("modelo_match es requerido para inferir homologación")

    if maestro_p is None:
        if maestro is None:
            raise TypeError("Debes enviar maestro_p cacheado o maestro crudo")
        maestro_p = preparar_maestro(maestro)
    else:
        columnas_preparadas = {
            "Producto_norm",
            "Producto_base_norm",
            "Unidad_norm",
            "TipoContenido",
            "Costo_log",
        }
        if not columnas_preparadas.issubset(set(maestro_p.columns)):
            maestro_p = preparar_maestro(maestro_p)

    if idx is None:
        idx = construir_indice_codigos(maestro_p)

    if umbral_match is None:
        umbral_match = getattr(modelo_match, "best_threshold", 0.72)

    fact_p = preparar_facturas(productos_facturas)
    resultados: list[dict] = []

    maestro_p = maestro_p.copy()
    maestro_p["_row_idx"] = np.arange(len(maestro_p), dtype=np.int32)
    maestro_p["_ruc_norm"] = maestro_p["RucProveedor"].astype(str).fillna("").str.strip()

    if maestro_emb is None or len(maestro_emb) != len(maestro_p):
        maestro_emb = modelo_match.encode_prepared_items(
            maestro_p.drop(columns=["_row_idx", "_ruc_norm"], errors="ignore")
        )

    for _, f in fact_p.iterrows():
        ruc_original = str(f.get("RucProveedor", "")).strip()
        
        exacto = buscar_match_exacto(f, maestro_p, idx)
        
        if exacto is None and ruc_original in RUC_EQUIVALENTES:
            f_alt = f.copy()
            f_alt["RucProveedor"] = RUC_EQUIVALENTES[ruc_original]
            
            f_alt["CodProducto"] = str(f_alt.get("CodProducto", "")).lstrip('0')
            
            exacto = buscar_match_exacto(f_alt, maestro_p, idx)

        if exacto is not None:
            row = exacto.drop(labels=["_row_idx", "_ruc_norm"], errors="ignore").to_dict()
            row.update({
                "OrigenCandidato": "EXACTO",
                "EmbeddingScore": 1.0,
                "LexicalSupport": 1.0,
                "StructureSupport": 1.0,
                "Support": 1.0,
                "RetrievalScore": 1.0,
                "ScoreModelo": 1.0,
                "ScoreFinal": 1.0,
                "Score": 1.0,
                "TipoResultado": "EXACTO",
                **_build_factura_output_fields(f, exacto),
                "Rank": 1,
            })
            resultados.append(row)
            continue

        cand = _hybrid_candidates(
            f=f,
            maestro_p=maestro_p,
            maestro_emb=maestro_emb,
            modelo_match=modelo_match,
            top_n_embedding=top_n_candidates,
            top_n_lexical=top_n_candidates,
        )

        if cand.empty:
            resultados.append({
                "RucProveedor": f["RucProveedor"],
                **_build_factura_output_fields(f),
                "TipoResultado": "SIN_CANDIDATOS",
                "Score": 0.0,
                "Rank": 1,
            })
            continue

        pares = _build_pair_frame(f, cand)
        cand["ScoreModelo"] = modelo_match.predict_pairs(pares)

        cand = _compute_final_scores(cand)

        cand = cand.sort_values(
            ["ScoreFinal", "Support", "LexicalSupport", "StructureSupport", "ScoreModelo", "EmbeddingScore"],
            ascending=[False, False, False, False, False, False],
        ).head(top_k)

        top1 = cand.iloc[0]
        top2_score = float(cand.iloc[1]["ScoreFinal"]) if len(cand) > 1 else 0.0
        margin = float(top1["ScoreFinal"]) - top2_score

        tentative_ok = (
            float(top1["ScoreFinal"]) >= float(umbral_match)
            and float(top1["Support"]) >= 0.30
            and (
                margin >= 0.03
                or float(top1["Support"]) >= 0.55
            )
            and not (
                float(top1["LexicalSupport"]) < 0.12
                and float(top1["StructureSupport"]) < 0.35
            )
        )

        tipo = "TENTATIVO" if tentative_ok else "POSIBLE_NUEVO_PRODUCTO"

        for rank, (_, c) in enumerate(cand.iterrows(), start=1):
            row = c.drop(labels=["_row_idx", "_ruc_norm"], errors="ignore").to_dict()
            row.update({
                "TipoResultado": tipo if rank == 1 else "ALTERNATIVA",
                **_build_factura_output_fields(f, c),
                "Rank": rank,
            })
            resultados.append(row)

    return pd.DataFrame(resultados)