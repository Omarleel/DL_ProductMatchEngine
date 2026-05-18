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
from ml_pipeline.utils.limpieza import normalizar_codigo, normalizar_unidad
from .feature_engineering import add_aux_pair_features


_INTERNAL_MAESTRO_COLUMNS = ("_row_idx", "_ruc_norm")


def _drop_internal_maestro_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(_INTERNAL_MAESTRO_COLUMNS), errors="ignore")


def _drop_internal_maestro_labels(row: pd.Series) -> pd.Series:
    return row.drop(labels=list(_INTERNAL_MAESTRO_COLUMNS), errors="ignore")


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


def _norm_ruc_alias(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _norm_cod_alias(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return normalizar_codigo(text)


def _norm_unit_alias(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return normalizar_unidad(str(value).strip())


def _buscar_conversion_cantidad(
    fila_factura: pd.Series,
    candidato: pd.Series,
    quantity_conversion_lookup: Optional[dict] = None,
) -> Optional[dict]:
    if not quantity_conversion_lookup:
        return None

    ruc = _norm_ruc_alias(fila_factura.get("RucProveedor", ""))
    cod_cpe = _norm_cod_alias(fila_factura.get("CodProducto", ""))
    cod_master = _norm_cod_alias(candidato.get("CodProducto", ""))
    unidad_cpe = _norm_unit_alias(fila_factura.get("UnidadMedidaCompra", ""))
    unidad_compra = _norm_unit_alias(candidato.get("UnidadMedidaCompra", ""))

    if not cod_cpe or not cod_master:
        return None

    candidates = [
        ("exact", (ruc, cod_master, cod_cpe, unidad_cpe, unidad_compra)),
        ("ruc_product", (ruc, cod_master, cod_cpe)),
        ("global_exact", (cod_master, cod_cpe, unidad_cpe, unidad_compra)),
        ("global_product", (cod_master, cod_cpe)),
    ]

    for index_name, key in candidates:
        row = quantity_conversion_lookup.get(index_name, {}).get(key)
        if row:
            row = dict(row)
            row["NivelMatchConversion"] = index_name
            return row

    return None


def _buscar_match_aprendido(
    fila_factura: pd.Series,
    maestro: pd.DataFrame,
    learned_alias_idx: Optional[dict[tuple[str, str], int]] = None,
) -> Optional[pd.Series]:
    """
    Busca una equivalencia supervisada aprendida durante entrenamiento.

    El modelo neural ayuda a generalizar, pero si en el dataset de entrenamiento ya existe
    una relación exacta (RUC, código CPE/factura) -> código maestro, esa señal debe ganar
    antes del ranking aproximado.
    """
    if not learned_alias_idx:
        return None

    key = (
        _norm_ruc_alias(fila_factura.get("RucProveedor", "")),
        _norm_cod_alias(fila_factura.get("CodProducto", "")),
    )
    row_idx = learned_alias_idx.get(key)
    if row_idx is None:
        # Permite aliases globales solo cuando el entrenamiento los dejó sin RUC.
        row_idx = learned_alias_idx.get(("", key[1]))

    if row_idx is None:
        return None

    try:
        return maestro.loc[row_idx]
    except Exception:
        return None


def _build_factura_output_fields(
    f: pd.Series,
    candidato: Optional[pd.Series] = None,
    quantity_conversion_lookup: Optional[dict] = None,
) -> dict:
    """
    Campos comunes de salida para la línea de factura.

    Prioridad para inferir cantidades:
      1. Cantidad facturada * factor histórico del diccionario de conversión.
      2. Fallback ValorTotal / CostoCaja del candidato cuando no hay conversión histórica.

    El booleano UsoFallbackValorTotal queda en True únicamente cuando se usó el fallback.
    """
    candidato = candidato if candidato is not None else pd.Series(dtype=object)

    valor_total = _safe_float(f.get("ValorTotal", 0.0))
    cantidad_factura = _safe_float(f.get("Cantidad", 0.0))
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

    conversion = _buscar_conversion_cantidad(f, candidato, quantity_conversion_lookup)
    factor_conversion_cantidad = 0.0
    conversion_encontrada = False
    conversion_nivel = ""
    conversion_muestras = 0.0
    conversion_confianza = 0.0
    unidad_cpe_usada = _norm_unit_alias(f.get("UnidadMedidaCompra", ""))
    unidad_compra_usada = _norm_unit_alias(candidato.get("UnidadMedidaCompra", ""))

    cantidad_compra = 0.0
    cantidad_unidades = 0.0
    contenido_total_linea = 0.0
    peso_total_kg = 0.0
    cantidad_inferida_desde = ""
    uso_fallback_valor_total = False

    if conversion is not None:
        factor_conversion_cantidad = _safe_float(conversion.get("FactorCantidadCompra", 0.0))
        if cantidad_factura > 0.0 and factor_conversion_cantidad > 0.0:
            cantidad_compra = _round_quantity(cantidad_factura * factor_conversion_cantidad)
            cantidad_inferida_desde = "DiccionarioConversionCantidad"
            conversion_encontrada = True
            conversion_nivel = str(conversion.get("NivelMatchConversion", ""))
            conversion_muestras = _safe_float(conversion.get("MuestrasModa", conversion.get("Muestras", 0.0)))
            conversion_confianza = _safe_float(conversion.get("ConfianzaModa", 0.0))
            unidad_cpe_usada = str(conversion.get("UnidadMedidaCpe", unidad_cpe_usada))
            unidad_compra_usada = str(conversion.get("UnidadMedidaCompra", unidad_compra_usada))

    if cantidad_compra <= 0.0 and valor_total > 0.0 and costo_candidato > 0.0:
        cantidad_compra = _round_quantity(valor_total / costo_candidato)
        cantidad_inferida_desde = "ValorTotal/CostoCajaMaestro"
        uso_fallback_valor_total = True

    cantidad_compra_calc = _safe_float(cantidad_compra)
    if cantidad_compra_calc > 0.0 and factor_para_cantidad > 0.0:
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
        "CantidadFactura": _round_quantity(cantidad_factura),
        "FactorFactura": factor_factura,
        "ContenidoFactura": f["ContenidoUnidad"],
        "ContenidoTotalFactura": f["ContenidoTotal"],
        "TipoContenidoFactura": f["TipoContenido"],
        "Producto_norm_factura": f["Producto_norm"],
        "Unidad_norm_factura": f["Unidad_norm"],
        "Costo_log_factura": f["Costo_log"],
        "CantidadCompraFactura": _round_quantity(cantidad_compra),
        "UnidadCompraCantidadFactura": unidad_compra_usada,
        "CantidadUnidadesFactura": _round_quantity(cantidad_unidades),
        "FactorCantidadUsado": _round_quantity(factor_para_cantidad),
        "CostoCajaCantidadUsado": costo_candidato,
        "FactorConversionCantidadUsado": _round_quantity(factor_conversion_cantidad),
        "UnidadMedidaCpeCantidadUsada": unidad_cpe_usada,
        "ConversionCantidadEncontrada": bool(conversion_encontrada),
        "ConversionCantidadNivel": conversion_nivel,
        "ConversionCantidadMuestras": _round_quantity(conversion_muestras),
        "ConversionCantidadConfianza": _round_quantity(conversion_confianza),
        "UsoFallbackValorTotal": bool(uso_fallback_valor_total),
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

    ruc_key = _norm_ruc_alias(f.get("RucProveedor", ""))
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
        maestro=maestro_p,
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

    # El score del modelo puede estar mal calibrado para candidatos sin evidencia textual.
    # Por eso no debe dominar el ranking si el candidato casi no comparte familia/producto
    # con la factura. En el caso observado, "ENVASE PAMOLSA..." ganaba solo por
    # ScoreModelo alto pese a LexicalSupport ~0.006.
    text_gate = np.clip((lexical - 0.03) / 0.30, 0.0, 1.0)
    support_gate = np.clip((support - 0.12) / 0.40, 0.0, 1.0)
    model_evidence_gate = text_gate * support_gate
    model_score_guarded = model_score * model_evidence_gate

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
            0.20 * model_score_guarded
            + 0.35 * support
            + 0.16 * lexical
            + 0.12 * struct
            + 0.17 * emb_rank
            + close_match_bonus
        )

        gate = np.clip(
            0.15 + 0.85 * (0.55 * lexical + 0.45 * struct),
            0.15,
            1.00,
        )

        cand["ScoreFinal"] = raw_score * gate

    # Veto suave para falsos positivos semánticamente ajenos. No elimina el candidato
    # del top_k, solo evita que gane por una probabilidad neural espuria.
    low_text_support = (lexical < 0.08) & (support < 0.28)
    low_text_cap = 0.04 + 0.28 * support + 0.06 * emb_rank + 0.04 * struct
    cand.loc[low_text_support, "ScoreFinal"] = np.minimum(
        cand.loc[low_text_support, "ScoreFinal"],
        low_text_cap[low_text_support],
    )

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
    learned_alias_idx: Optional[dict[tuple[str, str], int]] = None,
    quantity_conversion_lookup: Optional[dict] = None,
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

    if any(col not in maestro_p.columns for col in _INTERNAL_MAESTRO_COLUMNS):
        maestro_p = maestro_p.copy()
        if "_row_idx" not in maestro_p.columns:
            maestro_p["_row_idx"] = np.arange(len(maestro_p), dtype=np.int32)
        if "_ruc_norm" not in maestro_p.columns:
            maestro_p["_ruc_norm"] = maestro_p["RucProveedor"].map(_norm_ruc_alias)

    if maestro_emb is None or len(maestro_emb) != len(maestro_p):
        maestro_emb = modelo_match.encode_prepared_items(_drop_internal_maestro_columns(maestro_p))

    for _, f in fact_p.iterrows():
        ruc_original = str(f.get("RucProveedor", "")).strip()
        
        exacto = _buscar_match_aprendido(f, maestro_p, learned_alias_idx)
        exacto_origen = "APRENDIDO" if exacto is not None else "EXACTO"

        if exacto is None:
            exacto = buscar_match_exacto(f, maestro_p, idx)
            exacto_origen = "EXACTO" if exacto is not None else exacto_origen
        
        if exacto is None and ruc_original in RUC_EQUIVALENTES:
            f_alt = f.copy()
            f_alt["RucProveedor"] = RUC_EQUIVALENTES[ruc_original]
            
            f_alt["CodProducto"] = str(f_alt.get("CodProducto", "")).lstrip('0')
            
            exacto = buscar_match_exacto(f_alt, maestro_p, idx)
            if exacto is not None:
                exacto_origen = "EXACTO_RUC_EQUIVALENTE"

        if exacto is not None:
            row = _drop_internal_maestro_labels(exacto).to_dict()
            row.update({
                "OrigenCandidato": exacto_origen,
                "EmbeddingScore": 1.0,
                "LexicalSupport": 1.0,
                "StructureSupport": 1.0,
                "Support": 1.0,
                "RetrievalScore": 1.0,
                "ScoreModelo": 1.0,
                "ScoreFinal": 1.0,
                "Score": 1.0,
                "TipoResultado": exacto_origen,
                **_build_factura_output_fields(f, exacto, quantity_conversion_lookup),
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
                **_build_factura_output_fields(f, quantity_conversion_lookup=quantity_conversion_lookup),
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
            row = _drop_internal_maestro_labels(c).to_dict()
            row.update({
                "TipoResultado": tipo if rank == 1 else "ALTERNATIVA",
                **_build_factura_output_fields(f, c, quantity_conversion_lookup),
                "Rank": rank,
            })
            resultados.append(row)

    return pd.DataFrame(resultados)