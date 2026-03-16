from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED
from .matching import construir_indice_codigos, jaccard, similitud_log
from .preparacion import preparar_facturas, preparar_maestro


def _build_maestro_por_ruc(maestro: pd.DataFrame) -> dict[str, pd.DataFrame]:
    maestro_tmp = maestro.copy()
    maestro_tmp["_ruc_key"] = maestro_tmp["RucProveedor"].astype(str).str.strip()

    maestro_por_ruc: dict[str, pd.DataFrame] = {}
    for ruc, group in maestro_tmp.groupby("_ruc_key", sort=False):
        maestro_por_ruc[ruc] = group.drop(columns=["_ruc_key"]).reset_index(drop=True)

    return maestro_por_ruc


def _cached_jaccard_factory():
    cache: dict[tuple[str, str], float] = {}

    def cached(a: str, b: str) -> float:
        key = (str(a), str(b))
        value = cache.get(key)
        if value is None:
            value = float(jaccard(a, b))
            cache[key] = value
        return value

    return cached


def _cached_similitud_log_factory():
    cache: dict[tuple[float, float, float], float] = {}

    def cached(a: float, b: float, escala: float) -> float:
        key = (float(a), float(b), float(escala))
        value = cache.get(key)
        if value is None:
            value = float(similitud_log(a, b, escala=escala))
            cache[key] = value
        return value

    return cached


def resolver_positivos_por_codigo(historial: pd.DataFrame, maestro: pd.DataFrame) -> pd.DataFrame:
    idx = construir_indice_codigos(maestro)
    filas = []

    for f in historial.itertuples(index=False):
        key = (str(f.RucProveedor).strip(), f.CodProducto)
        m_idx = idx.get(key)

        if m_idx is None:
            continue

        m = maestro.loc[m_idx]

        filas.append({
            "fact_cod": f.CodProducto,
            "fact_text": f.Producto_norm,
            "fact_base_text": f.Producto_base_norm,
            "fact_unit": f.Unidad_norm,
            "fact_type": f.TipoContenido,
            "fact_cost": f.Costo_log,
            "fact_peso": f.PesoUnitario,
            "fact_factor": f.Factor_log,
            "fact_content": f.ContenidoUnidad_log,
            "fact_total": f.ContenidoTotal_log,
            "master_cod": m["CodProducto"],
            "master_text": m["Producto_norm"],
            "master_base_text": m["Producto_base_norm"],
            "master_unit": m["Unidad_norm"],
            "master_type": m["TipoContenido"],
            "master_cost": m["Costo_log"],
            "master_peso": m["PesoUnitario"],
            "master_factor": m["Factor_log"],
            "master_content": m["ContenidoUnidad_log"],
            "master_total": m["ContenidoTotal_log"],
            "label": 1,
            "RucProveedor": f.RucProveedor,
        })

    return pd.DataFrame(filas)


def muestrear_negativos(
    positivos: pd.DataFrame,
    maestro: pd.DataFrame,
    n_neg_por_pos: int = 5,
) -> pd.DataFrame:
    negativos = []
    rng = np.random.default_rng(SEED)

    maestro_por_ruc = _build_maestro_por_ruc(maestro)
    cached_jaccard = _cached_jaccard_factory()
    cached_similitud_log = _cached_similitud_log_factory()

    for p in positivos.itertuples(index=False):
        ruc_key = str(p.RucProveedor).strip()
        maestro_ruc = maestro_por_ruc.get(ruc_key)

        if maestro_ruc is None or maestro_ruc.empty:
            continue

        pool = maestro_ruc[maestro_ruc["CodProducto"] != p.master_cod].copy()
        if pool.empty:
            continue

        fact_text = str(p.fact_text)
        fact_base_text = str(p.fact_base_text)
        fact_unit = str(p.fact_unit)
        fact_type = str(p.fact_type)

        fact_cost = float(p.fact_cost)
        fact_factor = float(p.fact_factor)
        fact_content = float(p.fact_content)
        fact_total = float(p.fact_total)
        fact_peso = float(p.fact_peso)

        pool["sim_text"] = pool["Producto_norm"].map(
            lambda x: cached_jaccard(fact_text, str(x))
        )
        pool["sim_base"] = pool["Producto_base_norm"].map(
            lambda x: cached_jaccard(fact_base_text, str(x))
        )
        pool["sim_cost"] = pool["Costo_log"].map(
            lambda x: cached_similitud_log(fact_cost, float(x), 1.4)
        )
        pool["sim_factor"] = pool["Factor_log"].map(
            lambda x: cached_similitud_log(fact_factor, float(x), 2.8)
        )
        pool["sim_content"] = pool["ContenidoUnidad_log"].map(
            lambda x: cached_similitud_log(fact_content, float(x), 2.8)
        )
        pool["sim_total"] = pool["ContenidoTotal_log"].map(
            lambda x: cached_similitud_log(fact_total, float(x), 2.2)
        )
        pool["sim_peso"] = pool["PesoUnitario"].map(
            lambda x: cached_similitud_log(fact_peso, float(x), 2.0)
        )

        pool["same_type"] = (pool["TipoContenido"].astype(str) == fact_type).astype(float)
        pool["same_unit"] = (pool["Unidad_norm"].astype(str) == fact_unit).astype(float)

        pool["family_overlap"] = (
            (pool["sim_base"] >= 0.45).astype(float)
            + (pool["sim_text"] >= 0.35).astype(float)
        ) / 2.0

        pool["presentation_close"] = (
            0.35 * pool["sim_factor"]
            + 0.25 * pool["sim_content"]
            + 0.25 * pool["sim_total"]
            + 0.15 * pool["sim_peso"]
        )

        pool["same_presentation_band"] = (
            (pool["presentation_close"] >= 0.82).astype(float)
        )

        pool["subtype_tension"] = np.clip(
            pool["sim_base"] - pool["sim_text"],
            0.0,
            1.0,
        )

        pool["hardness"] = (
            0.28 * pool["sim_text"]
            + 0.22 * pool["sim_base"]
            + 0.24 * pool["presentation_close"]
            + 0.08 * pool["same_type"]
            + 0.05 * pool["same_unit"]
            + 0.05 * pool["sim_cost"]
            + 0.08 * pool["subtype_tension"]
        )

        pool["hardness_subtipo"] = (
            0.34 * pool["sim_base"]
            + 0.28 * pool["presentation_close"]
            + 0.12 * pool["same_type"]
            + 0.08 * pool["same_unit"]
            + 0.18 * pool["subtype_tension"]
        )

        hermanos = pool[
            (pool["sim_base"] >= 0.50)
            & (pool["presentation_close"] >= 0.78)
        ].copy()

        if hermanos.empty:
            hermanos = pool[
                (pool["sim_base"] >= 0.40)
                & (pool["presentation_close"] >= 0.70)
            ].copy()

        top_general_n = max(n_neg_por_pos * 6, 16)
        top_subtipo_n = max(n_neg_por_pos * 4, 10)

        top_general = pool.nlargest(top_general_n, columns="hardness")

        if not hermanos.empty:
            top_subtipo = hermanos.nlargest(top_subtipo_n, columns="hardness_subtipo")
        else:
            top_subtipo = pool.nlargest(top_subtipo_n, columns="hardness_subtipo")

        top_hard = pd.concat([top_general, top_subtipo], ignore_index=True)
        top_hard = top_hard.drop_duplicates(subset=["CodProducto"])

        n_hard = min(len(top_hard), max(1, int(round(n_neg_por_pos * 0.8))))
        n_easy = max(0, n_neg_por_pos - n_hard)

        if len(top_hard) > n_hard:
            weights = top_hard["hardness_subtipo"].fillna(0.0).values.astype(np.float64)
            weights = np.clip(weights, 1e-6, None)
            weights = weights / weights.sum()

            chosen_idx = rng.choice(
                len(top_hard),
                size=n_hard,
                replace=False,
                p=weights,
            )
            hard_sel = top_hard.iloc[chosen_idx]
        else:
            hard_sel = top_hard

        used = set(hard_sel["CodProducto"].astype(str).tolist())

        remaining = pool.loc[
            ~pool["CodProducto"].astype(str).isin(used)
        ].copy()

        if n_easy > 0 and not remaining.empty:
            easy_pool = remaining.copy()

            # Preferir algunos "semi-duros" antes que totalmente aleatorios
            easy_pool["easy_weight"] = (
                0.45 * easy_pool["sim_text"]
                + 0.25 * easy_pool["sim_base"]
                + 0.20 * easy_pool["presentation_close"]
                + 0.10 * easy_pool["sim_cost"]
            )

            easy_pool = easy_pool.nlargest(max(n_easy * 4, 8), columns="easy_weight")

            if len(easy_pool) > n_easy:
                easy_weights = easy_pool["easy_weight"].fillna(0.0).values.astype(np.float64)
                easy_weights = np.clip(easy_weights, 1e-6, None)
                easy_weights = easy_weights / easy_weights.sum()

                easy_idx = rng.choice(
                    len(easy_pool),
                    size=n_easy,
                    replace=False,
                    p=easy_weights,
                )
                easy_sel = easy_pool.iloc[easy_idx]
            else:
                easy_sel = easy_pool

            seleccionados = pd.concat([hard_sel, easy_sel], ignore_index=True)
        else:
            seleccionados = hard_sel.reset_index(drop=True)

        seleccionados = (
            seleccionados
            .drop_duplicates(subset=["CodProducto"])
            .head(n_neg_por_pos)
        )

        for m in seleccionados.itertuples(index=False):
            negativos.append({
                "fact_cod": p.fact_cod,
                "fact_text": p.fact_text,
                "fact_base_text": p.fact_base_text,
                "fact_unit": p.fact_unit,
                "fact_type": p.fact_type,
                "fact_cost": p.fact_cost,
                "fact_peso": p.fact_peso,
                "fact_factor": p.fact_factor,
                "fact_content": p.fact_content,
                "fact_total": p.fact_total,
                "master_cod": m.CodProducto,
                "master_text": m.Producto_norm,
                "master_base_text": m.Producto_base_norm,
                "master_unit": m.Unidad_norm,
                "master_type": m.TipoContenido,
                "master_cost": m.Costo_log,
                "master_peso": m.PesoUnitario,
                "master_factor": m.Factor_log,
                "master_content": m.ContenidoUnidad_log,
                "master_total": m.ContenidoTotal_log,
                "label": 0,
                "RucProveedor": p.RucProveedor,
            })

    return pd.DataFrame(negativos)


def construir_dataset_entrenamiento(
    maestro: pd.DataFrame,
    productos_facturas: pd.DataFrame,
    n_neg_por_pos: int = 5,
) -> pd.DataFrame:
    maestro_p = preparar_maestro(maestro)
    hist_p = preparar_facturas(productos_facturas)

    positivos = resolver_positivos_por_codigo(hist_p, maestro_p)

    if positivos.empty:
        raise ValueError(
            "No se generaron positivos automáticos. Necesitas facturas históricas "
            "con coincidencias exactas entre el código de factura y alguno de los "
            "3 códigos del maestro."
        )

    negativos = muestrear_negativos(positivos, maestro_p, n_neg_por_pos=n_neg_por_pos)

    pares = pd.concat([positivos, negativos], ignore_index=True)
    pares = pares.drop_duplicates(
        subset=[
            "fact_cod",
            "master_cod",
            "fact_text",
            "master_text",
            "label",
        ]
    )
    pares = pares.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return pares