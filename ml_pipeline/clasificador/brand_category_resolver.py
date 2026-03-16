from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from ml_pipeline.utils.preparacion import preparar_facturas, preparar_maestro
from ml_pipeline.clasificador.factor_resolver import _jaccard


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


@dataclass
class MaestroBrandCategoryResolver:
    maestro: pd.DataFrame
    min_score: float = 0.55

    def __post_init__(self):
        maestro_p = preparar_maestro(self.maestro).copy()

        # Solo filas con algo útil para resolver
        if "target_marca" not in maestro_p.columns:
            maestro_p["target_marca"] = maestro_p.get("Marca", "SIN_MARCA")
        if "target_categoria" not in maestro_p.columns:
            maestro_p["target_categoria"] = maestro_p.get("Categoria", "SIN_CATEGORIA")

        maestro_p["RucProveedor"] = maestro_p.get("RucProveedor", "").astype(str).str.strip()
        maestro_p["CodProducto"] = maestro_p.get("CodProducto", "").astype(str).str.strip()
        maestro_p["Producto_base_norm"] = maestro_p.get("Producto_base_norm", maestro_p.get("Producto", "")).fillna("").astype(str)

        self.maestro_p = maestro_p
        self.idx_exact = {
            (row["RucProveedor"], row["CodProducto"]): i
            for i, row in maestro_p.reset_index(drop=True).iterrows()
        }
        self.maestro_p = maestro_p.reset_index(drop=True)
        self.by_ruc = {ruc: grp for ruc, grp in self.maestro_p.groupby("RucProveedor", dropna=False)}

    def _resolve_one(self, row: pd.Series) -> dict:
        ruc = _safe_str(row.get("RucProveedor", ""))
        cod = _safe_str(row.get("CodProducto", ""))
        text = _safe_str(row.get("Producto_base_norm", row.get("Producto", "")))

        # 1) Exacto por proveedor + código
        idx = self.idx_exact.get((ruc, cod))
        if idx is not None:
            m = self.maestro_p.loc[idx]
            return {
                "brand_category_source": "maestro_exact",
                "brand_category_match_score": 1.0,
                "brand_category_match_cod": _safe_str(m.get("CodProducto", "")),
                "brand_category_match_producto": _safe_str(m.get("Producto", "")),
                "resolved_marca": _safe_str(m.get("target_marca", "SIN_MARCA")),
                "resolved_categoria": _safe_str(m.get("target_categoria", "SIN_CATEGORIA")),
            }

        # 2) Fuzzy por proveedor
        grupo = self.by_ruc.get(ruc)
        if grupo is None or grupo.empty or not text:
            return {
                "brand_category_source": "modelo_nn",
                "brand_category_match_score": 0.0,
                "brand_category_match_cod": "",
                "brand_category_match_producto": "",
                "resolved_marca": None,
                "resolved_categoria": None,
            }

        sims = grupo["Producto_base_norm"].apply(lambda x: _jaccard(text, _safe_str(x)))
        if sims.empty:
            return {
                "brand_category_source": "modelo_nn",
                "brand_category_match_score": 0.0,
                "brand_category_match_cod": "",
                "brand_category_match_producto": "",
                "resolved_marca": None,
                "resolved_categoria": None,
            }

        best_idx = sims.idxmax()
        best_score = float(sims.loc[best_idx])

        if best_score < self.min_score:
            return {
                "brand_category_source": "modelo_nn",
                "brand_category_match_score": best_score,
                "brand_category_match_cod": "",
                "brand_category_match_producto": "",
                "resolved_marca": None,
                "resolved_categoria": None,
            }

        m = grupo.loc[best_idx]
        return {
            "brand_category_source": "maestro_fuzzy",
            "brand_category_match_score": best_score,
            "brand_category_match_cod": _safe_str(m.get("CodProducto", "")),
            "brand_category_match_producto": _safe_str(m.get("Producto", "")),
            "resolved_marca": _safe_str(m.get("target_marca", "SIN_MARCA")),
            "resolved_categoria": _safe_str(m.get("target_categoria", "SIN_CATEGORIA")),
        }

    def resolve_many(self, productos_facturas: pd.DataFrame) -> pd.DataFrame:
        fact_p = preparar_facturas(productos_facturas).copy()
        rows = [self._resolve_one(r) for _, r in fact_p.iterrows()]
        return pd.DataFrame(rows)