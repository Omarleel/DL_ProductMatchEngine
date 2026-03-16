import numpy as np
import pandas as pd

class SampleWeightStrategy:
    @staticmethod
    def compute(train_df: pd.DataFrame) -> np.ndarray:
        n_pos = max(int((train_df["label"] == 1).sum()), 1)
        n_neg = max(int((train_df["label"] == 0).sum()), 1)

        sample_weight = np.ones(len(train_df), dtype=np.float32)
        labels = train_df["label"].values

        sample_weight[labels == 1] = n_neg / n_pos
        neg_mask = labels == 0

        cost_gap = np.abs(train_df["fact_cost"].values - train_df["master_cost"].values)
        factor_gap = np.abs(train_df["fact_factor"].values - train_df["master_factor"].values)
        content_gap = np.abs(train_df["fact_content"].values - train_df["master_content"].values)
        total_gap = np.abs(train_df["fact_total"].values - train_df["master_total"].values)

        peso_a = train_df["fact_peso"].values.astype(np.float32)
        peso_b = train_df["master_peso"].values.astype(np.float32)
        peso_rel_gap = np.abs(peso_a - peso_b) / np.maximum(
            np.maximum(np.abs(peso_a), np.abs(peso_b)),
            1e-6,
        )

        same_type = (
            train_df["fact_type"].astype(str).values
            == train_df["master_type"].astype(str).values
        ).astype(np.float32)

        known_type = (
            (train_df["fact_type"].astype(str).values != "NONE")
            & (train_df["master_type"].astype(str).values != "NONE")
        ).astype(np.float32)

        same_base = (
            train_df["fact_base_text"].astype(str).str.strip().values
            == train_df["master_base_text"].astype(str).str.strip().values
        ).astype(np.float32)

        dificultad_neg = (
            0.18 * np.exp(-1.2 * cost_gap)
            + 0.18 * np.exp(-2.0 * factor_gap)
            + 0.16 * np.exp(-2.0 * content_gap)
            + 0.26 * np.exp(-2.2 * total_gap)
            + 0.12 * np.exp(-4.0 * peso_rel_gap)
            + 0.10 * (same_type * known_type)
        )

        conflicto_presentacion = (
            0.30 * (factor_gap > 0.10).astype(np.float32)
            + 0.20 * (content_gap > 0.12).astype(np.float32)
            + 0.30 * (total_gap > 0.12).astype(np.float32)
            + 0.20 * ((same_type == 0).astype(np.float32) * known_type)
        )

        bonus_mismo_nombre = 1.0 + 0.60 * same_base

        sample_weight[neg_mask] *= (
            1.0
            + (2.4 * dificultad_neg[neg_mask] * bonus_mismo_nombre[neg_mask])
            + (1.6 * conflicto_presentacion[neg_mask] * bonus_mismo_nombre[neg_mask])
        )

        return np.clip(sample_weight, 1.0, 6.0)

