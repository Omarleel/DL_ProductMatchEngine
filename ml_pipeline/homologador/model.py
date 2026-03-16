from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from ml_pipeline.core.thresholding import ThresholdOptimizer
from ml_pipeline.core.weighting import SampleWeightStrategy
from .builder import MatchModelBuilder2
from .config import Model2Config
from .dataset import DatasetBuilder2
from .persistence import ModelPersistence2
from .preprocessing import PreprocessingAssets2
from .feature_engineering import AUX_FEATURE_COLUMNS, add_aux_pair_features


class ModeloHomologadorProductos:
    def __init__(
        self,
        max_tokens: int = 18000,
        max_char_tokens: int = 160,
        word_seq_len: int = 32,
        char_seq_len: int = 96,
        text_embedding_dim: int = 96,
        char_embedding_dim: int = 32,
        item_embedding_dim: int = 128,
        unit_embedding_dim: int = 12,
        type_embedding_dim: int = 8,
        dropout_rate: float = 0.20,
        l2_reg: float = 1e-5,
        learning_rate: float = 8e-4,
    ):
        self.config = Model2Config(
            max_tokens=max_tokens,
            max_char_tokens=max_char_tokens,
            word_seq_len=word_seq_len,
            char_seq_len=char_seq_len,
            text_embedding_dim=text_embedding_dim,
            char_embedding_dim=char_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            unit_embedding_dim=unit_embedding_dim,
            type_embedding_dim=type_embedding_dim,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
        )
        self.assets = PreprocessingAssets2(self.config)
        self.model: Optional[tf.keras.Model] = None
        self.item_encoder: Optional[tf.keras.Model] = None
        self.best_threshold: float = 0.72

    def guardar(self, carpeta_modelo: str) -> None:
        ModelPersistence2.save(self, carpeta_modelo)

    @classmethod
    def cargar(cls, carpeta_modelo: str) -> "ModeloHomologadorProductos":
        config = ModelPersistence2.read_config(carpeta_modelo)
        instancia = cls(**asdict(config))
        ModelPersistence2.load(instancia, carpeta_modelo)
        return instancia

    def construir(self) -> tf.keras.Model:
        builder = MatchModelBuilder2(self.config, self.assets)
        self.model, self.item_encoder = builder.build()
        return self.model

    def adaptar_activos(self, pares: pd.DataFrame) -> None:
        self.assets.adapt_vocabularies(pares)

    def _ensure_aux_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if all(c in df.columns for c in AUX_FEATURE_COLUMNS):
            return df
        return add_aux_pair_features(df)

    @staticmethod
    def _to_ds(
        df: pd.DataFrame,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: int = 256,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        return DatasetBuilder2.to_dataset(
            df=df,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def split_train_valid(
        self,
        pares: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        groups = (
            pares["RucProveedor"].astype(str).fillna("")
            + "|"
            + pares["fact_cod"].astype(str).fillna("")
            + "|"
            + pares["fact_text"].astype(str).fillna("")
        )
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, valid_idx = next(gss.split(pares, pares["label"], groups=groups))
        return (
            pares.iloc[train_idx].copy().reset_index(drop=True),
            pares.iloc[valid_idx].copy().reset_index(drop=True),
        )

    def fit(
        self,
        pares: pd.DataFrame,
        epochs: int = 16,
        batch_size: int = 256,
    ) -> dict:
        train_df, valid_df = self.split_train_valid(pares)
        return self.fit_on_split(
            train_df=train_df,
            valid_df=valid_df,
            epochs=epochs,
            batch_size=batch_size,
        )

    def fit_on_split(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        epochs: int = 16,
        batch_size: int = 256,
    ) -> dict:
        train_df = self._ensure_aux_features(train_df)
        valid_df = self._ensure_aux_features(valid_df)

        self.adaptar_activos(train_df)
        self.construir()
        self.assets.adapt_normalizers(train_df)

        sample_weight = SampleWeightStrategy.compute(train_df)

        ds_train = self._to_ds(
            train_df,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=True,
        )
        ds_valid = self._to_ds(valid_df, batch_size=batch_size, shuffle=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        history = self.model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        y_valid = valid_df["label"].astype(int).values
        probs_valid = self.model.predict(ds_valid, verbose=0).reshape(-1)

        best_th, best_f1, best_precision, best_recall = ThresholdOptimizer.find_best(
            y_true=y_valid,
            probs=probs_valid,
        )
        self.best_threshold = best_th

        pair_metrics = self.evaluate_pairs(valid_df, batch_size=batch_size)
        ranking_metrics = self.evaluate_ranking(valid_df, batch_size=batch_size)

        return {
            "history": {k: [float(x) for x in v] for k, v in history.history.items()},
            "best_threshold": float(self.best_threshold),
            "best_f1_valid": float(best_f1),
            "best_precision_valid": float(best_precision),
            "best_recall_valid": float(best_recall),
            "pair_metrics_valid": pair_metrics,
            "ranking_metrics_valid": ranking_metrics,
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        }

    def predict_pairs(self, pares: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de predecir.")
        pares = self._ensure_aux_features(pares)
        ds = self._to_ds(pares.assign(label=0), batch_size=batch_size, shuffle=False)
        return self.model.predict(ds, verbose=0).reshape(-1)

    def evaluate_pairs(self, pares: pd.DataFrame, batch_size: int = 512) -> dict:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de evaluar.")
        pares = self._ensure_aux_features(pares)
        ds = self._to_ds(pares, batch_size=batch_size, shuffle=False)
        y_true = pares["label"].astype(int).values
        probs = self.model.predict(ds, verbose=0).reshape(-1)
        best_th, best_f1, best_precision, best_recall = ThresholdOptimizer.find_best(y_true, probs)
        y_pred_current = (probs >= float(self.best_threshold)).astype(int)
        return {
            "n_samples": int(len(pares)),
            "positive_rate": float(np.mean(y_true)),
            "pr_auc": float(average_precision_score(y_true, probs)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "best_threshold_eval": float(best_th),
            "best_f1_eval": float(best_f1),
            "best_precision_eval": float(best_precision),
            "best_recall_eval": float(best_recall),
            "current_threshold": float(self.best_threshold),
            "current_f1": float(f1_score(y_true, y_pred_current, zero_division=0)),
            "current_precision": float(precision_score(y_true, y_pred_current, zero_division=0)),
            "current_recall": float(recall_score(y_true, y_pred_current, zero_division=0)),
        }

    def evaluate_ranking(self, pares: pd.DataFrame, batch_size: int = 512) -> dict:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de evaluar.")
        df = self._ensure_aux_features(pares.copy())
        df["_score"] = self.predict_pairs(df, batch_size=batch_size)

        if "RucProveedor" in df.columns:
            df["_group"] = (
                df["RucProveedor"].astype(str).fillna("")
                + "|"
                + df["fact_cod"].astype(str).fillna("")
                + "|"
                + df["fact_text"].astype(str).fillna("")
            )
        else:
            df["_group"] = (
                df["fact_cod"].astype(str).fillna("")
                + "|"
                + df["fact_text"].astype(str).fillna("")
            )

        total_groups = 0
        hit1 = 0
        hit3 = 0
        hit5 = 0

        for _, g in df.groupby("_group", sort=False):
            g = g.sort_values("_score", ascending=False).reset_index(drop=True)
            pos_ranks = np.where(g["label"].astype(int).values == 1)[0]
            if len(pos_ranks) == 0:
                continue
            total_groups += 1
            best_pos_rank = int(pos_ranks.min()) + 1
            hit1 += int(best_pos_rank <= 1)
            hit3 += int(best_pos_rank <= 3)
            hit5 += int(best_pos_rank <= 5)

        denom = max(total_groups, 1)
        return {
            "n_groups": int(total_groups),
            "hit_at_1": float(hit1 / denom),
            "hit_at_3": float(hit3 / denom),
            "hit_at_5": float(hit5 / denom),
        }

    def _prepared_to_item_inputs(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "text": df["Producto_norm"].astype(str).values,
            "base_text": df["Producto_base_norm"].astype(str).values,
            "unit": df["Unidad_norm"].astype(str).values,
            "type": df["TipoContenido"].astype(str).values,
            "cost": df["Costo_log"].values.astype(np.float32),
            "peso": df["PesoUnitario"].values.astype(np.float32),
            "factor": df["Factor_log"].values.astype(np.float32),
            "content": df["ContenidoUnidad_log"].values.astype(np.float32),
            "total": df["ContenidoTotal_log"].values.astype(np.float32),
        }

    def encode_prepared_items(self, df: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
        if self.item_encoder is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de codificar.")
        inputs = self._prepared_to_item_inputs(df)
        return self.item_encoder.predict(inputs, batch_size=batch_size, verbose=0)