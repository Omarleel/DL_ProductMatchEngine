from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from .builder import AttributeModelBuilderV2
from .categories import CategoryLexicon
from .config import AttributeModelConfigV2
from .dataset import AttributeDatasetBuilderV2
from .feature_engineering import add_item_aux_features
from .labels import MISSING_BRAND, MISSING_CATEGORY
from .persistence import AttributeModelPersistenceV2
from .preprocessing import PreprocessingAssetsAttrV2


class ModeloClasificadorProductos:
    def __init__(
        self,
        max_tokens: int = 25000,
        max_char_tokens: int = 260,
        word_seq_len: int = 48,
        char_seq_len: int = 140,
        text_embedding_dim: int = 128,
        char_embedding_dim: int = 32,
        unit_embedding_dim: int = 12,
        type_embedding_dim: int = 8,
        hint_embedding_dim: int = 20,
        provider_embedding_dim: int = 20,
        trunk_dim: int = 256,
        dropout_rate: float = 0.20,
        l2_reg: float = 1e-5,
        learning_rate: float = 7e-4,
    ):
        self.config = AttributeModelConfigV2(
            max_tokens=max_tokens,
            max_char_tokens=max_char_tokens,
            word_seq_len=word_seq_len,
            char_seq_len=char_seq_len,
            text_embedding_dim=text_embedding_dim,
            char_embedding_dim=char_embedding_dim,
            unit_embedding_dim=unit_embedding_dim,
            type_embedding_dim=type_embedding_dim,
            hint_embedding_dim=hint_embedding_dim,
            provider_embedding_dim=provider_embedding_dim,
            trunk_dim=trunk_dim,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
        )
        self.assets = PreprocessingAssetsAttrV2(self.config)
        self.model: Optional[tf.keras.Model] = None
        self.category_lexicon: Optional[CategoryLexicon] = None

    def construir(self) -> tf.keras.Model:
        self.model = AttributeModelBuilderV2(self.config, self.assets).build()
        return self.model

    def guardar(self, carpeta_modelo: str) -> None:
        AttributeModelPersistenceV2.save(self, carpeta_modelo)

    @classmethod
    def cargar(cls, carpeta_modelo: str) -> "ModeloClasificadorProductos":
        config = AttributeModelPersistenceV2.read_config(carpeta_modelo)
        instancia = cls(**asdict(config))
        AttributeModelPersistenceV2.load(instancia, carpeta_modelo)
        return instancia

    def _class_weight_map(self, values: pd.Series, missing_label: str) -> dict[str, float]:
        vals = values.fillna(missing_label).astype(str)
        vals = vals[vals != missing_label]
        freq = Counter(vals.tolist())
        if not freq:
            return {}
        max_count = max(freq.values())
        return {k: float(np.sqrt(max_count / v)) for k, v in freq.items()}

    def preparar_dataset(self, df: pd.DataFrame, fit_assets: bool = False) -> pd.DataFrame:
        work = add_item_aux_features(df.copy(), self.category_lexicon)

        if fit_assets:
            self.assets.adapt_inputs(work)
            self.assets.adapt_targets(work)

        work["target_marca_id"] = self.assets.target_brand_lookup(work["target_marca"].astype(str).values).numpy().astype(np.int32)
        work["target_categoria_id"] = self.assets.target_category_lookup(work["target_categoria"].astype(str).values).numpy().astype(np.int32)

        brand_weights = self._class_weight_map(work["target_marca"], MISSING_BRAND)
        category_weights = self._class_weight_map(work["target_categoria"], MISSING_CATEGORY)
        work["class_weight_marca"] = work["target_marca"].map(lambda x: brand_weights.get(str(x), 1.0)).astype(np.float32)
        work["class_weight_categoria"] = work["target_categoria"].map(lambda x: category_weights.get(str(x), 1.0)).astype(np.float32)
        work["sample_weight_base"] = pd.to_numeric(work.get("sample_weight_base", 1.0), errors="coerce").fillna(1.0).astype(np.float32)

        for c in [
            "mask_factor_venta",
            "mask_factor_conversion",
            "mask_peso_unitario",
            "mask_peso_caja",
            "mask_marca",
            "mask_categoria",
        ]:
            work[c] = pd.to_numeric(work.get(c, 0.0), errors="coerce").fillna(0.0).astype(np.float32)

        return work

    def split_train_valid(self, dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        groups = (
            dataset["RucProveedor"].astype(str).fillna("")
            + "|"
            + dataset["fact_cod"].astype(str).fillna("")
            + "|"
            + dataset["text"].astype(str).fillna("")
        )
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, valid_idx = next(gss.split(dataset, groups=groups))
        return dataset.iloc[train_idx].copy().reset_index(drop=True), dataset.iloc[valid_idx].copy().reset_index(drop=True)

    def fit(self, dataset: pd.DataFrame, category_lexicon: CategoryLexicon, epochs: int = 20, batch_size: int = 256) -> dict:
        self.category_lexicon = category_lexicon
        train_df, valid_df = self.split_train_valid(dataset)
        train_df = self.preparar_dataset(train_df, fit_assets=True)
        valid_df = self.preparar_dataset(valid_df, fit_assets=False)
        self.construir()

        ds_train = AttributeDatasetBuilderV2.to_dataset(train_df, batch_size=batch_size, shuffle=True)
        ds_valid = AttributeDatasetBuilderV2.to_dataset(valid_df, batch_size=batch_size, shuffle=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ]

        history = self.model.fit(ds_train, validation_data=ds_valid, epochs=epochs, callbacks=callbacks, verbose=1)
        metrics_valid = self.evaluate(valid_df, batch_size=batch_size)
        return {
            "history": {k: [float(x) for x in v] for k, v in history.history.items()},
            "metrics_valid": metrics_valid,
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "train_rows_historial": int((train_df["source_type"] == "historial").sum()),
            "train_rows_maestro": int((train_df["source_type"] == "maestro").sum()),
        }

    def predict_raw(self, df: pd.DataFrame, batch_size: int = 512) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de predecir.")
        work = add_item_aux_features(df.copy(), self.category_lexicon)
        x = AttributeDatasetBuilderV2._build_x(work)
        ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        preds = self.model.predict(ds, verbose=0)
        return {k: np.asarray(v) for k, v in preds.items()}

    def _vocab_value(self, vocab: list[str], idx: int) -> str:
        return str(vocab[idx]) if 0 <= idx < len(vocab) else ""

    def predict(self, df: pd.DataFrame, batch_size: int = 512) -> pd.DataFrame:
        raw = self.predict_raw(df, batch_size=batch_size)
        brand_vocab = self.assets.target_brand_lookup.get_vocabulary()
        cat_vocab = self.assets.target_category_lookup.get_vocabulary()
        out = df.copy().reset_index(drop=True)

        pred_factor_venta = np.expm1(raw["factor_venta_output"].reshape(-1)).clip(min=0.0)
        pred_factor_conversion = np.expm1(raw["factor_conversion_output"].reshape(-1)).clip(min=0.0)
        pred_peso_unitario = np.expm1(raw["peso_unitario_output"].reshape(-1)).clip(min=0.0)
        pred_peso_caja = (pred_factor_venta * pred_peso_unitario).clip(min=0.0)

        out["pred_factorVenta"] = pred_factor_venta
        out["pred_factorConversion"] = pred_factor_conversion
        out["pred_pesoUnitarioKg"] = pred_peso_unitario
        out["pred_pesoCajaKg"] = pred_peso_caja

        brand_probs = raw["marca_output"]
        cat_probs = raw["categoria_output"]
        brand_idx = brand_probs.argmax(axis=1)
        cat_idx = cat_probs.argmax(axis=1)
        out["pred_marca"] = [self._vocab_value(brand_vocab, int(i)) for i in brand_idx]
        out["pred_categoria"] = [self._vocab_value(cat_vocab, int(i)) for i in cat_idx]
        out["conf_marca"] = brand_probs.max(axis=1)
        out["conf_categoria"] = cat_probs.max(axis=1)
        return out

    def evaluate(self, df: pd.DataFrame, batch_size: int = 512) -> dict:
        pred_df = self.predict(df, batch_size=batch_size)
        metrics: dict[str, float] = {}

        for y_true_col, y_pred_col, mask_col in [
            ("target_factor_venta", "pred_factorVenta", "mask_factor_venta"),
            ("target_factor_conversion", "pred_factorConversion", "mask_factor_conversion"),
            ("target_peso_unitario_kg", "pred_pesoUnitarioKg", "mask_peso_unitario"),
            ("target_peso_caja_kg", "pred_pesoCajaKg", "mask_peso_caja"),
        ]:
            mask = pred_df[mask_col].astype(float).values > 0
            if mask.sum() == 0:
                continue
            y_true = pd.to_numeric(pred_df.loc[mask, y_true_col], errors="coerce").fillna(0.0).values
            y_pred = pd.to_numeric(pred_df.loc[mask, y_pred_col], errors="coerce").fillna(0.0).values
            metrics[f"{y_true_col}_mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics[f"{y_true_col}_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        for y_true_col, y_pred_col, missing, mask_col in [
            ("target_marca", "pred_marca", MISSING_BRAND, "mask_marca"),
            ("target_categoria", "pred_categoria", MISSING_CATEGORY, "mask_categoria"),
        ]:
            mask = pred_df[mask_col].astype(float).values > 0
            if mask.sum() == 0:
                continue
            y_true = pred_df.loc[mask, y_true_col].astype(str).values
            y_pred = pred_df.loc[mask, y_pred_col].astype(str).values
            metrics[f"{y_true_col}_acc"] = float(accuracy_score(y_true, y_pred))
            metrics[f"{y_true_col}_f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        return metrics
