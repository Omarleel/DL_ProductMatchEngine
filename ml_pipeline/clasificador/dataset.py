from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from .schema import AttributeSchemaV2


class AttributeDatasetBuilderV2:
    @staticmethod
    def _build_x(df: pd.DataFrame) -> dict:
        return {
            "text": df["text"].astype(str).values.reshape(-1, 1),
            "base_text": df["base_text"].astype(str).values.reshape(-1, 1),
            "provider": df["provider"].astype(str).values.reshape(-1, 1),
            "unit": df["unit"].astype(str).values.reshape(-1, 1),
            "type": df["type"].astype(str).values.reshape(-1, 1),
            "brand_hint": df["brand_hint"].astype(str).values.reshape(-1, 1),
            "category_hint": df["category_hint"].astype(str).values.reshape(-1, 1),
            "cost": df["cost"].astype(np.float32).values.reshape(-1, 1),
            "factor": df["factor"].astype(np.float32).values.reshape(-1, 1),
            "content": df["content"].astype(np.float32).values.reshape(-1, 1),
            "total": df["total"].astype(np.float32).values.reshape(-1, 1),
            "peso": df["peso"].astype(np.float32).values.reshape(-1, 1),
            "aux_num": df[list(AttributeSchemaV2.AUX_COLUMNS)].astype(np.float32).values,
        }

    @staticmethod
    def _build_y(df: pd.DataFrame) -> dict:
        return {
            "factor_venta_output": df["target_factor_venta_log"].fillna(0.0).astype(np.float32).values,
            "factor_conversion_output": df["target_factor_conversion_log"].fillna(0.0).astype(np.float32).values,
            "peso_unitario_output": df["target_peso_unitario_kg_log"].fillna(0.0).astype(np.float32).values,
            "peso_caja_output": df["target_peso_caja_kg_log"].fillna(0.0).astype(np.float32).values,
            "marca_output": df["target_marca_id"].astype(np.int32).values,
            "categoria_output": df["target_categoria_id"].astype(np.int32).values,
        }

    @staticmethod
    def _build_sample_weights(df: pd.DataFrame) -> dict:
        base = df["sample_weight_base"].astype(np.float32).fillna(1.0).values
        return {
            "factor_venta_output": base * df["mask_factor_venta"].astype(np.float32).values,
            "factor_conversion_output": base * df["mask_factor_conversion"].astype(np.float32).values,
            "peso_unitario_output": base * df["mask_peso_unitario"].astype(np.float32).values,
            "peso_caja_output": base * df["mask_peso_caja"].astype(np.float32).values,
            "marca_output": base * df["class_weight_marca"].astype(np.float32).values * df["mask_marca"].astype(np.float32).values,
            "categoria_output": base * df["class_weight_categoria"].astype(np.float32).values * df["mask_categoria"].astype(np.float32).values,
        }

    @classmethod
    def to_dataset(cls, df: pd.DataFrame, batch_size: int = 256, shuffle: bool = False) -> tf.data.Dataset:
        x = cls._build_x(df)
        y = cls._build_y(df)
        sample_weights = cls._build_sample_weights(df)
        ds = tf.data.Dataset.from_tensor_slices((x, y, sample_weights))
        if shuffle:
            ds = ds.shuffle(min(len(df), 10000), reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
