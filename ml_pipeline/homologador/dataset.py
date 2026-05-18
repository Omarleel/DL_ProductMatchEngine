from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from .feature_engineering import AUX_FEATURE_COLUMNS, add_aux_pair_features


class DatasetBuilder2:
    @staticmethod
    def to_dataset(
        df: pd.DataFrame,
        sample_weight=None,
        batch_size: int = 256,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        work_df = df.copy()

        missing_aux = [c for c in AUX_FEATURE_COLUMNS if c not in work_df.columns]
        if missing_aux:
            work_df = add_aux_pair_features(work_df)

        x = {
            "fact_text": work_df["fact_text"].astype(str).values,
            "fact_base_text": work_df["fact_base_text"].astype(str).values,
            "fact_unit": work_df["fact_unit"].astype(str).values,
            "fact_type": work_df["fact_type"].astype(str).values,
            "fact_cost": work_df["fact_cost"].astype(np.float32).values,
            "fact_peso": work_df["fact_peso"].astype(np.float32).values,
            "fact_factor": work_df["fact_factor"].astype(np.float32).values,
            "fact_content": work_df["fact_content"].astype(np.float32).values,
            "fact_total": work_df["fact_total"].astype(np.float32).values,

            "master_text": work_df["master_text"].astype(str).values,
            "master_base_text": work_df["master_base_text"].astype(str).values,
            "master_unit": work_df["master_unit"].astype(str).values,
            "master_type": work_df["master_type"].astype(str).values,
            "master_cost": work_df["master_cost"].astype(np.float32).values,
            "master_peso": work_df["master_peso"].astype(np.float32).values,
            "master_factor": work_df["master_factor"].astype(np.float32).values,
            "master_content": work_df["master_content"].astype(np.float32).values,
            "master_total": work_df["master_total"].astype(np.float32).values,

            "aux_num": work_df[AUX_FEATURE_COLUMNS].astype(np.float32).values,
        }

        y = work_df["label"].astype(np.float32).values

        if sample_weight is None:
            ds = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            ds = tf.data.Dataset.from_tensor_slices((x, y, sw))

        if shuffle:
            ds = ds.shuffle(min(len(work_df), 10000), reshuffle_each_iteration=True)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds