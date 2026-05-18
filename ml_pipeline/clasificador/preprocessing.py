from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import AttributeModelConfigV2
from .schema import AttributeSchemaV2


class PreprocessingAssetsAttrV2:
    def __init__(self, config: AttributeModelConfigV2):
        self.word_vec = tf.keras.layers.TextVectorization(
            max_tokens=config.max_tokens,
            output_mode="int",
            output_sequence_length=config.word_seq_len,
            standardize="lower_and_strip_punctuation",
        )
        self.char_vec = tf.keras.layers.TextVectorization(
            max_tokens=config.max_char_tokens,
            output_mode="int",
            output_sequence_length=config.char_seq_len,
            standardize="lower_and_strip_punctuation",
            split="character",
        )
        self.provider_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)
        self.unit_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)
        self.type_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)
        self.brand_hint_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)
        self.category_hint_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)

        self.target_brand_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)
        self.target_category_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)

        self.numeric_normalizers: Dict[str, tf.keras.layers.Normalization] = {
            base: tf.keras.layers.Normalization(axis=-1, name=f"{base}_normalizer")
            for base in AttributeSchemaV2.NUMERIC_COLUMNS
        }
        self.aux_normalizer = tf.keras.layers.Normalization(axis=-1, name="aux_normalizer")

    def adapt_inputs(self, df: pd.DataFrame) -> None:
        texts = pd.concat([df["text"].astype(str), df["base_text"].astype(str)], ignore_index=True).values
        self.word_vec.adapt(texts)
        self.char_vec.adapt(texts)
        self.provider_lookup.adapt(df["provider"].astype(str).values)
        self.unit_lookup.adapt(df["unit"].astype(str).values)
        self.type_lookup.adapt(df["type"].astype(str).values)
        self.brand_hint_lookup.adapt(df["brand_hint"].astype(str).values)
        self.category_hint_lookup.adapt(df["category_hint"].astype(str).values)
        for c in AttributeSchemaV2.NUMERIC_COLUMNS:
            self.numeric_normalizers[c].adapt(df[c].astype(np.float32).values.reshape(-1, 1))
        self.aux_normalizer.adapt(df[list(AttributeSchemaV2.AUX_COLUMNS)].astype(np.float32).values)

    def adapt_targets(self, df: pd.DataFrame) -> None:
        self.target_brand_lookup.adapt(df["target_marca"].astype(str).values)
        self.target_category_lookup.adapt(df["target_categoria"].astype(str).values)
