from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Model2Config
from .schema import FeatureSchema2


class PreprocessingAssets2:
    def __init__(self, config: Model2Config):
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
        self.unit_lookup = tf.keras.layers.StringLookup(
            output_mode="int",
            mask_token=None,
        )
        self.type_lookup = tf.keras.layers.StringLookup(
            output_mode="int",
            mask_token=None,
        )
        self.normalizers: Dict[str, tf.keras.layers.Normalization] = {
            base: tf.keras.layers.Normalization(axis=-1, name=f"{base}_normalizer")
            for base in FeatureSchema2.NUMERIC_BASES
        }

    def adapt_vocabularies(self, pares: pd.DataFrame) -> None:
        textos = pd.concat(
            [
                pares[FeatureSchema2.fact("text")],
                pares[FeatureSchema2.master("text")],
                pares[FeatureSchema2.fact("base_text")],
                pares[FeatureSchema2.master("base_text")],
            ],
            ignore_index=True,
        ).astype(str).values

        unidades = pd.concat(
            [pares[FeatureSchema2.fact("unit")], pares[FeatureSchema2.master("unit")]],
            ignore_index=True,
        ).astype(str).values

        tipos = pd.concat(
            [pares[FeatureSchema2.fact("type")], pares[FeatureSchema2.master("type")]],
            ignore_index=True,
        ).astype(str).values

        self.word_vec.adapt(textos)
        self.char_vec.adapt(textos)
        self.unit_lookup.adapt(unidades)
        self.type_lookup.adapt(tipos)

    def adapt_normalizers(self, train_df: pd.DataFrame) -> None:
        for base in FeatureSchema2.NUMERIC_BASES:
            values = np.concatenate(
                [
                    train_df[FeatureSchema2.fact(base)].astype(np.float32).values,
                    train_df[FeatureSchema2.master(base)].astype(np.float32).values,
                ]
            ).reshape(-1, 1)
            self.normalizers[base].adapt(values)