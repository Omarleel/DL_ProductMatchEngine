from __future__ import annotations

from typing import Dict

import tensorflow as tf

from .config import AttributeModelConfigV2
from .preprocessing import PreprocessingAssetsAttrV2


class AttributeModelBuilderV2:
    def __init__(self, config: AttributeModelConfigV2, assets: PreprocessingAssetsAttrV2):
        self.config = config
        self.assets = assets

    def build(self) -> tf.keras.Model:
        inputs: Dict[str, tf.keras.Input] = {
            "text": tf.keras.Input(shape=(1,), dtype=tf.string, name="text"),
            "base_text": tf.keras.Input(shape=(1,), dtype=tf.string, name="base_text"),
            "provider": tf.keras.Input(shape=(1,), dtype=tf.string, name="provider"),
            "unit": tf.keras.Input(shape=(1,), dtype=tf.string, name="unit"),
            "type": tf.keras.Input(shape=(1,), dtype=tf.string, name="type"),
            "brand_hint": tf.keras.Input(shape=(1,), dtype=tf.string, name="brand_hint"),
            "category_hint": tf.keras.Input(shape=(1,), dtype=tf.string, name="category_hint"),
            "cost": tf.keras.Input(shape=(1,), dtype=tf.float32, name="cost"),
            "factor": tf.keras.Input(shape=(1,), dtype=tf.float32, name="factor"),
            "content": tf.keras.Input(shape=(1,), dtype=tf.float32, name="content"),
            "total": tf.keras.Input(shape=(1,), dtype=tf.float32, name="total"),
            "peso": tf.keras.Input(shape=(1,), dtype=tf.float32, name="peso"),
            "aux_num": tf.keras.Input(shape=(9,), dtype=tf.float32, name="aux_num"),
        }

        text_repr = self._build_text_block(inputs["text"], name="text")
        base_repr = self._build_text_block(inputs["base_text"], name="base_text")
        provider_repr = self._embed_lookup(self.assets.provider_lookup, inputs["provider"], self.config.provider_embedding_dim, "provider")
        unit_repr = self._embed_lookup(self.assets.unit_lookup, inputs["unit"], self.config.unit_embedding_dim, "unit")
        type_repr = self._embed_lookup(self.assets.type_lookup, inputs["type"], self.config.type_embedding_dim, "type")
        brand_hint_repr = self._embed_lookup(self.assets.brand_hint_lookup, inputs["brand_hint"], self.config.hint_embedding_dim, "brand_hint")
        category_hint_repr = self._embed_lookup(self.assets.category_hint_lookup, inputs["category_hint"], self.config.hint_embedding_dim, "category_hint")

        num_blocks = [self.assets.numeric_normalizers[c](inputs[c]) for c in ["cost", "factor", "content", "total", "peso"]]
        num_repr = tf.keras.layers.Concatenate(name="num_concat")(num_blocks)
        num_repr = self._small_mlp(num_repr, 48, name="num")

        aux_repr = self.assets.aux_normalizer(inputs["aux_num"])
        aux_repr = self._small_mlp(aux_repr, 48, name="aux")

        trunk = tf.keras.layers.Concatenate(name="trunk_concat")([
            text_repr,
            base_repr,
            provider_repr,
            unit_repr,
            type_repr,
            brand_hint_repr,
            category_hint_repr,
            num_repr,
            aux_repr,
        ])
        trunk = tf.keras.layers.BatchNormalization(name="trunk_bn")(trunk)
        trunk = tf.keras.layers.Dense(
            self.config.trunk_dim,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="trunk_dense_1",
        )(trunk)
        trunk = tf.keras.layers.Dropout(self.config.dropout_rate, name="trunk_dropout_1")(trunk)
        trunk = tf.keras.layers.Dense(
            self.config.trunk_dim // 2,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="trunk_dense_2",
        )(trunk)
        trunk = tf.keras.layers.Dropout(self.config.dropout_rate * 0.75, name="trunk_dropout_2")(trunk)

        factor_venta_output = self._regression_head(trunk, "factor_venta_output")
        factor_conversion_output = self._regression_head(trunk, "factor_conversion_output")
        peso_unitario_output = self._regression_head(trunk, "peso_unitario_output")
        peso_caja_output = tf.keras.layers.Lambda(
            lambda xs: tf.math.log1p(tf.math.expm1(xs[0]) * tf.math.expm1(xs[1])),
            name="peso_caja_output",
        )([factor_venta_output, peso_unitario_output])

        marca_output = self._classification_head(trunk, self.assets.target_brand_lookup.vocabulary_size(), "marca_output")
        categoria_output = self._classification_head(trunk, self.assets.target_category_lookup.vocabulary_size(), "categoria_output")

        model = tf.keras.Model(
            inputs=list(inputs.values()),
            outputs={
                "factor_venta_output": factor_venta_output,
                "factor_conversion_output": factor_conversion_output,
                "peso_unitario_output": peso_unitario_output,
                "peso_caja_output": peso_caja_output,
                "marca_output": marca_output,
                "categoria_output": categoria_output,
            },
            name="modelo_multitask_atributos_producto_v2",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss={
                "factor_venta_output": tf.keras.losses.Huber(delta=0.25),
                "factor_conversion_output": tf.keras.losses.Huber(delta=0.25),
                "peso_unitario_output": tf.keras.losses.Huber(delta=0.20),
                "peso_caja_output": tf.keras.losses.Huber(delta=0.20),
                "marca_output": tf.keras.losses.SparseCategoricalCrossentropy(),
                "categoria_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            loss_weights={
                "factor_venta_output": 1.0,
                "factor_conversion_output": 1.0,
                "peso_unitario_output": 1.0,
                "peso_caja_output": 0.75,
                "marca_output": 1.25,
                "categoria_output": 1.40,
            },
            metrics={
                "factor_venta_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
                "factor_conversion_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
                "peso_unitario_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
                "peso_caja_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
                "marca_output": [
                    tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
                ],
                "categoria_output": [
                    tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
                ],
            },
        )
        return model

    def _build_text_block(self, inp: tf.keras.Input, name: str) -> tf.Tensor:
        word_ids = self.assets.word_vec(inp)
        word_emb = tf.keras.layers.Embedding(
            input_dim=self.assets.word_vec.vocabulary_size(),
            output_dim=self.config.text_embedding_dim,
            name=f"{name}_word_emb",
        )(word_ids)
        word_conv_3 = tf.keras.layers.SeparableConv1D(96, 3, padding="same", activation="gelu", name=f"{name}_word_conv_3")(word_emb)
        word_conv_5 = tf.keras.layers.SeparableConv1D(96, 5, padding="same", activation="gelu", name=f"{name}_word_conv_5")(word_emb)
        word_pool = tf.keras.layers.Concatenate(name=f"{name}_word_pool_concat")([
            tf.keras.layers.GlobalMaxPooling1D(name=f"{name}_word_gmp")(word_conv_3),
            tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_word_gap")(word_conv_5),
        ])

        char_ids = self.assets.char_vec(inp)
        char_emb = tf.keras.layers.Embedding(
            input_dim=self.assets.char_vec.vocabulary_size(),
            output_dim=self.config.char_embedding_dim,
            name=f"{name}_char_emb",
        )(char_ids)
        char_conv = tf.keras.layers.SeparableConv1D(64, 5, padding="same", activation="gelu", name=f"{name}_char_conv")(char_emb)
        char_pool = tf.keras.layers.Concatenate(name=f"{name}_char_pool_concat")([
            tf.keras.layers.GlobalMaxPooling1D(name=f"{name}_char_gmp")(char_conv),
            tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_char_gap")(char_conv),
        ])

        x = tf.keras.layers.Concatenate(name=f"{name}_concat")([word_pool, char_pool])
        return self._small_mlp(x, 96, name=name)

    def _embed_lookup(self, lookup, inp, emb_dim: int, name: str) -> tf.Tensor:
        ids = lookup(inp)
        emb = tf.keras.layers.Embedding(
            input_dim=lookup.vocabulary_size(),
            output_dim=emb_dim,
            name=f"{name}_embedding",
        )(ids)
        return tf.keras.layers.Flatten(name=f"{name}_flatten")(emb)

    def _small_mlp(self, x: tf.Tensor, units: int, name: str) -> tf.Tensor:
        x = tf.keras.layers.Dense(
            units,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_dense",
        )(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate * 0.5, name=f"{name}_dropout")(x)
        return x

    def _regression_head(self, trunk: tf.Tensor, name: str) -> tf.Tensor:
        x = tf.keras.layers.Dense(64, activation="gelu", name=f"{name}_dense_1")(trunk)
        x = tf.keras.layers.Dropout(self.config.dropout_rate * 0.5, name=f"{name}_dropout_1")(x)
        return tf.keras.layers.Dense(1, activation="softplus", name=name)(x)

    def _classification_head(self, trunk: tf.Tensor, n_classes: int, name: str) -> tf.Tensor:
        x = tf.keras.layers.Dense(96, activation="gelu", name=f"{name}_dense_1")(trunk)
        x = tf.keras.layers.Dropout(self.config.dropout_rate * 0.5, name=f"{name}_dropout_1")(x)
        return tf.keras.layers.Dense(n_classes, activation="softmax", name=name)(x)
