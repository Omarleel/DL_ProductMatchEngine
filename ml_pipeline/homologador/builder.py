from __future__ import annotations

from typing import Dict

import tensorflow as tf

from .config import Model2Config
from .preprocessing import PreprocessingAssets2
from .schema import FeatureSchema2
from .feature_engineering import AUX_FEATURE_COLUMNS

class MatchModelBuilder2:
    def __init__(self, config: Model2Config, assets: PreprocessingAssets2):
        self.config = config
        self.assets = assets
        self.item_encoder: tf.keras.Model | None = None

    def build(self) -> tuple[tf.keras.Model, tf.keras.Model]:
        item_encoder = self._build_item_encoder()
        self.item_encoder = item_encoder

        pair_inputs = self._build_pair_inputs()

        fact_inputs = {
            base: pair_inputs[FeatureSchema2.fact(base)]
            for base in FeatureSchema2.TEXT_BASES + FeatureSchema2.CATEGORICAL_BASES + FeatureSchema2.NUMERIC_BASES
        }
        master_inputs = {
            base: pair_inputs[FeatureSchema2.master(base)]
            for base in FeatureSchema2.TEXT_BASES + FeatureSchema2.CATEGORICAL_BASES + FeatureSchema2.NUMERIC_BASES
        }

        fact_emb = item_encoder(fact_inputs)
        master_emb = item_encoder(master_inputs)

        fact_num = self._normalized_numeric_block(fact_inputs, prefix="fact")
        master_num = self._normalized_numeric_block(master_inputs, prefix="master")

        num_abs_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="num_abs_diff",
        )([fact_num, master_num])

        num_mul = tf.keras.layers.Multiply(name="num_mul")([fact_num, master_num])

        unit_match = self._equal_match(
            self.assets.unit_lookup(pair_inputs["fact_unit"]),
            self.assets.unit_lookup(pair_inputs["master_unit"]),
            "unit_match",
        )
        type_match = self._equal_match(
            self.assets.type_lookup(pair_inputs["fact_type"]),
            self.assets.type_lookup(pair_inputs["master_type"]),
            "type_match",
        )

        emb_abs_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="emb_abs_diff",
        )([fact_emb, master_emb])

        emb_mul = tf.keras.layers.Multiply(name="emb_mul")([fact_emb, master_emb])

        cosine = tf.keras.layers.Dot(axes=1, normalize=True, name="embedding_cosine")(
            [fact_emb, master_emb]
        )

        # NUEVO: bloque para features auxiliares de ranking / marca / estructura
        aux_num = pair_inputs["aux_num"]
        aux = tf.keras.layers.BatchNormalization(name="aux_bn")(aux_num)
        aux = tf.keras.layers.Dense(
            32,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="aux_dense_1",
        )(aux)
        aux = tf.keras.layers.Dropout(self.config.dropout_rate * 0.5, name="aux_dropout_1")(aux)
        aux = tf.keras.layers.Dense(
            16,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="aux_dense_2",
        )(aux)

        pair_features = tf.keras.layers.Concatenate(name="pair_features")([
            fact_emb,
            master_emb,
            emb_abs_diff,
            emb_mul,
            cosine,
            fact_num,
            master_num,
            num_abs_diff,
            num_mul,
            unit_match,
            type_match,
            aux,  # NUEVO
        ])

        x = tf.keras.layers.BatchNormalization(name="pair_bn")(pair_features)
        x = tf.keras.layers.Dense(
            256,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="pair_dense_1",
        )(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate + 0.05, name="pair_dropout_1")(x)
        x = tf.keras.layers.Dense(
            128,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="pair_dense_2",
        )(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate, name="pair_dropout_2")(x)
        x = tf.keras.layers.Dense(
            64,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="pair_dense_3",
        )(x)
        y = tf.keras.layers.Dense(1, activation="sigmoid", name="match_prob")(x)

        pair_model = tf.keras.Model(
            inputs=list(pair_inputs.values()),
            outputs=y,
            name="homologador_pair_ranker",
        )
        pair_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.005),
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
            ],
        )
        return pair_model, item_encoder

    def _build_pair_inputs(self) -> Dict[str, tf.keras.Input]:
        inputs: Dict[str, tf.keras.Input] = {}

        for base in FeatureSchema2.TEXT_BASES + FeatureSchema2.CATEGORICAL_BASES:
            inputs[FeatureSchema2.fact(base)] = tf.keras.Input(
                shape=(1,),
                dtype=tf.string,
                name=FeatureSchema2.fact(base),
            )
            inputs[FeatureSchema2.master(base)] = tf.keras.Input(
                shape=(1,),
                dtype=tf.string,
                name=FeatureSchema2.master(base),
            )

        for base in FeatureSchema2.NUMERIC_BASES:
            inputs[FeatureSchema2.fact(base)] = tf.keras.Input(
                shape=(1,),
                dtype=tf.float32,
                name=FeatureSchema2.fact(base),
            )
            inputs[FeatureSchema2.master(base)] = tf.keras.Input(
                shape=(1,),
                dtype=tf.float32,
                name=FeatureSchema2.master(base),
            )

        # NUEVO: vector auxiliar numérico
        inputs["aux_num"] = tf.keras.Input(
            shape=(len(AUX_FEATURE_COLUMNS),),
            dtype=tf.float32,
            name="aux_num",
        )

        return inputs

    def _build_item_encoder(self) -> tf.keras.Model:
        inputs: Dict[str, tf.keras.Input] = {
            "text": tf.keras.Input(shape=(1,), dtype=tf.string, name="text"),
            "base_text": tf.keras.Input(shape=(1,), dtype=tf.string, name="base_text"),
            "unit": tf.keras.Input(shape=(1,), dtype=tf.string, name="unit"),
            "type": tf.keras.Input(shape=(1,), dtype=tf.string, name="type"),
            "cost": tf.keras.Input(shape=(1,), dtype=tf.float32, name="cost"),
            "peso": tf.keras.Input(shape=(1,), dtype=tf.float32, name="peso"),
            "factor": tf.keras.Input(shape=(1,), dtype=tf.float32, name="factor"),
            "content": tf.keras.Input(shape=(1,), dtype=tf.float32, name="content"),
            "total": tf.keras.Input(shape=(1,), dtype=tf.float32, name="total"),
        }

        word_encoder = self._build_text_encoder(kind="word")
        char_encoder = self._build_text_encoder(kind="char")
        unit_encoder = self._build_categorical_encoder(
            lookup=self.assets.unit_lookup,
            emb_dim=self.config.unit_embedding_dim,
            name="unit",
        )
        type_encoder = self._build_categorical_encoder(
            lookup=self.assets.type_lookup,
            emb_dim=self.config.type_embedding_dim,
            name="type",
        )

        text_word = word_encoder(inputs["text"])
        text_char = char_encoder(inputs["text"])
        base_word = word_encoder(inputs["base_text"])
        base_char = char_encoder(inputs["base_text"])

        text_repr = tf.keras.layers.Concatenate(name="text_repr")([text_word, text_char])
        text_repr = self._small_mlp(text_repr, 96, "text_repr")

        base_repr = tf.keras.layers.Concatenate(name="base_repr")([base_word, base_char])
        base_repr = self._small_mlp(base_repr, 64, "base_repr")

        unit_repr = unit_encoder(inputs["unit"])
        type_repr = type_encoder(inputs["type"])

        num_repr = self._normalized_numeric_block(inputs, prefix="item")
        num_repr = self._small_mlp(num_repr, 48, "num_repr")

        x = tf.keras.layers.Concatenate(name="item_features")([
            text_repr,
            base_repr,
            unit_repr,
            type_repr,
            num_repr,
        ])
        x = tf.keras.layers.BatchNormalization(name="item_bn")(x)
        x = tf.keras.layers.Dense(
            256,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="item_dense_1",
        )(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate, name="item_dropout_1")(x)
        x = tf.keras.layers.Dense(
            self.config.item_embedding_dim,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="item_dense_2",
        )(x)

        emb = tf.keras.layers.Lambda(
            lambda z: tf.math.l2_normalize(z, axis=1),
            name="item_embedding",
        )(x)

        return tf.keras.Model(inputs=inputs, outputs=emb, name="homologador_item_encoder")

    def _normalized_numeric_block(self, inputs: Dict[str, tf.Tensor], prefix: str) -> tf.Tensor:
        cols = []
        for base in FeatureSchema2.NUMERIC_BASES:
            cols.append(self.assets.normalizers[base](inputs[base]))
            cols.append(
                tf.keras.layers.Lambda(
                    lambda x: tf.cast(x > 0.0, tf.float32),
                    name=f"{prefix}_{base}_available",
                )(inputs[base])
            )
        return tf.keras.layers.Concatenate(name=f"{prefix}_numeric_features")(cols)

    def _build_text_encoder(self, kind: str) -> tf.keras.Model:
        if kind == "word":
            vectorizer = self.assets.word_vec
            vocab_size = max(int(vectorizer.vocabulary_size()), 2)
            emb_dim = self.config.text_embedding_dim
            seq_len = self.config.word_seq_len
            name = "word"
        elif kind == "char":
            vectorizer = self.assets.char_vec
            vocab_size = max(int(vectorizer.vocabulary_size()), 2)
            emb_dim = self.config.char_embedding_dim
            seq_len = self.config.char_seq_len
            name = "char"
        else:
            raise ValueError(f"Tipo de encoder desconocido: {kind}")

        inp = tf.keras.Input(shape=(1,), dtype=tf.string, name=f"{name}_text_input")
        seq = vectorizer(inp)
        x = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=emb_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_embedding",
        )(seq)
        x = tf.keras.layers.SpatialDropout1D(self.config.dropout_rate * 0.5, name=f"{name}_spatial_dropout")(x)

        conv3 = tf.keras.layers.SeparableConv1D(
            filters=emb_dim,
            kernel_size=3,
            padding="same",
            activation="gelu",
            depthwise_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            pointwise_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_conv3",
        )(x)
        conv5 = tf.keras.layers.SeparableConv1D(
            filters=emb_dim,
            kernel_size=5,
            padding="same",
            activation="gelu",
            depthwise_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            pointwise_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_conv5",
        )(x)

        max_pool = tf.keras.layers.GlobalMaxPooling1D(name=f"{name}_gmp")(conv3)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_gap")(conv5)
        length_hint = tf.keras.layers.Lambda(
            lambda s: tf.cast(
                tf.reduce_sum(tf.cast(tf.not_equal(s, 0), tf.float32), axis=1, keepdims=True) / float(seq_len),
                tf.float32,
            ),
            name=f"{name}_length_hint",
        )(seq)

        out = tf.keras.layers.Concatenate(name=f"{name}_text_features")([max_pool, avg_pool, length_hint])
        out = tf.keras.layers.Dense(
            emb_dim,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_text_dense",
        )(out)
        return tf.keras.Model(inp, out, name=f"{name}_text_encoder")

    def _build_categorical_encoder(
        self,
        lookup: tf.keras.layers.StringLookup,
        emb_dim: int,
        name: str,
    ) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(1,), dtype=tf.string, name=f"{name}_input")
        idx = lookup(inp)
        x = tf.keras.layers.Embedding(
            input_dim=max(int(lookup.vocabulary_size()), 2),
            output_dim=emb_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_embedding",
        )(idx)
        x = tf.keras.layers.Flatten(name=f"{name}_flatten")(x)
        x = tf.keras.layers.Dense(
            emb_dim,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_dense",
        )(x)
        return tf.keras.Model(inp, x, name=f"{name}_encoder")

    def _small_mlp(self, x: tf.Tensor, units: int, prefix: str) -> tf.Tensor:
        x = tf.keras.layers.Dense(
            units,
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{prefix}_dense_1",
        )(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate * 0.5, name=f"{prefix}_dropout")(x)
        x = tf.keras.layers.Dense(
            max(units // 2, 16),
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{prefix}_dense_2",
        )(x)
        return x

    @staticmethod
    def _equal_match(a: tf.Tensor, b: tf.Tensor, name: str) -> tf.Tensor:
        return tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32),
            output_shape=(1,),
            name=name,
        )([a, b])