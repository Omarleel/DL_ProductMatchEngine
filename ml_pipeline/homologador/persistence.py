from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf

from .config import Model2Config

PathLike = Union[str, Path]
JsonPayload = Union[dict, list]


class HomologatorModelPersistence:
    @staticmethod
    def save(instance: "ModeloHomologadorProductos", carpeta_modelo: PathLike) -> None:
        path = Path(carpeta_modelo).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        log_errors = []   # almacenaremos errores aquí

        # 1. Pesos del modelo
        try:
            if instance.model is None:
                raise RuntimeError("No hay modelo para guardar.")
            #instance.model.save_weights(path / "pair_model.weights.h5")
            instance.model.save_weights(path / "pair_model.weights", save_format='tf')
        except Exception as e:
            log_errors.append(f"Error guardando pesos: {e}")
            # Si falla aquí, no tiene sentido continuar
            with open(path / "save_errors.log", "w", encoding="utf-8") as f:
                f.write("\n".join(log_errors))
            raise

        # 2. Vocabularios
        try:
            HomologatorModelPersistence._write_json(path / "word_vocabulary.json", instance.assets.word_vec.get_vocabulary())
            HomologatorModelPersistence._write_json(path / "char_vocabulary.json", instance.assets.char_vec.get_vocabulary())
            HomologatorModelPersistence._write_json(path / "unit_vocabulary.json", instance.assets.unit_lookup.get_vocabulary())
            HomologatorModelPersistence._write_json(path / "type_vocabulary.json", instance.assets.type_lookup.get_vocabulary())
        except Exception as e:
            log_errors.append(f"Error guardando vocabularios: {e}")

        # 3. Normalizadores
        normalizer_dir = path / "normalizers"
        try:
            normalizer_dir.mkdir(exist_ok=True)
            for base, normalizer in instance.assets.normalizers.items():
                weights = normalizer.get_weights()
                np.savez(
                    normalizer_dir / f"{base}.npz",
                    **{f"arr_{i}": np.asarray(w) for i, w in enumerate(weights)},
                )
        except Exception as e:
            log_errors.append(f"Error guardando normalizadores: {e}")

        # 4. Meta
        try:
            meta = asdict(instance.config) | {"best_threshold": instance.best_threshold}
            HomologatorModelPersistence._write_json(path / "meta.json", meta)
        except Exception as e:
            log_errors.append(f"Error guardando meta: {e}")

        # Si hubo errores, los registramos
        if log_errors:
            with open(path / "save_errors.log", "w", encoding="utf-8") as f:
                f.write("\n".join(log_errors))

    @staticmethod
    def load(instance: "ModeloHomologadorProductos", carpeta_modelo: PathLike) -> None:
        path = Path(carpeta_modelo).expanduser()

        HomologatorModelPersistence._require_file(path / "meta.json")
        HomologatorModelPersistence._require_file(path / "pair_model.weights.index")
        #HomologatorModelPersistence._require_file(path / "pair_model.weights.h5")
        HomologatorModelPersistence._require_file(path / "word_vocabulary.json")
        HomologatorModelPersistence._require_file(path / "char_vocabulary.json")
        HomologatorModelPersistence._require_file(path / "unit_vocabulary.json")
        HomologatorModelPersistence._require_file(path / "type_vocabulary.json")

        instance.assets.word_vec.set_vocabulary(HomologatorModelPersistence._read_json(path / "word_vocabulary.json"))
        instance.assets.char_vec.set_vocabulary(HomologatorModelPersistence._read_json(path / "char_vocabulary.json"))
        HomologatorModelPersistence._safe_set_lookup_vocabulary(
            instance.assets.unit_lookup,
            HomologatorModelPersistence._read_json(path / "unit_vocabulary.json"),
        )
        HomologatorModelPersistence._safe_set_lookup_vocabulary(
            instance.assets.type_lookup,
            HomologatorModelPersistence._read_json(path / "type_vocabulary.json"),
        )

        instance.construir()

        normalizer_dir = path / "normalizers"
        for base, normalizer in instance.assets.normalizers.items():
            npz_path = normalizer_dir / f"{base}.npz"
            HomologatorModelPersistence._require_file(npz_path)
            with np.load(npz_path, allow_pickle=True) as data:
                keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))
                weights = [data[k] for k in keys]
            normalizer.set_weights(weights)

        #instance.model.load_weights(path / "pair_model.weights.h5")
        instance.model.load_weights(path / "pair_model.weights")
        meta = HomologatorModelPersistence._read_json(path / "meta.json")
        instance.best_threshold = float(meta.get("best_threshold", 0.72))

    @staticmethod
    def read_config(carpeta_modelo: PathLike) -> Model2Config:
        meta_path = Path(carpeta_modelo).expanduser() / "meta.json"
        HomologatorModelPersistence._require_file(meta_path)

        meta = HomologatorModelPersistence._read_json(meta_path)
        return Model2Config(
            max_tokens=meta.get("max_tokens", 18000),
            max_char_tokens=meta.get("max_char_tokens", 160),
            word_seq_len=meta.get("word_seq_len", 32),
            char_seq_len=meta.get("char_seq_len", 96),
            text_embedding_dim=meta.get("text_embedding_dim", 96),
            char_embedding_dim=meta.get("char_embedding_dim", 32),
            item_embedding_dim=meta.get("item_embedding_dim", 128),
            unit_embedding_dim=meta.get("unit_embedding_dim", 12),
            type_embedding_dim=meta.get("type_embedding_dim", 8),
            dropout_rate=meta.get("dropout_rate", 0.20),
            l2_reg=meta.get("l2_reg", 1e-5),
            learning_rate=meta.get("learning_rate", 8e-4),
        )

    @staticmethod
    def _safe_set_lookup_vocabulary(
        lookup: tf.keras.layers.StringLookup,
        vocabulary: list[str],
    ) -> None:
        try:
            lookup.set_vocabulary(vocabulary)
        except Exception:
            lookup.set_vocabulary(vocabulary[1:])

    @staticmethod
    def _write_json(path: Path, payload: JsonPayload) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_json(path: Path) -> JsonPayload:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo requerido: '{path}'")