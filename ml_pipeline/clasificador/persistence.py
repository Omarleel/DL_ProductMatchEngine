from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np

from .config import AttributeModelConfigV2

PathLike = Union[str, Path]


class AttributeModelPersistenceV2:
    @staticmethod
    def save(instance: "ModeloClasificadorProductos", carpeta_modelo: PathLike) -> None:
        if instance.model is None:
            raise RuntimeError("No hay modelo para guardar.")
        path = Path(carpeta_modelo).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        instance.model.save_weights(path / "attribute_model_v2.weights.h5")
        AttributeModelPersistenceV2._write_json(path / "word_vocabulary.json", instance.assets.word_vec.get_vocabulary())
        AttributeModelPersistenceV2._write_json(path / "char_vocabulary.json", instance.assets.char_vec.get_vocabulary())
        AttributeModelPersistenceV2._write_json(path / "provider_vocabulary.json", instance.assets.provider_lookup.get_vocabulary())
        AttributeModelPersistenceV2._write_json(path / "unit_vocabulary.json", instance.assets.unit_lookup.get_vocabulary())
        AttributeModelPersistenceV2._write_json(path / "type_vocabulary.json", instance.assets.type_lookup.get_vocabulary())
        AttributeModelPersistenceV2._write_json(path / "brand_hint_vocabulary.json", instance.assets.brand_hint_lookup.get_vocabulary())
        AttributeModelPersistenceV2._write_json(path / "category_hint_vocabulary.json", instance.assets.category_hint_lookup.get_vocabulary())
        AttributeModelPersistenceV2._write_json(path / "target_brand_vocabulary.json", instance.assets.target_brand_lookup.get_vocabulary())
        AttributeModelPersistenceV2._write_json(path / "target_category_vocabulary.json", instance.assets.target_category_lookup.get_vocabulary())

        normalizer_dir = path / "normalizers"
        normalizer_dir.mkdir(exist_ok=True)
        for base, normalizer in instance.assets.numeric_normalizers.items():
            weights = normalizer.get_weights()
            np.savez(normalizer_dir / f"{base}.npz", **{f"arr_{i}": np.asarray(w) for i, w in enumerate(weights)})

        aux_weights = instance.assets.aux_normalizer.get_weights()
        np.savez(normalizer_dir / "aux_num.npz", **{f"arr_{i}": np.asarray(w) for i, w in enumerate(aux_weights)})

        if instance.category_lexicon is not None:
            instance.category_lexicon.save(path / "category_lexicon.json")

        meta = asdict(instance.config)
        meta["has_category_lexicon"] = instance.category_lexicon is not None
        AttributeModelPersistenceV2._write_json(path / "meta.json", meta)

    @staticmethod
    def load(instance: "ModeloClasificadorProductos", carpeta_modelo: PathLike) -> None:
        path = Path(carpeta_modelo).expanduser()
        AttributeModelPersistenceV2._require_file(path / "meta.json")

        instance.assets.word_vec.set_vocabulary(AttributeModelPersistenceV2._read_json(path / "word_vocabulary.json"))
        instance.assets.char_vec.set_vocabulary(AttributeModelPersistenceV2._read_json(path / "char_vocabulary.json"))
        for layer_name, vocab_file in [
            (instance.assets.provider_lookup, "provider_vocabulary.json"),
            (instance.assets.unit_lookup, "unit_vocabulary.json"),
            (instance.assets.type_lookup, "type_vocabulary.json"),
            (instance.assets.brand_hint_lookup, "brand_hint_vocabulary.json"),
            (instance.assets.category_hint_lookup, "category_hint_vocabulary.json"),
            (instance.assets.target_brand_lookup, "target_brand_vocabulary.json"),
            (instance.assets.target_category_lookup, "target_category_vocabulary.json"),
        ]:
            AttributeModelPersistenceV2._safe_set_lookup_vocabulary(layer_name, AttributeModelPersistenceV2._read_json(path / vocab_file))

        instance.construir()

        normalizer_dir = path / "normalizers"
        for base, normalizer in instance.assets.numeric_normalizers.items():
            npz_path = normalizer_dir / f"{base}.npz"
            AttributeModelPersistenceV2._require_file(npz_path)
            with np.load(npz_path, allow_pickle=True) as data:
                keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))
                normalizer.set_weights([data[k] for k in keys])

        aux_path = normalizer_dir / "aux_num.npz"
        AttributeModelPersistenceV2._require_file(aux_path)
        with np.load(aux_path, allow_pickle=True) as data:
            keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))
            instance.assets.aux_normalizer.set_weights([data[k] for k in keys])

        instance.model.load_weights(path / "attribute_model_v2.weights.h5")

        from .categories import CategoryLexicon
        lexicon_path = path / "category_lexicon.json"
        if lexicon_path.exists():
            instance.category_lexicon = CategoryLexicon.load(lexicon_path)

    @staticmethod
    def read_config(carpeta_modelo: PathLike) -> AttributeModelConfigV2:
        meta = AttributeModelPersistenceV2._read_json(Path(carpeta_modelo).expanduser() / "meta.json")
        return AttributeModelConfigV2(
            max_tokens=meta.get("max_tokens", 25000),
            max_char_tokens=meta.get("max_char_tokens", 260),
            word_seq_len=meta.get("word_seq_len", 48),
            char_seq_len=meta.get("char_seq_len", 140),
            text_embedding_dim=meta.get("text_embedding_dim", 128),
            char_embedding_dim=meta.get("char_embedding_dim", 32),
            unit_embedding_dim=meta.get("unit_embedding_dim", 12),
            type_embedding_dim=meta.get("type_embedding_dim", 8),
            hint_embedding_dim=meta.get("hint_embedding_dim", 20),
            provider_embedding_dim=meta.get("provider_embedding_dim", 20),
            trunk_dim=meta.get("trunk_dim", 256),
            dropout_rate=meta.get("dropout_rate", 0.20),
            l2_reg=meta.get("l2_reg", 1e-5),
            learning_rate=meta.get("learning_rate", 7e-4),
        )

    @staticmethod
    def _safe_set_lookup_vocabulary(lookup, vocabulary: list[str]) -> None:
        try:
            lookup.set_vocabulary(vocabulary)
        except Exception:
            lookup.set_vocabulary(vocabulary[1:])

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_json(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo requerido: '{path}'")
