from __future__ import annotations
import numpy as np
import tensorflow as tf


from pathlib import Path
from typing import Union

SEED = 42
def init_seeds() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"



def ensure_project_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def dataset_path(nombre: PathLike) -> Path:
    return RAW_DATA_DIR / Path(nombre)


def model_path(nombre: PathLike) -> Path:
    return ARTIFACTS_DIR / Path(nombre)


def result_path(nombre: PathLike) -> Path:
    return RESULTADOS_DIR / Path(nombre)


def require_file(path: PathLike, descripcion: str = "archivo") -> Path:
    ruta = Path(path)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el {descripcion}: '{ruta}'")
    return ruta


def processed_data_path(nombre: PathLike) -> Path:
    return PROCESSED_DATA_DIR / Path(nombre)


def result_path(nombre: PathLike) -> Path:
    return PROCESSED_DATA_DIR / Path(nombre)
