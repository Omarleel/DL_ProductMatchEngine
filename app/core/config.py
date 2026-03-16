from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)


class SedeDBConfig(BaseModel):
    nombre: str
    host: str
    user: str
    password: str
    database: str
    port: int = 1433


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "DL CSM API")
    app_env: str = os.getenv("APP_ENV", "dev")
    api_v1_prefix: str = "/api/v1"

    claisifcador_model_name: str = os.getenv("CLASIFICADOR_MODEL_NAME", "clasificador_productos_v1")
    homologador_model_name: str = os.getenv("HOMOLOGADOR_MODEL_NAME", "homologador_productos_v1")

    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    artifacts_dir: Path = BASE_DIR / "artifacts"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

def cargar_configuracion_sedes() -> Dict[str, SedeDBConfig]:
    sedes: Dict[str, SedeDBConfig] = {}
    for key, value in os.environ.items():
        if key.startswith("SQL_") and key.endswith("_HOST_DB"):
            parts = key.split("_")
            id_sede = "_".join(parts[1:-2]) 
            
            try:
                sedes[id_sede] = SedeDBConfig(
                    nombre=os.getenv(f"SQL_{id_sede}_NOMBRE", id_sede),
                    host=value,
                    user=os.getenv(f"SQL_{id_sede}_USER_DB", ""),
                    password=os.getenv(f"SQL_{id_sede}_PASSWORD_DB", ""),
                    database=os.getenv(f"SQL_{id_sede}_DATABASE_DB", ""),
                    port=int(os.getenv(f"SQL_{id_sede}_PORT_DB", 1433)),
                )
            except Exception as exc:
                print(f"Error cargando configuración para {id_sede}: {exc}")
    return sedes

CONFIG_SEDES = cargar_configuracion_sedes()