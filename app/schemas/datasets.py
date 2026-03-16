from __future__ import annotations

from pydantic import BaseModel, Field


class GenerarDatasetsRequest(BaseModel):
    sede: str = Field(..., description="Clave de la sede configurada en .env, por ejemplo: CSM_IQUITOS")
    overwrite: bool = Field(default=True, description="Si es False, no sobrescribe archivos existentes")


class DatasetGeneradoInfo(BaseModel):
    nombre: str
    ruta: str
    filas: int


class GenerarDatasetsResponse(BaseModel):
    sede: str
    maestro: DatasetGeneradoInfo
    productos_facturas: DatasetGeneradoInfo