from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field
from app.services.tenant_service import DEFAULT_TENANT

class GenerarDatasetsRequest(BaseModel):
    tenant: str = Field(
        default=DEFAULT_TENANT,
        description="Tenant/sucursal donde se guardarán los datasets.",
        validation_alias=AliasChoices("tenant", "sede"),
    ),
    overwrite: bool = Field(default=True, description="Si es False, no sobrescribe archivos existentes")


class DatasetGeneradoInfo(BaseModel):
    nombre: str
    ruta: str
    filas: int


class GenerarDatasetsResponse(BaseModel):
    tenant: str
    maestro: DatasetGeneradoInfo
    productos_facturas: DatasetGeneradoInfo
