from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field


DEFAULT_TENANT = "default"


class PrediccionItemRequest(BaseModel):
    RucProveedor: Optional[str] = None
    CodProducto: Optional[str] = None
    Producto: Optional[str] = None
    UnidadMedidaCompra: Optional[str] = None
    CostoCaja: Optional[float] = 0.0


class ClasificacionBatchRequest(BaseModel):
    tenant: str = Field(
        default=DEFAULT_TENANT,
        description="Tenant/sucursal que define qué maestro y modelo usar.",
        validation_alias=AliasChoices("tenant", "sede"),
    )
    items: List[PrediccionItemRequest] = Field(default_factory=list)
    include_factor_debug: bool = False


class PrediccionBatchResponse(BaseModel):
    tenant: str
    total: int
    resultados: List[Dict[str, Any]]


class EntrenamientoRequest(BaseModel):
    tenant: str = Field(
        default=DEFAULT_TENANT,
        description="Tenant/sucursal cuyos datasets y artefactos se entrenarán.",
        validation_alias=AliasChoices("tenant", "sede"),
    )
    epochs: int = Field(default=24, ge=1, le=100)
    batch_size: int = Field(default=256, ge=16, le=1024)
    force_replace: bool = False
