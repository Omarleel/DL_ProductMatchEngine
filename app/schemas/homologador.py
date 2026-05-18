from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import AliasChoices, BaseModel, Field


DEFAULT_TENANT = "default"


class HomologacionItemRequest(BaseModel):
    RucProveedor: Optional[str] = None
    CodProducto: Optional[str] = None
    Producto: Optional[str] = None
    UnidadMedidaCompra: Optional[str] = None
    CostoCaja: Optional[float] = 0.0
    ValorTotal: Optional[float] = 0.0


class HomologacionRequest(BaseModel):
    tenant: str = Field(
        default=DEFAULT_TENANT,
        description="Tenant/sucursal que define qué maestro y modelo usar.",
        validation_alias=AliasChoices("tenant", "sede"),
    )
    items: List[HomologacionItemRequest] = Field(default_factory=list)
    top_k: int = Field(default=2, ge=1, le=20)
    umbral_match: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_n_candidates: int = Field(default=80, ge=1, le=500)
    guardar_resultado: bool = Field(
        default=False,
        description="Si es True, guarda un CSV de resultados; por defecto no se guarda.",
    )


class HomologacionResponse(BaseModel):
    tenant: str
    total: int
    resultados: List[Dict[str, Any]]
    output_csv: Optional[str] = None


class EntrenamientoHomologadorRequest(BaseModel):
    tenant: str = Field(
        default=DEFAULT_TENANT,
        description="Tenant/sucursal cuyos datasets y artefactos se entrenarán.",
        validation_alias=AliasChoices("tenant", "sede"),
    )
    n_neg_por_pos: int = Field(default=4, ge=1, le=10)
    epochs_warmup: int = Field(default=10, ge=1, le=50)
    epochs_final: int = Field(default=16, ge=1, le=100)
    batch_size: int = Field(default=256, ge=16, le=1024)
    # Parámetros para Hard Negative Mining
    top_n_candidates: int = Field(default=60, ge=10, le=200)
    k_hard_per_positive: int = Field(default=2, ge=1, le=10)
    min_model_score: float = Field(default=0.25, ge=0.0, le=1.0)
    min_support: float = Field(default=0.08, ge=0.0, le=1.0)
    force_replace: bool = Field(default=False)
