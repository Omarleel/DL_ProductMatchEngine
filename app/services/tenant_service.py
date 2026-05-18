from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from app.core.config import get_settings


DEFAULT_TENANT = "default"
TENANTS_DIRNAME = "tenants"


_TENANT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TENANT_SEP_RE = re.compile(r"[_ .-]+")


def normalizar_tenant(tenant: Optional[str] = None, *, fallback: Optional[str] = None) -> str:
    """
    Normaliza el identificador de tenant/sucursal para usarlo de forma segura en rutas.

    - Si tenant viene vacío, usa fallback.
    - Si ambos vienen vacíos, usa DEFAULT_TENANT para compatibilidad con la estructura legacy.
    - Los tenants no-default se normalizan en mayúsculas y con separadores seguros.
    """
    raw = tenant if tenant not in (None, "") else fallback
    raw = DEFAULT_TENANT if raw in (None, "") else str(raw)
    cleaned = _TENANT_SAFE_RE.sub("_", raw.strip())
    cleaned = _TENANT_SEP_RE.sub("_", cleaned).strip("_")

    if not cleaned:
        cleaned = DEFAULT_TENANT

    if cleaned.lower() == DEFAULT_TENANT:
        return DEFAULT_TENANT

    return cleaned.upper()


def es_tenant_default(tenant: Optional[str]) -> bool:
    return normalizar_tenant(tenant) == DEFAULT_TENANT


def get_tenant_data_dir(tenant: Optional[str] = None) -> Path:
    settings = get_settings()
    tenant_norm = normalizar_tenant(tenant)
    if tenant_norm == DEFAULT_TENANT:
        return settings.data_dir
    return settings.data_dir / TENANTS_DIRNAME / tenant_norm


def get_tenant_raw_data_dir(tenant: Optional[str] = None) -> Path:
    tenant_norm = normalizar_tenant(tenant)
    settings = get_settings()
    if tenant_norm == DEFAULT_TENANT:
        return settings.raw_data_dir
    return get_tenant_data_dir(tenant_norm) / "raw"


def get_tenant_processed_data_dir(tenant: Optional[str] = None) -> Path:
    tenant_norm = normalizar_tenant(tenant)
    settings = get_settings()
    if tenant_norm == DEFAULT_TENANT:
        return settings.processed_data_dir
    return get_tenant_data_dir(tenant_norm) / "processed"


def get_tenant_results_data_dir(tenant: Optional[str] = None) -> Path:
    tenant_norm = normalizar_tenant(tenant)
    settings = get_settings()
    if tenant_norm == DEFAULT_TENANT:
        return settings.results_data_dir
    return get_tenant_data_dir(tenant_norm) / "results"


def get_tenant_artifacts_dir(tenant: Optional[str] = None) -> Path:
    settings = get_settings()
    tenant_norm = normalizar_tenant(tenant)
    if tenant_norm == DEFAULT_TENANT:
        return settings.artifacts_dir
    return settings.artifacts_dir / TENANTS_DIRNAME / tenant_norm


def ensure_tenant_dirs(tenant: Optional[str] = None) -> None:
    for directory in (
        get_tenant_raw_data_dir(tenant),
        get_tenant_processed_data_dir(tenant),
        get_tenant_results_data_dir(tenant),
        get_tenant_artifacts_dir(tenant),
    ):
        directory.mkdir(parents=True, exist_ok=True)


def list_known_tenants() -> list[str]:
    """Devuelve tenants detectados por datasets o artefactos existentes."""
    settings = get_settings()
    tenants: set[str] = set()

    legacy_markers = [
        settings.raw_data_dir / "maestro.csv",
        settings.artifacts_dir / settings.clasificador_model_name,
        settings.artifacts_dir / settings.homologador_model_name,
    ]
    if any(path.exists() for path in legacy_markers):
        tenants.add(DEFAULT_TENANT)

    for root in (settings.data_dir / TENANTS_DIRNAME, settings.artifacts_dir / TENANTS_DIRNAME):
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir():
                tenants.add(normalizar_tenant(path.name))

    return sorted(tenants, key=lambda item: (item == DEFAULT_TENANT, item))
