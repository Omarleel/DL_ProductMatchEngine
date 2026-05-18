from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.core.config import get_settings
from app.schemas.clasificador import (
    ClasificacionBatchRequest,
    PrediccionBatchResponse,
    EntrenamientoRequest,
)
from app.services.inferencia_clasificador_service import predecir_desde_items
from app.services.entrenamiento_clasificador_service import entrenar_modelo
from app.services.tenant_service import DEFAULT_TENANT, get_tenant_artifacts_dir, normalizar_tenant
from ml_pipeline.utils.retraining import build_run_id
import json

router = APIRouter(prefix="/clasificador", tags=["clasificador"])


@router.post("/predecir", response_model=PrediccionBatchResponse)
def clasificar(payload: ClasificacionBatchRequest) -> PrediccionBatchResponse:
    try:
        tenant = normalizar_tenant(payload.tenant)
        items_dict = [item.model_dump() for item in payload.items]

        resultados = predecir_desde_items(
            items=items_dict,
            tenant=tenant,
            include_factor_debug=payload.include_factor_debug,
        )

        return PrediccionBatchResponse(tenant=tenant, total=len(resultados), resultados=resultados)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {exc}") from exc


@router.post("/entrenar")
async def iniciar_entrenamiento(payload: EntrenamientoRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        tenant = normalizar_tenant(payload.tenant)
        run_id = build_run_id(f"api_clasificador_{tenant}")

        background_tasks.add_task(
            entrenar_modelo,
            tenant=tenant,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            force_replace=payload.force_replace,
            run_id=run_id,
        )

        return {
            "status": "procesando",
            "tenant": tenant,
            "run_id": run_id,
            "force_replace": payload.force_replace,
            "mensaje": (
                "El reentrenamiento del clasificador comenzó en segundo plano para el tenant indicado. "
                "El modelo campeón solo será reemplazado si el candidato mejora en la validación actual, "
                "salvo que 'force_replace=true'."
            ),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al iniciar entrenamiento: {exc}") from exc


@router.get("/reporte")
def obtener_ultimo_reporte(tenant: str = DEFAULT_TENANT) -> dict:
    """
    Retorna el JSON con las métricas e historial del último reentrenamiento del clasificador del tenant.
    """
    try:
        tenant_norm = normalizar_tenant(tenant)
        settings = get_settings()
        reporte_path = get_tenant_artifacts_dir(tenant_norm) / settings.clasificador_model_name / "train_report.json"

        if not reporte_path.exists():
            raise HTTPException(
                status_code=444,
                detail=f"No se encontró ningún reporte para el clasificador del tenant '{tenant_norm}'.",
            )

        with open(reporte_path, "r", encoding="utf-8") as f:
            reporte = json.load(f)
        return {"tenant": tenant_norm, **reporte}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer el reporte del clasificador: {exc}",
        )
