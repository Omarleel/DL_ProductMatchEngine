from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.schemas.clasificador import (
    ClasificacionBatchRequest,
    PrediccionBatchResponse,
    EntrenamientoRequest,
)
from app.services.inferencia_clasificador_service import predecir_desde_items
from app.services.entrenamiento_clasificador_service import entrenar_modelo
from ml_pipeline.utils.retraining import build_run_id

router = APIRouter(prefix="/clasificador", tags=["clasificador"])


@router.post("/predecir", response_model=PrediccionBatchResponse)
def clasificar(payload: ClasificacionBatchRequest) -> PrediccionBatchResponse:
    try:
        items_dict = [item.model_dump() for item in payload.items]

        resultados = predecir_desde_items(
            items=items_dict,
            include_factor_debug=payload.include_factor_debug,
        )

        return PrediccionBatchResponse(total=len(resultados), resultados=resultados)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {exc}") from exc


@router.post("/entrenar")
async def iniciar_entrenamiento(payload: EntrenamientoRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        run_id = build_run_id("api_clasificador")

        background_tasks.add_task(
            entrenar_modelo,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            force_replace=payload.force_replace,
            run_id=run_id,
        )

        return {
            "status": "procesando",
            "run_id": run_id,
            "force_replace": payload.force_replace,
            "mensaje": (
                "El reentrenamiento del clasificador comenzó en segundo plano. "
                "El modelo campeón solo será reemplazado si el candidato mejora en la validación actual, "
                "salvo que 'force_replace=true'."
            ),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al iniciar entrenamiento: {exc}") from exc