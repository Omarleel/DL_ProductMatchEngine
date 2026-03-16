from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.schemas.homologador import (
    HomologacionRequest,
    HomologacionResponse,
    EntrenamientoHomologadorRequest,
)
from app.services.inferencia_homologacion_service import homologar_items
from app.services.entrenamiento_homologador_service import entrenar_modelo_homologador
from ml_pipeline.utils.retraining import build_run_id

router = APIRouter(prefix="/homologador", tags=["homologador"])


@router.post("/predecir", response_model=HomologacionResponse)
def homologar(payload: HomologacionRequest) -> HomologacionResponse:
    try:
        items_dict = [item.model_dump() for item in payload.items]

        resultados = homologar_items(
            items=items_dict,
            top_k=payload.top_k,
            umbral_match=payload.umbral_match,
            top_n_candidates=payload.top_n_candidates,
        )
        return HomologacionResponse(total=len(resultados), resultados=resultados)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en homologación: {exc}") from exc


@router.post("/entrenar")
async def iniciar_entrenamiento(payload: EntrenamientoHomologadorRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        run_id = build_run_id("api_homologador")

        background_tasks.add_task(
            entrenar_modelo_homologador,
            n_neg_por_pos=payload.n_neg_por_pos,
            epochs_warmup=payload.epochs_warmup,
            epochs_final=payload.epochs_final,
            batch_size=payload.batch_size,
            top_n_candidates=payload.top_n_candidates,
            k_hard_per_positive=payload.k_hard_per_positive,
            min_model_score=payload.min_model_score,
            min_support=payload.min_support,
            force_replace=payload.force_replace,
            run_id=run_id,
        )

        return {
            "status": "procesando",
            "run_id": run_id,
            "force_replace": payload.force_replace,
            "mensaje": (
                "El reentrenamiento del homologador comenzó en segundo plano. "
                "El modelo campeón solo será reemplazado si el candidato mejora en la validación actual, "
                "salvo que 'force_replace=true'."
            ),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al iniciar entrenamiento del homologador: {exc}") from exc