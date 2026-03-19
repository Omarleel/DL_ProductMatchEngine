from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    BackgroundTasks,
    Request,
    UploadFile,
    File,
    Form,
)

from app.schemas.homologador import (
    HomologacionRequest,
    HomologacionResponse,
    EntrenamientoHomologadorRequest,
)
from app.services.inferencia_homologacion_service import (
    homologar_items,
    cargar_items_desde_csv,
)
from app.services.entrenamiento_homologador_service import entrenar_modelo_homologador
from ml_pipeline.utils.retraining import build_run_id

router = APIRouter(prefix="/homologador", tags=["homologador"])


@router.post("/predecir", response_model=HomologacionResponse)
async def homologar(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    top_k: int = Form(default=2),
    umbral_match: Optional[float] = Form(default=None),
    top_n_candidates: int = Form(default=80),
) -> HomologacionResponse:
    try:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            payload_raw = await request.json()
            payload = HomologacionRequest.model_validate(payload_raw)

            items_dict = [item.model_dump() for item in payload.items]

            resultados, output_csv = homologar_items(
                items=items_dict,
                top_k=payload.top_k,
                umbral_match=payload.umbral_match,
                top_n_candidates=payload.top_n_candidates,
            )

            return HomologacionResponse(
                total=len(resultados),
                resultados=resultados,
                output_csv=output_csv,
            )

        if "multipart/form-data" in content_type:
            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail="Debes enviar un archivo CSV en el campo 'file'.",
                )

            if not file.filename or not file.filename.lower().endswith(".csv"):
                raise HTTPException(
                    status_code=400,
                    detail="El archivo debe tener extensión .csv.",
                )

            file_bytes = await file.read()
            items_dict = cargar_items_desde_csv(file_bytes)

            resultados, output_csv = homologar_items(
                items=items_dict,
                top_k=top_k,
                umbral_match=umbral_match,
                top_n_candidates=top_n_candidates,
            )

            return HomologacionResponse(
                total=len(resultados),
                resultados=resultados,
                output_csv=output_csv,
            )

        raise HTTPException(
            status_code=415,
            detail="Content-Type no soportado. Usa application/json o multipart/form-data.",
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error en homologación: {}".format(exc)) from exc


@router.post("/entrenar")
async def iniciar_entrenamiento(
    payload: EntrenamientoHomologadorRequest,
    background_tasks: BackgroundTasks,
) -> dict:
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
        raise HTTPException(
            status_code=500,
            detail="Error al iniciar entrenamiento del homologador: {}".format(exc),
        ) from exc