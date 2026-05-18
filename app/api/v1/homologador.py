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

from app.core.config import get_settings
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
from app.services.tenant_service import DEFAULT_TENANT, get_tenant_artifacts_dir, normalizar_tenant
from ml_pipeline.utils.retraining import build_run_id
import json

router = APIRouter(prefix="/homologador", tags=["homologador"])


@router.post("/predecir", response_model=HomologacionResponse)
async def homologar(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    tenant: str = Form(default=DEFAULT_TENANT),
    top_k: int = Form(default=2),
    umbral_match: Optional[float] = Form(default=None),
    top_n_candidates: int = Form(default=80),
    guardar_resultado: bool = Form(default=False),
) -> HomologacionResponse:
    try:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            payload_raw = await request.json()
            payload = HomologacionRequest.model_validate(payload_raw)
            tenant_norm = normalizar_tenant(payload.tenant)

            items_dict = [item.model_dump() for item in payload.items]

            resultados, output_csv = homologar_items(
                items=items_dict,
                tenant=tenant_norm,
                top_k=payload.top_k,
                umbral_match=payload.umbral_match,
                top_n_candidates=payload.top_n_candidates,
                guardar_resultado=payload.guardar_resultado,
            )

            return HomologacionResponse(
                tenant=tenant_norm,
                total=len(resultados),
                resultados=resultados,
                output_csv=output_csv,
            )

        if "multipart/form-data" in content_type:
            tenant_norm = normalizar_tenant(tenant)
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
                tenant=tenant_norm,
                top_k=top_k,
                umbral_match=umbral_match,
                top_n_candidates=top_n_candidates,
                guardar_resultado=guardar_resultado,
            )

            return HomologacionResponse(
                tenant=tenant_norm,
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
        tenant = normalizar_tenant(payload.tenant)
        run_id = build_run_id(f"api_homologador_{tenant}")

        background_tasks.add_task(
            entrenar_modelo_homologador,
            tenant=tenant,
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
            "tenant": tenant,
            "run_id": run_id,
            "force_replace": payload.force_replace,
            "mensaje": (
                "El reentrenamiento del homologador comenzó en segundo plano para el tenant indicado. "
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


@router.get("/reporte")
def obtener_ultimo_reporte_homologador(tenant: str = DEFAULT_TENANT) -> dict:
    """
    Retorna el JSON con métricas del último reentrenamiento del homologador del tenant.
    """
    try:
        tenant_norm = normalizar_tenant(tenant)
        settings = get_settings()
        reporte_path = get_tenant_artifacts_dir(tenant_norm) / settings.homologador_model_name / "train_report.json"

        if not reporte_path.exists():
            raise HTTPException(
                status_code=444,
                detail=f"No se encontró ningún reporte para el homologador del tenant '{tenant_norm}'.",
            )

        with open(reporte_path, "r", encoding="utf-8") as f:
            reporte = json.load(f)
        return {"tenant": tenant_norm, **reporte}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer el reporte del homologador: {exc}",
        )
