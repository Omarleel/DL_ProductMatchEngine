from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.datasets import GenerarDatasetsRequest, GenerarDatasetsResponse
from app.services.dataset_service import generar_datasets_raw

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/generar", response_model=GenerarDatasetsResponse)
def generar_datasets(payload: GenerarDatasetsRequest) -> GenerarDatasetsResponse:
    try:
        resultado = generar_datasets_raw(
            sede=payload.sede,
            overwrite=payload.overwrite,
        )
        return GenerarDatasetsResponse(**resultado)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generando datasets: {exc}") from exc