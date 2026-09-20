from fastapi import APIRouter

from app.core.config import get_settings
from app.models.api import HealthResponse


router = APIRouter(
    tags=["Health"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """
    Confirm that the FastAPI backend is running.
    """

    settings = get_settings()

    return HealthResponse(
        status="healthy",
        application=settings.app_name,
        environment=settings.app_env,
    )