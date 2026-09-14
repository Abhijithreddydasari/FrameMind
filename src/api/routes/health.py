"""Health check endpoints."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from src.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness check response with component status."""

    ready: bool
    checks: dict[str, bool]
    timestamp: datetime


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check - always returns OK if the service is running."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment.value,
        timestamp=datetime.utcnow(),
    )


@router.get("/live")
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Kubernetes readiness probe - checks all dependencies."""
    checks: dict[str, bool] = {}

    # Check app state
    checks["app"] = getattr(request.app.state, "ready", False)

    # Check Redis connectivity (non-blocking)
    try:
        checks["redis"] = await request.app.state.cache.ping()
    except Exception:
        checks["redis"] = False

    # Check storage path exists
    checks["storage"] = settings.storage_path.exists()
    try:
        from sqlalchemy import text

        async with request.app.state.metadata_store.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        checks["database"] = True
        checks["queue"] = bool(await request.app.state.queue.ping())
    except Exception:
        checks["database"] = False
        checks["queue"] = False

    if settings.autogaze_enabled:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as client:
                result = await client.get(
                    settings.autogaze_url.rstrip("/") + "/health",
                    headers={"Authorization": f"Bearer {settings.autogaze_token}"},
                )
                checks["autogaze"] = result.is_success and bool(result.json().get("cuda"))
        except Exception:
            checks["autogaze"] = False

    all_ready = all(checks.values())
    response.status_code = 200 if all_ready else 503

    return ReadinessResponse(
        ready=all_ready,
        checks=checks,
        timestamp=datetime.utcnow(),
    )


@router.get("/info")
async def info() -> dict[str, Any]:
    """Application information endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment.value,
        "features": {
            "clip_model": settings.clip_model,
            "vlm_provider": settings.vlm_provider,
            "storage_backend": settings.storage_backend,
        },
    }
