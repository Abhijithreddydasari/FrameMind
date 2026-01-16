"""Dependency injection for FastAPI routes."""
from typing import Annotated, AsyncGenerator

from fastapi import Depends, Request

from src.cache.redis_cache import RedisCache
from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.storage.metadata import MetadataStore

logger = get_logger(__name__)


# Settings dependency
def get_settings_dep() -> Settings:
    """Get application settings."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


# Redis cache dependency
async def get_redis_cache() -> AsyncGenerator[RedisCache, None]:
    """Get Redis cache instance."""
    cache = RedisCache()
    try:
        await cache.connect()
        yield cache
    finally:
        await cache.close()


RedisCacheDep = Annotated[RedisCache, Depends(get_redis_cache)]


# Request ID extraction
def get_request_id(request: Request) -> str:
    """Extract or generate request ID."""
    return request.headers.get("X-Request-ID", "unknown")


RequestIdDep = Annotated[str, Depends(get_request_id)]


# Readiness check
async def require_ready(request: Request) -> None:
    """Ensure application is ready to handle requests."""
    if not getattr(request.app.state, "ready", False):
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Service not ready")


ReadyDep = Annotated[None, Depends(require_ready)]


# Metadata store dependency
def get_metadata_store(request: Request) -> MetadataStore:
    """Get metadata store instance."""
    store = getattr(request.app.state, "metadata_store", None)
    if store is None:
        raise RuntimeError("Metadata store not initialized")
    return store


MetadataStoreDep = Annotated[MetadataStore, Depends(get_metadata_store)]
