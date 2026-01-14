"""FastAPI application factory and lifespan management."""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import RateLimitMiddleware
from src.api.routes import health, ingest, query
from src.core.config import settings
from src.core.exceptions import FrameMindError, RateLimitExceededError
from src.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown."""
    # Startup
    setup_logging()
    logger.info(
        "Starting FrameMind",
        version=settings.app_version,
        environment=settings.environment.value,
    )

    # Initialize ML models (lazy loading in production)
    if not settings.is_production:
        logger.info("Development mode: ML models will be loaded on first use")

    # Store shared state in app.state
    app.state.ready = True

    yield

    # Shutdown
    logger.info("Shutting down FrameMind")
    app.state.ready = False

    # Cleanup tasks
    await asyncio.sleep(0.1)  # Allow pending requests to complete


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade Video Intelligence Engine with CLIP-based frame selection and VLM integration",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

    # Exception handlers
    @app.exception_handler(RateLimitExceededError)
    async def rate_limit_handler(
        request: Request, exc: RateLimitExceededError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": exc.message,
                "retry_after": exc.retry_after,
            },
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(FrameMindError)
    async def framemind_error_handler(
        request: Request, exc: FrameMindError
    ) -> JSONResponse:
        logger.error("Application error", error=exc.message, details=exc.details)
        return JSONResponse(
            status_code=400,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(
        ingest.router,
        prefix=f"{settings.api_prefix}/ingest",
        tags=["Ingestion"],
    )
    app.include_router(
        query.router,
        prefix=f"{settings.api_prefix}/query",
        tags=["Query"],
    )

    return app


# Application instance
app = create_app()


def run() -> None:
    """Run the application with uvicorn."""
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
