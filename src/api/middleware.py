"""FastAPI middleware components."""
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.config import settings
from src.core.logging import clear_log_context, get_logger, log_context

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis sliding window.
    
    Note: Actual rate limiting logic is in the cache module.
    This middleware adds rate limit headers to responses.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.enabled = settings.rate_limit_requests > 0

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """Process request with rate limit headers."""
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/ready", "/live"):
            return await call_next(request)

        response = await call_next(request)

        # Add rate limit headers (actual limiting done at route level)
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
        response.headers["X-RateLimit-Window"] = str(settings.rate_limit_window)

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """Log request and response details."""
        request_id = request.headers.get("X-Request-ID", "unknown")

        # Bind context for all logs in this request
        log_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "Request failed",
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise
        finally:
            clear_log_context()
