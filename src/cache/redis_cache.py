"""Redis caching and rate limiting implementation.

Provides:
- Key-value caching with TTL
- Job state management
- Sliding window rate limiting
- Embedding cache
"""
import json
import time
from datetime import datetime
from typing import Any
from uuid import UUID

import redis.asyncio as redis

from src.core.config import settings
from src.core.exceptions import CacheError, RateLimitExceededError
from src.core.logging import get_logger
from src.core.models import JobStatus

logger = get_logger(__name__)


class RedisCache:
    """Redis-based cache with rate limiting support.
    
    Provides async methods for:
    - General key-value caching
    - Job state management
    - Sliding window rate limiting
    - Embedding storage
    
    Example:
        cache = RedisCache()
        await cache.connect()
        
        # Cache a value
        await cache.set("key", {"data": "value"}, ttl=3600)
        
        # Check rate limit
        allowed = await cache.check_rate_limit("user:123")
        
        await cache.close()
    """

    def __init__(self, url: str | None = None) -> None:
        """Initialize Redis cache.
        
        Args:
            url: Redis URL (defaults to settings.redis_url_str)
        """
        self.url = url or settings.redis_url_str
        self._client: redis.Redis | None = None
        self._connected = False

    @property
    def client(self) -> redis.Redis:
        """Get Redis client, ensuring connection."""
        if not self._client or not self._connected:
            raise CacheError("Redis not connected. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        try:
            self._client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info("Connected to Redis", url=self.url)

        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise CacheError(f"Redis connection failed: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._connected = False
            logger.info("Redis connection closed")

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    # ============ General Caching ============

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get a cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in cache", key=key)
            return None
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: dict[str, Any] | list[Any] | str,
        ttl: int | None = None,
    ) -> bool:
        """Set a cached value.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (None for no expiry)
            
        Returns:
            True if successful
        """
        try:
            serialized = json.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
            return True
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete a cached value.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self.client.exists(key) > 0
        except Exception:
            return False

    # ============ Job State Management ============

    async def set_job(self, job_id: UUID, data: dict[str, Any]) -> bool:
        """Store job state.
        
        Args:
            job_id: Job identifier
            data: Job data
            
        Returns:
            True if successful
        """
        key = f"job:{job_id}"
        try:
            await self.client.hset(key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in data.items()
            })
            # Set TTL of 7 days for job data
            await self.client.expire(key, 86400 * 7)
            return True
        except Exception as e:
            logger.error("Failed to set job", job_id=str(job_id), error=str(e))
            return False

    async def get_job(self, job_id: UUID) -> dict[str, Any] | None:
        """Get job state.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job data or None if not found
        """
        key = f"job:{job_id}"
        try:
            data = await self.client.hgetall(key)
            if not data:
                return None
            return data
        except Exception as e:
            logger.error("Failed to get job", job_id=str(job_id), error=str(e))
            return None

    async def update_job(self, job_id: UUID, updates: dict[str, Any]) -> bool:
        """Update job fields.
        
        Args:
            job_id: Job identifier
            updates: Fields to update
            
        Returns:
            True if successful
        """
        key = f"job:{job_id}"
        try:
            await self.client.hset(key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in updates.items()
            })
            return True
        except Exception as e:
            logger.error("Failed to update job", job_id=str(job_id), error=str(e))
            return False

    async def update_job_status(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: float | None = None,
        error: str | None = None,
    ) -> bool:
        """Update job status with optional progress and error.
        
        Args:
            job_id: Job identifier
            status: New status
            progress: Optional progress value
            error: Optional error message
            
        Returns:
            True if successful
        """
        updates: dict[str, Any] = {
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if progress is not None:
            updates["progress"] = progress
        if error:
            updates["error"] = error

        return await self.update_job(job_id, updates)

    async def delete_job(self, job_id: UUID) -> bool:
        """Delete job state.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if successful
        """
        key = f"job:{job_id}"
        try:
            await self.client.delete(key)
            # Also delete related keys
            pattern = f"job:{job_id}:*"
            async for key in self.client.scan_iter(match=pattern):
                await self.client.delete(key)
            return True
        except Exception as e:
            logger.error("Failed to delete job", job_id=str(job_id), error=str(e))
            return False

    # ============ Rate Limiting ============

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int | None = None,
        window: int | None = None,
    ) -> tuple[bool, int, int]:
        """Check and update rate limit using sliding window.
        
        Args:
            identifier: Unique identifier (e.g., IP, user ID)
            limit: Request limit (defaults to settings)
            window: Time window in seconds (defaults to settings)
            
        Returns:
            Tuple of (allowed, remaining, reset_time)
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        limit = limit or settings.rate_limit_requests
        window = window or settings.rate_limit_window

        key = f"ratelimit:{identifier}"
        now = time.time()
        window_start = now - window

        try:
            pipe = self.client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Add current request
            pipe.zadd(key, {str(now): now})

            # Count requests in window
            pipe.zcard(key)

            # Set expiry
            pipe.expire(key, window)

            results = await pipe.execute()
            count = results[2]

            remaining = max(0, limit - count)
            reset_time = int(now + window)

            if count > limit:
                raise RateLimitExceededError(retry_after=window)

            return True, remaining, reset_time

        except RateLimitExceededError:
            raise
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            # Fail open - allow request if Redis fails
            return True, limit, int(now + window)

    async def get_rate_limit_info(
        self,
        identifier: str,
        window: int | None = None,
    ) -> dict[str, int]:
        """Get current rate limit status without incrementing.
        
        Args:
            identifier: Unique identifier
            window: Time window in seconds
            
        Returns:
            Dict with limit info
        """
        window = window or settings.rate_limit_window
        key = f"ratelimit:{identifier}"
        now = time.time()
        window_start = now - window

        try:
            # Count requests in current window
            count = await self.client.zcount(key, window_start, now)

            return {
                "count": count,
                "limit": settings.rate_limit_requests,
                "remaining": max(0, settings.rate_limit_requests - count),
                "window": window,
                "reset": int(now + window),
            }
        except Exception:
            return {
                "count": 0,
                "limit": settings.rate_limit_requests,
                "remaining": settings.rate_limit_requests,
                "window": window,
                "reset": int(now + window),
            }

    # ============ Embedding Cache ============

    async def cache_embedding(
        self,
        job_id: UUID,
        frame_index: int,
        embedding: list[float],
        ttl: int = 86400 * 7,  # 7 days
    ) -> bool:
        """Cache a frame embedding.
        
        Args:
            job_id: Job identifier
            frame_index: Frame index
            embedding: Embedding vector
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        key = f"embedding:{job_id}:{frame_index}"
        try:
            await self.client.setex(key, ttl, json.dumps(embedding))
            return True
        except Exception as e:
            logger.error(
                "Failed to cache embedding",
                job_id=str(job_id),
                frame_index=frame_index,
                error=str(e),
            )
            return False

    async def get_embedding(
        self,
        job_id: UUID,
        frame_index: int,
    ) -> list[float] | None:
        """Get a cached embedding.
        
        Args:
            job_id: Job identifier
            frame_index: Frame index
            
        Returns:
            Embedding vector or None
        """
        key = f"embedding:{job_id}:{frame_index}"
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(
                "Failed to get embedding",
                job_id=str(job_id),
                frame_index=frame_index,
                error=str(e),
            )
            return None

    async def get_all_embeddings(
        self,
        job_id: UUID,
    ) -> dict[int, list[float]]:
        """Get all cached embeddings for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dict mapping frame index to embedding
        """
        pattern = f"embedding:{job_id}:*"
        embeddings: dict[int, list[float]] = {}

        try:
            async for key in self.client.scan_iter(match=pattern):
                # Extract frame index from key
                parts = key.split(":")
                if len(parts) >= 3:
                    frame_index = int(parts[2])
                    value = await self.client.get(key)
                    if value:
                        embeddings[frame_index] = json.loads(value)

            return embeddings
        except Exception as e:
            logger.error(
                "Failed to get embeddings",
                job_id=str(job_id),
                error=str(e),
            )
            return {}
