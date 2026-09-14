"""Database-authoritative state with disposable Redis mirrors."""

from uuid import UUID

from src.core.logging import get_logger

logger = get_logger(__name__)


class JobOrchestrator:
    def __init__(self, store, redis=None):
        self.store, self.redis = store, redis

    async def update(self, job_id, **values):
        await self.store.update_job(UUID(str(job_id)), values)
        job = await self.store.get_job(UUID(str(job_id)))
        if job and self.redis:
            try:
                await self.redis.hset(
                    f"job:{job_id}",
                    mapping={
                        key: str(job[key])
                        for key in ("status", "progress", "created_at", "updated_at")
                    },
                )
            except Exception:
                logger.warning("Job cache unavailable", job_id=str(job_id))
        return job

    async def cancelled(self, job_id):
        job = await self.store.get_job(UUID(str(job_id)))
        return job is None or job["status"] == "cancelled"
