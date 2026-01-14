"""Pipeline orchestrator for managing job state transitions.

Provides a state machine for video processing jobs with
proper error handling and recovery capabilities.
"""
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from src.core.logging import get_logger
from src.core.models import JobStatus

logger = get_logger(__name__)


class PipelineStage(str, Enum):
    """Pipeline processing stages."""

    UPLOAD = "upload"
    PREPROCESS = "preprocess"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    COMPLETE = "complete"
    FAILED = "failed"


# Valid state transitions
VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.UPLOADING, JobStatus.PROCESSING, JobStatus.CANCELLED},
    JobStatus.UPLOADING: {JobStatus.PROCESSING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.PROCESSING: {JobStatus.EXTRACTING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.EXTRACTING: {JobStatus.ANALYZING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.ANALYZING: {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.COMPLETE: set(),  # Terminal state
    JobStatus.FAILED: {JobStatus.PENDING},  # Can retry
    JobStatus.CANCELLED: set(),  # Terminal state
}


class InvalidTransitionError(Exception):
    """Invalid job state transition."""

    def __init__(self, current: JobStatus, target: JobStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Cannot transition from {current.value} to {target.value}")


class JobOrchestrator:
    """Manages job lifecycle and state transitions.
    
    Ensures jobs follow valid state transitions and handles
    error recovery and retry logic.
    
    Example:
        orchestrator = JobOrchestrator(cache)
        
        # Transition job to next state
        await orchestrator.transition(job_id, JobStatus.PROCESSING)
        
        # Handle failure
        await orchestrator.fail(job_id, "Processing error")
        
        # Retry failed job
        await orchestrator.retry(job_id)
    """

    def __init__(self, cache: Any) -> None:
        """Initialize orchestrator with cache backend.
        
        Args:
            cache: Redis cache instance for job state storage
        """
        self.cache = cache

    async def get_status(self, job_id: UUID) -> JobStatus:
        """Get current job status."""
        job_data = await self.cache.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job not found: {job_id}")

        return JobStatus(job_data.get("status", "pending"))

    async def transition(
        self,
        job_id: UUID,
        target_status: JobStatus,
        progress: float | None = None,
        message: str | None = None,
    ) -> bool:
        """Transition job to a new status.
        
        Args:
            job_id: Job identifier
            target_status: Target status
            progress: Optional progress value (0.0 to 1.0)
            message: Optional status message
            
        Returns:
            True if transition was successful
            
        Raises:
            InvalidTransitionError: If transition is not valid
        """
        current_status = await self.get_status(job_id)

        if target_status not in VALID_TRANSITIONS.get(current_status, set()):
            raise InvalidTransitionError(current_status, target_status)

        update_data: dict[str, Any] = {
            "status": target_status.value,
            "updated_at": datetime.utcnow().isoformat(),
        }

        if progress is not None:
            update_data["progress"] = progress
        if message:
            update_data["message"] = message

        await self.cache.update_job(job_id, update_data)

        logger.info(
            "Job status transition",
            job_id=str(job_id),
            from_status=current_status.value,
            to_status=target_status.value,
        )

        return True

    async def fail(
        self,
        job_id: UUID,
        error: str,
        recoverable: bool = True,
    ) -> None:
        """Mark job as failed.
        
        Args:
            job_id: Job identifier
            error: Error message
            recoverable: Whether the job can be retried
        """
        current_status = await self.get_status(job_id)

        if current_status in (JobStatus.COMPLETE, JobStatus.CANCELLED):
            logger.warning(
                "Cannot fail job in terminal state",
                job_id=str(job_id),
                current_status=current_status.value,
            )
            return

        update_data = {
            "status": JobStatus.FAILED.value,
            "error": error,
            "recoverable": recoverable,
            "updated_at": datetime.utcnow().isoformat(),
        }

        await self.cache.update_job(job_id, update_data)

        logger.error(
            "Job failed",
            job_id=str(job_id),
            error=error,
            recoverable=recoverable,
        )

    async def retry(self, job_id: UUID) -> bool:
        """Retry a failed job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if retry was initiated
        """
        job_data = await self.cache.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job not found: {job_id}")

        current_status = JobStatus(job_data.get("status", "pending"))

        if current_status != JobStatus.FAILED:
            logger.warning(
                "Can only retry failed jobs",
                job_id=str(job_id),
                current_status=current_status.value,
            )
            return False

        if not job_data.get("recoverable", True):
            logger.warning(
                "Job is not recoverable",
                job_id=str(job_id),
            )
            return False

        # Check retry count
        retry_count = int(job_data.get("retry_count", 0))
        max_retries = 3

        if retry_count >= max_retries:
            logger.warning(
                "Max retries exceeded",
                job_id=str(job_id),
                retry_count=retry_count,
            )
            return False

        # Reset to pending
        update_data = {
            "status": JobStatus.PENDING.value,
            "error": None,
            "progress": 0.0,
            "retry_count": retry_count + 1,
            "updated_at": datetime.utcnow().isoformat(),
        }

        await self.cache.update_job(job_id, update_data)

        logger.info(
            "Job retry initiated",
            job_id=str(job_id),
            retry_count=retry_count + 1,
        )

        return True

    async def cancel(self, job_id: UUID) -> bool:
        """Cancel a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancellation was successful
        """
        current_status = await self.get_status(job_id)

        if current_status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
            logger.warning(
                "Cannot cancel job in terminal state",
                job_id=str(job_id),
                current_status=current_status.value,
            )
            return False

        await self.transition(job_id, JobStatus.CANCELLED, message="Cancelled by user")

        return True

    async def get_progress(self, job_id: UUID) -> dict[str, Any]:
        """Get detailed job progress.
        
        Returns:
            Dict with status, progress, stage info, and timing
        """
        job_data = await self.cache.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job not found: {job_id}")

        status = JobStatus(job_data.get("status", "pending"))
        progress = float(job_data.get("progress", 0.0))

        # Calculate stage from status
        stage_map = {
            JobStatus.PENDING: PipelineStage.UPLOAD,
            JobStatus.UPLOADING: PipelineStage.UPLOAD,
            JobStatus.PROCESSING: PipelineStage.PREPROCESS,
            JobStatus.EXTRACTING: PipelineStage.EXTRACT,
            JobStatus.ANALYZING: PipelineStage.ANALYZE,
            JobStatus.COMPLETE: PipelineStage.COMPLETE,
            JobStatus.FAILED: PipelineStage.FAILED,
            JobStatus.CANCELLED: PipelineStage.FAILED,
        }

        return {
            "job_id": str(job_id),
            "status": status.value,
            "stage": stage_map.get(status, PipelineStage.FAILED).value,
            "progress": progress,
            "message": job_data.get("message"),
            "error": job_data.get("error"),
            "created_at": job_data.get("created_at"),
            "updated_at": job_data.get("updated_at"),
            "retry_count": int(job_data.get("retry_count", 0)),
        }
