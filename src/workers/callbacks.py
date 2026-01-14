"""Job completion callbacks and webhook handlers.

Provides mechanisms for notifying external systems when
jobs complete or fail.
"""
from datetime import datetime
from typing import Any, Callable, Coroutine
from uuid import UUID

import httpx

from src.core.logging import get_logger

logger = get_logger(__name__)


# Type alias for callback functions
CallbackFn = Callable[[UUID, dict[str, Any]], Coroutine[Any, Any, None]]


class CallbackRegistry:
    """Registry for job completion callbacks.
    
    Allows registering handlers that are called when jobs
    reach terminal states (complete, failed, cancelled).
    """

    def __init__(self) -> None:
        self._callbacks: dict[str, list[CallbackFn]] = {
            "complete": [],
            "failed": [],
            "cancelled": [],
        }

    def on_complete(self, callback: CallbackFn) -> CallbackFn:
        """Register a callback for job completion."""
        self._callbacks["complete"].append(callback)
        return callback

    def on_failed(self, callback: CallbackFn) -> CallbackFn:
        """Register a callback for job failure."""
        self._callbacks["failed"].append(callback)
        return callback

    def on_cancelled(self, callback: CallbackFn) -> CallbackFn:
        """Register a callback for job cancellation."""
        self._callbacks["cancelled"].append(callback)
        return callback

    async def trigger(
        self,
        event: str,
        job_id: UUID,
        data: dict[str, Any],
    ) -> None:
        """Trigger all callbacks for an event.
        
        Args:
            event: Event type (complete, failed, cancelled)
            job_id: Job identifier
            data: Event data to pass to callbacks
        """
        callbacks = self._callbacks.get(event, [])

        for callback in callbacks:
            try:
                await callback(job_id, data)
            except Exception as e:
                logger.error(
                    "Callback failed",
                    event=event,
                    job_id=str(job_id),
                    callback=callback.__name__,
                    error=str(e),
                )


# Global callback registry
callback_registry = CallbackRegistry()


class WebhookNotifier:
    """Sends webhook notifications for job events.
    
    Example:
        notifier = WebhookNotifier()
        notifier.register_webhook("https://example.com/webhook")
        
        await notifier.notify_complete(job_id, result)
    """

    def __init__(self, timeout: float = 10.0, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._webhooks: list[str] = []

    def register_webhook(self, url: str) -> None:
        """Register a webhook URL for notifications."""
        if url not in self._webhooks:
            self._webhooks.append(url)
            logger.info("Webhook registered", url=url)

    def unregister_webhook(self, url: str) -> None:
        """Unregister a webhook URL."""
        if url in self._webhooks:
            self._webhooks.remove(url)
            logger.info("Webhook unregistered", url=url)

    async def _send_webhook(
        self,
        url: str,
        event: str,
        job_id: UUID,
        data: dict[str, Any],
    ) -> bool:
        """Send a single webhook notification with retries."""
        payload = {
            "event": event,
            "job_id": str(job_id),
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()

                logger.info(
                    "Webhook sent",
                    url=url,
                    event=event,
                    job_id=str(job_id),
                )
                return True

            except httpx.TimeoutException:
                logger.warning(
                    "Webhook timeout",
                    url=url,
                    attempt=attempt + 1,
                )
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "Webhook HTTP error",
                    url=url,
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                )
            except Exception as e:
                logger.error(
                    "Webhook failed",
                    url=url,
                    error=str(e),
                    attempt=attempt + 1,
                )

            # Exponential backoff
            if attempt < self.max_retries - 1:
                import asyncio

                await asyncio.sleep(2 ** attempt)

        return False

    async def notify(
        self,
        event: str,
        job_id: UUID,
        data: dict[str, Any],
    ) -> int:
        """Send notifications to all registered webhooks.
        
        Returns:
            Number of successful notifications
        """
        if not self._webhooks:
            return 0

        import asyncio

        results = await asyncio.gather(
            *[
                self._send_webhook(url, event, job_id, data)
                for url in self._webhooks
            ],
            return_exceptions=True,
        )

        successful = sum(1 for r in results if r is True)
        return successful

    async def notify_complete(
        self,
        job_id: UUID,
        result: dict[str, Any],
    ) -> int:
        """Notify webhooks of job completion."""
        return await self.notify("job.complete", job_id, result)

    async def notify_failed(
        self,
        job_id: UUID,
        error: str,
    ) -> int:
        """Notify webhooks of job failure."""
        return await self.notify("job.failed", job_id, {"error": error})

    async def notify_progress(
        self,
        job_id: UUID,
        progress: float,
        stage: str,
    ) -> int:
        """Notify webhooks of job progress."""
        return await self.notify(
            "job.progress",
            job_id,
            {"progress": progress, "stage": stage},
        )


# Default webhook notifier instance
webhook_notifier = WebhookNotifier()
