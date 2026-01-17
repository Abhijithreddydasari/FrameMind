"""GPU parallelization infrastructure for video encoding.

Provides device management, async batching, and producer-consumer
patterns for efficient GPU utilization during video processing.
"""
import asyncio
import gc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Generic, TypeVar

import torch
from numpy.typing import NDArray

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class DeviceType(str, Enum):
    """Supported compute devices."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class DeviceInfo:
    """Information about a compute device."""

    device_type: DeviceType
    device_id: int
    name: str
    total_memory_mb: int
    available_memory_mb: int


@dataclass
class DeviceAllocation:
    """Device allocation for an encoder."""

    device: str  # e.g., "cuda:0", "cpu"
    max_batch_size: int
    reserved_memory_mb: int


class DeviceManager:
    """Manages GPU/CPU device allocation for encoders.
    
    Auto-detects available devices and allocates them optimally
    across spatial (CLIP) and temporal (X-CLIP) encoders.
    
    Example:
        manager = DeviceManager()
        manager.initialize()
        
        clip_device = manager.allocate("clip", memory_mb=400)
        xclip_device = manager.allocate("xclip", memory_mb=600)
    """

    # Approximate memory requirements per model (MB)
    MODEL_MEMORY = {
        "clip": 350,
        "xclip": 400,
    }

    # Batch memory scaling (MB per item)
    BATCH_MEMORY = {
        "clip": 50,  # Per frame
        "xclip": 150,  # Per 16-frame clip
    }

    def __init__(self) -> None:
        self._devices: list[DeviceInfo] = []
        self._allocations: dict[str, DeviceAllocation] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Detect and catalog available devices."""
        if self._initialized:
            return

        self._devices = []

        # Check CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory // (1024 * 1024)
                # Estimate available (80% of total as safe limit)
                available = int(total_mem * 0.8)

                self._devices.append(
                    DeviceInfo(
                        device_type=DeviceType.CUDA,
                        device_id=i,
                        name=props.name,
                        total_memory_mb=total_mem,
                        available_memory_mb=available,
                    )
                )
                logger.info(
                    "CUDA device detected",
                    device=f"cuda:{i}",
                    name=props.name,
                    memory_mb=total_mem,
                )

        # Check MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            # MPS doesn't expose memory info easily
            self._devices.append(
                DeviceInfo(
                    device_type=DeviceType.MPS,
                    device_id=0,
                    name="Apple Silicon GPU",
                    total_memory_mb=8000,  # Conservative estimate
                    available_memory_mb=6000,
                )
            )
            logger.info("MPS device detected")

        # Always have CPU as fallback
        self._devices.append(
            DeviceInfo(
                device_type=DeviceType.CPU,
                device_id=0,
                name="CPU",
                total_memory_mb=32000,  # Assume 32GB system RAM
                available_memory_mb=16000,
            )
        )

        self._initialized = True
        logger.info("Device manager initialized", device_count=len(self._devices))

    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return any(d.device_type != DeviceType.CPU for d in self._devices)

    @property
    def gpu_count(self) -> int:
        """Count available GPUs."""
        return sum(1 for d in self._devices if d.device_type == DeviceType.CUDA)

    def get_device_string(self, device_info: DeviceInfo) -> str:
        """Convert DeviceInfo to PyTorch device string."""
        if device_info.device_type == DeviceType.CUDA:
            return f"cuda:{device_info.device_id}"
        elif device_info.device_type == DeviceType.MPS:
            return "mps"
        return "cpu"

    def allocate(
        self,
        encoder_name: str,
        memory_mb: int | None = None,
        prefer_gpu: bool = True,
    ) -> DeviceAllocation:
        """Allocate a device for an encoder.
        
        Args:
            encoder_name: Name of encoder (e.g., "clip", "xclip")
            memory_mb: Required memory (defaults to MODEL_MEMORY)
            prefer_gpu: Prefer GPU over CPU
            
        Returns:
            DeviceAllocation with device string and batch size
        """
        if not self._initialized:
            self.initialize()

        required_mem = memory_mb or self.MODEL_MEMORY.get(encoder_name, 500)

        # Find best device
        best_device: DeviceInfo | None = None

        if prefer_gpu:
            # Try GPU first
            gpu_devices = [d for d in self._devices if d.device_type != DeviceType.CPU]
            for device in gpu_devices:
                if device.available_memory_mb >= required_mem:
                    best_device = device
                    break

        if best_device is None:
            # Fall back to CPU
            best_device = next(d for d in self._devices if d.device_type == DeviceType.CPU)

        # Calculate max batch size
        batch_mem = self.BATCH_MEMORY.get(encoder_name, 100)
        available_for_batch = best_device.available_memory_mb - required_mem
        max_batch = max(1, available_for_batch // batch_mem)

        # Cap batch size for practical purposes
        max_batch = min(max_batch, 32 if encoder_name == "clip" else 8)

        device_str = self.get_device_string(best_device)

        allocation = DeviceAllocation(
            device=device_str,
            max_batch_size=max_batch,
            reserved_memory_mb=required_mem,
        )

        self._allocations[encoder_name] = allocation

        logger.info(
            "Device allocated",
            encoder=encoder_name,
            device=device_str,
            max_batch=max_batch,
        )

        return allocation

    def allocate_dual_stream(self) -> tuple[DeviceAllocation, DeviceAllocation]:
        """Allocate devices for both CLIP and X-CLIP.
        
        Optimizes for multi-GPU if available.
        
        Returns:
            Tuple of (clip_allocation, xclip_allocation)
        """
        if not self._initialized:
            self.initialize()

        if self.gpu_count >= 2:
            # Multi-GPU: separate devices
            clip_alloc = DeviceAllocation(
                device="cuda:0",
                max_batch_size=32,
                reserved_memory_mb=self.MODEL_MEMORY["clip"],
            )
            xclip_alloc = DeviceAllocation(
                device="cuda:1",
                max_batch_size=8,
                reserved_memory_mb=self.MODEL_MEMORY["xclip"],
            )
            logger.info("Multi-GPU allocation: CLIP→cuda:0, X-CLIP→cuda:1")

        elif self.gpu_count == 1:
            # Single GPU: shared device, smaller batches
            clip_alloc = DeviceAllocation(
                device="cuda:0",
                max_batch_size=16,
                reserved_memory_mb=self.MODEL_MEMORY["clip"],
            )
            xclip_alloc = DeviceAllocation(
                device="cuda:0",
                max_batch_size=4,
                reserved_memory_mb=self.MODEL_MEMORY["xclip"],
            )
            logger.info("Single-GPU allocation: CLIP+X-CLIP→cuda:0")

        else:
            # CPU only
            clip_alloc = self.allocate("clip", prefer_gpu=False)
            xclip_alloc = self.allocate("xclip", prefer_gpu=False)
            logger.info("CPU-only allocation")

        self._allocations["clip"] = clip_alloc
        self._allocations["xclip"] = xclip_alloc

        return clip_alloc, xclip_alloc

    def release(self, encoder_name: str) -> None:
        """Release device allocation."""
        if encoder_name in self._allocations:
            del self._allocations[encoder_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Device released", encoder=encoder_name)


@dataclass
class BatchItem(Generic[T]):
    """Item in a batch queue."""

    index: int
    data: T
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchQueue(Generic[T]):
    """Async queue with batching support for producer-consumer pattern.
    
    Collects items and yields them in batches for efficient GPU processing.
    
    Example:
        queue = BatchQueue[np.ndarray](batch_size=8, prefetch=2)
        
        # Producer
        for frame in frames:
            await queue.put(frame)
        await queue.finish()
        
        # Consumer
        async for batch in queue.batches():
            embeddings = encoder.encode(batch)
    """

    def __init__(
        self,
        batch_size: int = 8,
        prefetch: int = 2,
        timeout: float = 30.0,
    ) -> None:
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.timeout = timeout

        self._queue: asyncio.Queue[BatchItem[T] | None] = asyncio.Queue(
            maxsize=batch_size * (prefetch + 1)
        )
        self._finished = False
        self._item_count = 0

    async def put(self, item: T, metadata: dict[str, Any] | None = None) -> None:
        """Add item to queue."""
        batch_item = BatchItem(
            index=self._item_count,
            data=item,
            metadata=metadata or {},
        )
        await asyncio.wait_for(
            self._queue.put(batch_item),
            timeout=self.timeout,
        )
        self._item_count += 1

    async def finish(self) -> None:
        """Signal that no more items will be added."""
        self._finished = True
        await self._queue.put(None)  # Sentinel

    async def batches(self) -> "AsyncBatchIterator[T]":
        """Iterate over batches."""
        return AsyncBatchIterator(self)

    async def get_batch(self) -> list[BatchItem[T]] | None:
        """Get next batch of items.
        
        Returns:
            List of batch items, or None if queue is exhausted
        """
        batch: list[BatchItem[T]] = []

        while len(batch) < self.batch_size:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.timeout,
                )

                if item is None:
                    # End of queue
                    break

                batch.append(item)

            except asyncio.TimeoutError:
                # Return partial batch on timeout
                break

        if not batch:
            return None

        return batch

    @property
    def is_finished(self) -> bool:
        """Check if queue is finished and empty."""
        return self._finished and self._queue.empty()


class AsyncBatchIterator(Generic[T]):
    """Async iterator for batch queue."""

    def __init__(self, queue: BatchQueue[T]) -> None:
        self._queue = queue

    def __aiter__(self) -> "AsyncBatchIterator[T]":
        return self

    async def __anext__(self) -> list[BatchItem[T]]:
        batch = await self._queue.get_batch()
        if batch is None:
            raise StopAsyncIteration
        return batch


@dataclass
class EncodingTask(Generic[T, R]):
    """Task for the encoding pipeline."""

    input_data: T
    result: R | None = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


EncoderFn = Callable[[list[T]], Coroutine[Any, Any, list[R]]]


class EncodingPipeline(Generic[T, R]):
    """Producer-consumer pipeline for parallel encoding.
    
    Coordinates CPU extraction and GPU encoding with proper
    batching and async overlap.
    
    Example:
        pipeline = EncodingPipeline(
            encoder_fn=clip_scorer.embed_frames,
            batch_size=8,
            device="cuda:0",
        )
        
        async def extract_frames():
            for frame in video.frames():
                yield frame
        
        results = await pipeline.process(extract_frames())
    """

    def __init__(
        self,
        encoder_fn: EncoderFn[T, R],
        batch_size: int = 8,
        prefetch: int = 2,
        device: str = "cuda:0",
    ) -> None:
        self.encoder_fn = encoder_fn
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.device = device

        self._queue: BatchQueue[T] | None = None
        self._results: list[tuple[int, R]] = []
        self._errors: list[tuple[int, Exception]] = []

    async def process(
        self,
        items: "AsyncIterableType[T]",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[R]:
        """Process items through the encoding pipeline.
        
        Args:
            items: Async iterable of input items
            progress_callback: Optional callback(processed, total)
            
        Returns:
            List of encoded results in original order
        """
        self._queue = BatchQueue[T](
            batch_size=self.batch_size,
            prefetch=self.prefetch,
        )
        self._results = []
        self._errors = []

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(self._produce(items))
        consumer_task = asyncio.create_task(
            self._consume(progress_callback)
        )

        await asyncio.gather(producer_task, consumer_task)

        # Check for errors
        if self._errors:
            first_error = self._errors[0][1]
            raise RuntimeError(f"Encoding failed: {first_error}") from first_error

        # Sort results by original index
        self._results.sort(key=lambda x: x[0])
        return [r for _, r in self._results]

    async def _produce(self, items: "AsyncIterableType[T]") -> None:
        """Producer: extract items and add to queue."""
        assert self._queue is not None

        try:
            async for item in items:
                await self._queue.put(item)
        finally:
            await self._queue.finish()

    async def _consume(
        self,
        progress_callback: Callable[[int, int], None] | None,
    ) -> None:
        """Consumer: encode batches from queue."""
        assert self._queue is not None

        processed = 0

        while True:
            batch = await self._queue.get_batch()
            if batch is None:
                break

            try:
                # Extract data from batch items
                batch_data = [item.data for item in batch]
                batch_indices = [item.index for item in batch]

                # Encode batch
                results = await self.encoder_fn(batch_data)

                # Store results with indices
                for idx, result in zip(batch_indices, results):
                    self._results.append((idx, result))

                processed += len(batch)

                if progress_callback:
                    progress_callback(processed, -1)  # Total unknown

            except Exception as e:
                logger.error("Batch encoding failed", error=str(e))
                for item in batch:
                    self._errors.append((item.index, e))


# Type alias for async iterables
from typing import AsyncIterable as AsyncIterableType


class DualStreamPipeline:
    """Orchestrates parallel spatial + temporal encoding.
    
    Runs both streams concurrently, maximizing GPU utilization.
    
    Example:
        pipeline = DualStreamPipeline(
            clip_encoder=clip_scorer,
            xclip_encoder=xclip_encoder,
            device_manager=device_manager,
        )
        
        spatial_embs, temporal_embs = await pipeline.process(video_path)
    """

    def __init__(
        self,
        spatial_encoder: Any,  # CLIPScorer
        temporal_encoder: Any,  # XCLIPEncoder
        device_manager: DeviceManager,
    ) -> None:
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder
        self.device_manager = device_manager

    async def process(
        self,
        video_path: str,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """Process video through both streams concurrently.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback(stream, processed, total)
            
        Returns:
            Tuple of (spatial_embeddings, temporal_embeddings)
        """
        # Run both streams concurrently
        spatial_task = asyncio.create_task(
            self._process_spatial(video_path, progress_callback)
        )
        temporal_task = asyncio.create_task(
            self._process_temporal(video_path, progress_callback)
        )

        spatial_results, temporal_results = await asyncio.gather(
            spatial_task,
            temporal_task,
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(spatial_results, Exception):
            logger.error("Spatial stream failed", error=str(spatial_results))
            spatial_results = []

        if isinstance(temporal_results, Exception):
            logger.error("Temporal stream failed", error=str(temporal_results))
            temporal_results = []

        return spatial_results, temporal_results

    async def _process_spatial(
        self,
        video_path: str,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> list[Any]:
        """Process spatial (frame) stream."""
        # Import here to avoid circular imports
        from src.ml.frame_selector import FrameSelector

        selector = FrameSelector()
        await selector.initialize()

        try:
            result = await selector.select_from_video(video_path)

            if progress_callback:
                progress_callback("spatial", len(result.embeddings), len(result.embeddings))

            return result.embeddings

        finally:
            await selector.cleanup()

    async def _process_temporal(
        self,
        video_path: str,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> list[Any]:
        """Process temporal (clip) stream."""
        # Will be implemented with XCLIPEncoder
        if self.temporal_encoder is None:
            return []

        try:
            clips = await self.temporal_encoder.extract_and_encode(video_path)

            if progress_callback:
                progress_callback("temporal", len(clips), len(clips))

            return clips

        except Exception as e:
            logger.error("Temporal encoding failed", error=str(e))
            return []


# Singleton device manager
_device_manager: DeviceManager | None = None


def get_device_manager() -> DeviceManager:
    """Get global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
        _device_manager.initialize()
    return _device_manager
