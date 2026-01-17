"""Unit tests for GPU parallelization infrastructure."""
import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.parallel_encoder import (
    BatchItem,
    BatchQueue,
    DeviceAllocation,
    DeviceInfo,
    DeviceManager,
    DeviceType,
    EncodingPipeline,
)


class TestDeviceManager:
    """Tests for DeviceManager."""

    def test_initialization(self) -> None:
        """Test device manager initializes correctly."""
        manager = DeviceManager()
        manager.initialize()
        
        assert manager._initialized is True
        # Should always have CPU as fallback
        assert any(d.device_type == DeviceType.CPU for d in manager._devices)

    def test_has_gpu_without_cuda(self) -> None:
        """Test has_gpu returns False when no GPU available."""
        manager = DeviceManager()
        
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                manager.initialize()
                assert manager.has_gpu is False

    def test_allocate_returns_device_allocation(self) -> None:
        """Test allocate returns DeviceAllocation."""
        manager = DeviceManager()
        
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                manager.initialize()
                allocation = manager.allocate("clip")
                
                assert isinstance(allocation, DeviceAllocation)
                assert allocation.device == "cpu"
                assert allocation.max_batch_size > 0

    def test_allocate_dual_stream_single_gpu(self) -> None:
        """Test dual stream allocation with single GPU."""
        manager = DeviceManager()
        
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_properties") as mock_props:
                    mock_props.return_value = MagicMock(
                        name="Test GPU",
                        total_memory=8 * 1024 * 1024 * 1024  # 8GB
                    )
                    manager.initialize()
                    clip_alloc, xclip_alloc = manager.allocate_dual_stream()
                    
                    assert clip_alloc.device == "cuda:0"
                    assert xclip_alloc.device == "cuda:0"

    def test_allocate_dual_stream_multi_gpu(self) -> None:
        """Test dual stream allocation with multiple GPUs."""
        manager = DeviceManager()
        
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch("torch.cuda.get_device_properties") as mock_props:
                    mock_props.return_value = MagicMock(
                        name="Test GPU",
                        total_memory=8 * 1024 * 1024 * 1024
                    )
                    manager.initialize()
                    clip_alloc, xclip_alloc = manager.allocate_dual_stream()
                    
                    assert clip_alloc.device == "cuda:0"
                    assert xclip_alloc.device == "cuda:1"

    def test_get_device_string(self) -> None:
        """Test device string generation."""
        manager = DeviceManager()
        
        cuda_device = DeviceInfo(
            device_type=DeviceType.CUDA,
            device_id=1,
            name="GPU",
            total_memory_mb=8000,
            available_memory_mb=6000,
        )
        assert manager.get_device_string(cuda_device) == "cuda:1"
        
        cpu_device = DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            name="CPU",
            total_memory_mb=32000,
            available_memory_mb=16000,
        )
        assert manager.get_device_string(cpu_device) == "cpu"


class TestBatchQueue:
    """Tests for BatchQueue."""

    @pytest.mark.asyncio
    async def test_put_and_get_batch(self) -> None:
        """Test basic put and get operations."""
        queue: BatchQueue[int] = BatchQueue(batch_size=3)
        
        await queue.put(1)
        await queue.put(2)
        await queue.put(3)
        await queue.finish()
        
        batch = await queue.get_batch()
        
        assert batch is not None
        assert len(batch) == 3
        assert [b.data for b in batch] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_partial_batch_on_finish(self) -> None:
        """Test partial batch is returned when queue finishes."""
        queue: BatchQueue[int] = BatchQueue(batch_size=5)
        
        await queue.put(1)
        await queue.put(2)
        await queue.finish()
        
        batch = await queue.get_batch()
        
        assert batch is not None
        assert len(batch) == 2

    @pytest.mark.asyncio
    async def test_none_on_empty_queue(self) -> None:
        """Test None is returned when queue is empty and finished."""
        queue: BatchQueue[int] = BatchQueue(batch_size=3)
        await queue.finish()
        
        batch = await queue.get_batch()
        assert batch is None

    @pytest.mark.asyncio
    async def test_batch_item_metadata(self) -> None:
        """Test metadata is preserved in batch items."""
        queue: BatchQueue[str] = BatchQueue(batch_size=2)
        
        await queue.put("item1", {"key": "value1"})
        await queue.put("item2", {"key": "value2"})
        await queue.finish()
        
        batch = await queue.get_batch()
        
        assert batch is not None
        assert batch[0].metadata == {"key": "value1"}
        assert batch[1].metadata == {"key": "value2"}

    @pytest.mark.asyncio
    async def test_is_finished_property(self) -> None:
        """Test is_finished property."""
        queue: BatchQueue[int] = BatchQueue(batch_size=2)
        
        assert queue.is_finished is False
        
        await queue.put(1)
        await queue.finish()
        
        # Not empty yet
        assert queue.is_finished is False
        
        await queue.get_batch()
        
        # Now empty and finished
        assert queue.is_finished is True


class TestEncodingPipeline:
    """Tests for EncodingPipeline."""

    @pytest.mark.asyncio
    async def test_process_simple(self) -> None:
        """Test basic pipeline processing."""
        async def encoder_fn(batch: list[int]) -> list[int]:
            return [x * 2 for x in batch]
        
        pipeline: EncodingPipeline[int, int] = EncodingPipeline(
            encoder_fn=encoder_fn,
            batch_size=2,
        )
        
        async def items():
            for i in range(5):
                yield i
        
        results = await pipeline.process(items())
        
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_process_preserves_order(self) -> None:
        """Test that results are in original order."""
        async def encoder_fn(batch: list[str]) -> list[str]:
            # Simulate async processing that might reorder
            await asyncio.sleep(0.01)
            return [s.upper() for s in batch]
        
        pipeline: EncodingPipeline[str, str] = EncodingPipeline(
            encoder_fn=encoder_fn,
            batch_size=3,
        )
        
        async def items():
            for char in "abcdefg":
                yield char
        
        results = await pipeline.process(items())
        
        assert results == ["A", "B", "C", "D", "E", "F", "G"]

    @pytest.mark.asyncio
    async def test_process_with_numpy(self) -> None:
        """Test pipeline with numpy arrays (like embeddings)."""
        async def encoder_fn(batch: list[np.ndarray]) -> list[np.ndarray]:
            return [arr * 2 for arr in batch]
        
        pipeline: EncodingPipeline[np.ndarray, np.ndarray] = EncodingPipeline(
            encoder_fn=encoder_fn,
            batch_size=2,
        )
        
        async def items():
            for i in range(3):
                yield np.array([i, i+1, i+2])
        
        results = await pipeline.process(items())
        
        assert len(results) == 3
        np.testing.assert_array_equal(results[0], [0, 2, 4])
        np.testing.assert_array_equal(results[1], [2, 4, 6])
        np.testing.assert_array_equal(results[2], [4, 6, 8])
