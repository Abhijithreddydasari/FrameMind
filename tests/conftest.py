"""Pytest configuration and fixtures."""

from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.core.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Test settings with safe defaults."""
    return Settings(
        environment="development",
        debug=True,
        redis_url="redis://localhost:6379/1",  # Use different DB for tests
        storage_path="./test_data",
        vlm_api_key="test-key",
    )


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Synchronous test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_redis() -> MagicMock:
    """Mock Redis client."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.setex = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.exists = AsyncMock(return_value=0)
    mock.hset = AsyncMock(return_value=1)
    mock.hgetall = AsyncMock(return_value={})
    mock.ping = AsyncMock(return_value=True)
    mock.pipeline = MagicMock(return_value=mock)
    mock.execute = AsyncMock(return_value=[0, 1, 1])
    return mock


@pytest.fixture
def sample_frame() -> Any:
    """Generate a sample frame for testing."""
    import numpy as np

    # Create a simple 640x480 RGB image
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames(sample_frame: Any) -> list[Any]:
    """Generate multiple sample frames."""
    import numpy as np

    frames = []
    for i in range(10):
        # Create frames with different colors to simulate scene changes
        frame = np.full((480, 640, 3), fill_value=i * 25, dtype=np.uint8)
        frames.append(frame)
    return frames
