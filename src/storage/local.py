"""Local filesystem storage backend."""
import shutil
from pathlib import Path
from typing import BinaryIO

import aiofiles

from src.core.config import settings
from src.core.exceptions import StorageError, VideoNotFoundError
from src.core.logging import get_logger
from src.storage.base import StorageBackend

logger = get_logger(__name__)


class LocalStorage(StorageBackend):
    """Local filesystem storage backend.
    
    Stores files in the configured storage path with
    hierarchical directory structure.
    
    Example:
        storage = LocalStorage()
        
        # Save a file
        path = await storage.save("videos/abc123/source.mp4", video_bytes)
        
        # Load it back
        data = await storage.load("videos/abc123/source.mp4")
    """

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path = base_path or settings.storage_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        """Resolve key to absolute path."""
        # Prevent path traversal
        clean_key = key.lstrip("/").replace("..", "")
        return self.base_path / clean_key

    async def save(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str | None = None,
    ) -> str:
        """Save data to local filesystem."""
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(data, bytes):
                async with aiofiles.open(path, "wb") as f:
                    await f.write(data)
            else:
                # File-like object
                async with aiofiles.open(path, "wb") as f:
                    while chunk := data.read(1024 * 1024):  # 1MB chunks
                        await f.write(chunk)

            logger.debug("File saved", path=str(path))
            return str(path)

        except Exception as e:
            logger.error("Failed to save file", key=key, error=str(e))
            raise StorageError(f"Failed to save {key}: {e}")

    async def load(self, key: str) -> bytes:
        """Load data from local filesystem."""
        path = self._resolve_path(key)

        if not path.exists():
            raise VideoNotFoundError(f"File not found: {key}")

        try:
            async with aiofiles.open(path, "rb") as f:
                return await f.read()

        except Exception as e:
            logger.error("Failed to load file", key=key, error=str(e))
            raise StorageError(f"Failed to load {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete file from local filesystem."""
        path = self._resolve_path(key)

        if not path.exists():
            return False

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

            logger.debug("File deleted", path=str(path))
            return True

        except Exception as e:
            logger.error("Failed to delete file", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if file exists."""
        path = self._resolve_path(key)
        return path.exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        """List files with prefix."""
        base = self._resolve_path(prefix) if prefix else self.base_path

        if not base.exists():
            return []

        keys = []
        for path in base.rglob("*"):
            if path.is_file():
                relative = path.relative_to(self.base_path)
                keys.append(str(relative))

        return keys

    def get_url(self, key: str) -> str:
        """Get file path as URL."""
        path = self._resolve_path(key)
        return f"file://{path}"

    async def get_path(self, key: str) -> Path:
        """Get resolved path for a key."""
        return self._resolve_path(key)

    async def copy(self, source_key: str, dest_key: str) -> str:
        """Copy a file within storage."""
        source = self._resolve_path(source_key)
        dest = self._resolve_path(dest_key)

        if not source.exists():
            raise VideoNotFoundError(f"Source not found: {source_key}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

        return str(dest)

    async def move(self, source_key: str, dest_key: str) -> str:
        """Move a file within storage."""
        source = self._resolve_path(source_key)
        dest = self._resolve_path(dest_key)

        if not source.exists():
            raise VideoNotFoundError(f"Source not found: {source_key}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, dest)

        return str(dest)
