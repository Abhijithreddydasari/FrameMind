"""Abstract storage backend interface."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class StorageBackend(ABC):
    """Abstract base class for storage backends.
    
    Provides a consistent interface for local and cloud storage.
    """

    @abstractmethod
    async def save(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str | None = None,
    ) -> str:
        """Save data to storage.
        
        Args:
            key: Storage key/path
            data: Binary data or file-like object
            content_type: MIME type
            
        Returns:
            Storage URL or path
        """
        ...

    @abstractmethod
    async def load(self, key: str) -> bytes:
        """Load data from storage.
        
        Args:
            key: Storage key/path
            
        Returns:
            Binary data
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data from storage.
        
        Args:
            key: Storage key/path
            
        Returns:
            True if deleted
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists.
        
        Args:
            key: Storage key/path
            
        Returns:
            True if exists
        """
        ...

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with prefix.
        
        Args:
            prefix: Key prefix to filter by
            
        Returns:
            List of matching keys
        """
        ...

    @abstractmethod
    def get_url(self, key: str) -> str:
        """Get URL for a stored object.
        
        Args:
            key: Storage key/path
            
        Returns:
            URL or file path
        """
        ...
