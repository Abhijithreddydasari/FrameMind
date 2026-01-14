"""Storage module - File and metadata storage abstraction."""
from src.storage.base import StorageBackend
from src.storage.local import LocalStorage

__all__ = ["StorageBackend", "LocalStorage"]
