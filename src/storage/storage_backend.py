from abc import ABC, abstractmethod
from typing import Optional, List


class StorageBackend(ABC):
    """
    Abstract interface for a storage backend used to persist and retrieve
    scraper output files.

    Implementations may target local filesystems, cloud object stores
    (for example S3 or GCS), or any other persistent storage system.

    All paths are expected to be relative to the backend's configured base
    location and use a consistent, backend-defined path format.
    """

    @abstractmethod
    def write_json(self, path: str, data: dict) -> None:
        """Write a dict as JSON to the given path (relative to base)."""
        pass

    @abstractmethod
    def read_json(self, path: str) -> Optional[dict]:
        """Read JSON from the given path. Returns None if not found."""
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if a file exists at the given path."""
        pass

    @abstractmethod
    def list_files(self, prefix: str, suffix: str = "") -> List[str]:
        """List files under a prefix, optionally filtered by suffix."""
        pass