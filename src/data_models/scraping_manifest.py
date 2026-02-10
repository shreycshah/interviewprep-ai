from dataclasses import dataclass, asdict
from typing import Optional, Dict

from src.storage.storage_backend import StorageBackend


@dataclass
class Manifest:
    """Scrape manifest — now uses StorageBackend instead of direct file I/O"""

    scrape_date: str
    scrape_type: str
    started_at: str
    completed_at: Optional[str]
    sources: Dict
    total_files: int
    last_sitemap_lastmod: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, storage: StorageBackend, path: str):
        """Save manifest via storage backend."""
        storage.write_json(path, self.to_dict())

    @classmethod
    def load(cls, storage: StorageBackend, path: str) -> Optional["Manifest"]:
        """Load manifest from storage backend."""
        data = storage.read_json(path)
        if data:
            return cls(**data)
        return None

    @classmethod
    def get_latest(cls, storage: StorageBackend, manifests_prefix: str) -> Optional["Manifest"]:
        """Load the most recent manifest file from storage."""
        # list_files returns sorted descending, so first match is latest
        files = storage.list_files(prefix=manifests_prefix, suffix=".json")
        # Filter to only scrape_ manifests
        manifest_files = [f for f in files if "scrape_" in f]
        if manifest_files:
            return cls.load(storage, manifest_files[0])
        return None