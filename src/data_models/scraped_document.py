from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, Optional
import hashlib
import json
import base64


@dataclass(frozen=True)
class ScrapedInterviewDocument:
    """
    Raw, unprocessed interview experience document emitted by scrapers.

    This class represents scraped data exactly as collected, with minimal
    validation and no normalization guarantees. Downstream preprocessing
    stages are responsible for cleaning, enrichment, and schema tightening.
    """
    document_id: str
    source_platform: str
    source_url: str
    title: str
    raw_content: str
    published_at: Optional[str]
    scraped_at: str
    scrape_type: str
    scrape_batch_id: str
    source_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert the document to a serializable dictionary."""
        data = asdict(self)
        # data["content_hash"] = self.content_hash
        return data

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ScrapedInterviewDocument':
        """Create instance from dictionary."""
        data_copy = data.copy()
        data_copy.pop('content_hash', None)  # Remove computed field
        return cls(**data_copy)

    @classmethod
    def from_json(cls, json_str: str) -> 'ScrapedInterviewDocument':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @staticmethod
    def now_iso() -> str:
        """Return the current UTC timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def generate_document_id(source_platform: str, source_url: str) -> str:
        """Generate a stable document identifier."""
        digest = hashlib.sha256(source_url.encode()).digest()
        short_hash = base64.urlsafe_b64encode(digest).decode()[:6]
        return f"{source_platform}_{short_hash}"