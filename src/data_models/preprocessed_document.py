from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
import json

# ── Valid enum values (mirrors DB enums) ──
VALID_PLATFORMS: Set[str] = {"leetcode", "geeksforgeeks", "medium"}
VALID_EXPERIENCE: Set[str] = {"intern", "entry", "mid", "senior", "staff", "unknown"}
VALID_OUTCOMES: Set[str] = {"offer", "reject", "pending", "unknown"}
VALID_DIFFICULTY: Set[str] = {"easy", "medium", "hard", "unknown"}
VALID_INTERVIEW_TYPES: Set[str] = {
    "phone_screen", "onsite", "online_assessment",
    "virtual", "on_campus", "off_campus", "walk_in",
}

# ── Platform alias mapping ──
PLATFORM_NORMALIZE: Dict[str, str] = {
    "leetcode": "leetcode",
    "lc": "leetcode",
    "geeksforgeeks": "geeksforgeeks",
    "gfg": "geeksforgeeks",
    "medium": "medium",
}


@dataclass(frozen=True)
class ProcessedInterviewDocument:
    """
    Fully processed interview document ready for persistence
    in the processed_documents table.

    All validation, normalization, and defaulting happens inside
    __post_init__ so that construction itself is the single
    validation gate.
    """

    # ── Identity & Lineage ──
    document_id: str
    source_platform: str
    source_url: str
    content_hash: str

    # ── Content ──
    title: str
    content: str
    word_count: int

    # ── Extracted Entities (required) ──
    company: str
    role: str
    experience_level: str = None

    # ── Extracted Structure ──
    interview_outcome: str = None
    difficulty: str = None
    num_rounds: Optional[int] = None
    interview_types: List[str] = field(default_factory=list)

    # ── Topics ──
    topics: List[str] = field(default_factory=list)

    # ── Timestamps ──
    published_at: Optional[str] = None
    scraped_at: str = ""
    preprocessed_at: Optional[str] = None

    # ── Scrape Lineage ──
    scrape_batch_id: str = ""
    source_metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate required fields, normalize enums, set defaults."""

        # ── 1. Required field validation ──
        required_fields = {
            "document_id": self.document_id,
            "source_platform": self.source_platform,
            "source_url": self.source_url,
            "content_hash": self.content_hash,
            "title": self.title,
            "content": self.content,
            "company": self.company,
            "role": self.role,
        }
        missing = [
            name for name, value in required_fields.items()
            if not value or not str(value).strip()
        ]
        if missing:
            raise ValueError(
                f"Required field(s) cannot be None or empty: {missing}."
            )

        # ── 2. Normalize platform (alias → canonical) then validate ──
        object.__setattr__(
            self, "source_platform",
            self._normalize_platform(self.source_platform),
        )
        if self.source_platform not in VALID_PLATFORMS:
            raise ValueError(
                f"Invalid source_platform: '{self.source_platform}'. "
                f"Must be one of {VALID_PLATFORMS}"
            )

        # ── 3. Safe-enum normalization (invalid → default) ──
        object.__setattr__(
            self, "experience_level",
            self._safe_enum(self.experience_level, VALID_EXPERIENCE, None),
        )
        object.__setattr__(
            self, "interview_outcome",
            self._safe_enum(self.interview_outcome, VALID_OUTCOMES, None),
        )
        object.__setattr__(
            self, "difficulty",
            self._safe_enum(self.difficulty, VALID_DIFFICULTY, None),
        )

        # ── 4. Filter interview_types to valid values only ──
        object.__setattr__(
            self, "interview_types",
            [
                t for val in self.interview_types
                if (t := self._safe_enum(val, VALID_INTERVIEW_TYPES, None)) is not None
            ],
        )

        # ── 5. Auto-set preprocessed_at if not provided ──
        if self.preprocessed_at is None:
            object.__setattr__(
                self, "preprocessed_at",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )

    # ── Helper methods ──

    @staticmethod
    def _safe_enum(value, valid_set: Set[str], default):
        """Normalize enum value: lowercase + strip, fallback to default."""
        if value is None:
            return default
        cleaned = str(value).lower().strip()
        return cleaned if cleaned in valid_set else default

    @staticmethod
    def _normalize_platform(raw: str) -> str:
        """Map platform aliases to canonical names."""
        return PLATFORM_NORMALIZE.get(raw.lower().strip(), raw.lower().strip())

    # ── Serialization ──

    def to_dict(self) -> Dict:
        """Convert to dictionary suitable for DB insertion."""
        data = asdict(self)
        if not data["preprocessed_at"]:
            data["preprocessed_at"] = self.now_iso()
        return data

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict) -> "ProcessedInterviewDocument":
        """Create instance from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ProcessedInterviewDocument":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    @staticmethod
    def now_iso() -> str:
        """Return current UTC timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")