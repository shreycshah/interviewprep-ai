# from dataclasses import dataclass, asdict, field
# from datetime import datetime, timezone
# from typing import Dict, List, Optional
# import json
#
# class ProcessedInterviewDocument:
#     """
#     Fully processed interview document ready for persistence
#     in the processed_documents table.
#
#     This represents post-normalization, post-PII removal,
#     and post-entity extraction data.
#     """
#
#     # ── Identity & Lineage ──
#     document_id: str
#     source_platform: str
#     source_url: str
#     content_hash: str
#
#     # ── Content ──
#     title: str
#     content: str
#     word_count: int
#
#     # ── Extracted Entities ──
#     company: Optional[str] = None
#     role: Optional[str] = None
#     experience_level: str = "unknown"
#
#     # ── Extracted Structure ──
#     interview_outcome: str = "unknown"
#     difficulty: str = "unknown"
#     num_rounds: Optional[int] = None
#     interview_types: List[str] = field(default_factory=list)
#
#     # ── Topics ──
#     topics: List[str] = field(default_factory=list)
#
#     # ── Timestamps ──
#     published_at: Optional[str] = None
#     scraped_at: str = ""
#     preprocessed_at: Optional[str] = None
#
#     # ── Scrape Lineage ──
#     scrape_batch_id: str = ""
#     source_metadata: Dict = field(default_factory=dict)
#
#     def to_dict(self) -> Dict:
#         """Convert to dictionary suitable for DB insertion."""
#         data = asdict(self)
#
#         # Ensure preprocessed_at default behavior matches DB
#         if not data["preprocessed_at"]:
#             data["preprocessed_at"] = self.now_iso()
#
#         return data
#
#     def to_json(self) -> str:
#         """Serialize to JSON."""
#         return json.dumps(self.to_dict(), ensure_ascii=False)
#
#     @classmethod
#     def from_dict(cls, data: Dict) -> "ProcessedInterviewDocument":
#         """Create instance from dictionary."""
#         return cls(**data)
#
#     @classmethod
#     def from_json(cls, json_str: str) -> "ProcessedInterviewDocument":
#         """Deserialize from JSON."""
#         return cls.from_dict(json.loads(json_str))
#
#     @staticmethod
#     def now_iso() -> str:
#         """Return current UTC timestamp in ISO 8601 format."""
#         return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# '''
# -- ============================================================
# -- CORE TABLE: Processed Interview Documents
# -- ============================================================
#
# CREATE TABLE processed_documents (
#     -- ── Identity & Lineage ──
#     document_id         TEXT PRIMARY KEY,           -- e.g. "geeksforgeeks_<sha256>"
#     source_platform     source_platform_enum NOT NULL,
#     source_url          TEXT NOT NULL UNIQUE,
#     content_hash        TEXT NOT NULL,              -- SHA256 of normalized content, for dedup
#
#     -- ── Content ──
#     title               TEXT NOT NULL,
#     cleaned_content     TEXT NOT NULL,              -- After normalization + PII scrubbing
#     word_count          INTEGER NOT NULL,
#
#     -- ── Extracted Entities ──
#     company             TEXT,                       -- Normalized name ("Microsoft", not "MSFT")
#     role                TEXT,                       -- e.g. "Software Engineer", "Data Scientist"
#     experience_level    experience_level_enum NOT NULL DEFAULT 'unknown',
#
#     -- ── Extracted Structure ──
#     interview_outcome   interview_outcome_enum NOT NULL DEFAULT 'unknown',
#     difficulty          difficulty_enum NOT NULL DEFAULT 'unknown',
#     num_rounds          INTEGER,                    -- Number of interview rounds mentioned
#     interview_types     TEXT[],                     -- e.g. {"phone_screen", "onsite", "oa"}
#
#     -- ── Topics ──
#     topics              TEXT[],                     -- e.g. {"dsa", "system_design", "behavioral"}
#
#     -- ── Timestamps ──
#     published_at        TIMESTAMPTZ,                -- Original publish date (nullable, not always available)
#     scraped_at          TIMESTAMPTZ NOT NULL,       -- When the scraper collected it
#     preprocessed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- When preprocessing completed
#
#     -- ── Scrape Lineage ──
#     scrape_batch_id     TEXT NOT NULL,
#     source_metadata     JSONB DEFAULT '{}',         -- Platform-specific fields (upvotes, tags, etc.)
# );
# '''