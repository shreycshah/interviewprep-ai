'''
-- ============================================================
-- CORE TABLE: Processed Interview Documents
-- ============================================================

CREATE TABLE processed_documents (
    -- ── Identity & Lineage ──
    document_id         TEXT PRIMARY KEY,           -- e.g. "geeksforgeeks_<sha256>"
    source_platform     source_platform_enum NOT NULL,
    source_url          TEXT NOT NULL UNIQUE,
    content_hash        TEXT NOT NULL,              -- SHA256 of normalized content, for dedup

    -- ── Content ──
    title               TEXT NOT NULL,
    cleaned_content     TEXT NOT NULL,              -- After normalization + PII scrubbing
    word_count          INTEGER NOT NULL,

    -- ── Extracted Entities ──
    company             TEXT,                       -- Normalized name ("Microsoft", not "MSFT")
    role                TEXT,                       -- e.g. "Software Engineer", "Data Scientist"
    experience_level    experience_level_enum NOT NULL DEFAULT 'unknown',
    location            TEXT,                       -- e.g. "Bangalore", "San Francisco"

    -- ── Extracted Structure ──
    interview_outcome   interview_outcome_enum NOT NULL DEFAULT 'unknown',
    difficulty          difficulty_enum NOT NULL DEFAULT 'unknown',
    num_rounds          INTEGER,                    -- Number of interview rounds mentioned
    interview_types     TEXT[],                     -- e.g. {"phone_screen", "onsite", "oa"}

    -- ── Topics ──
    topics              TEXT[],                     -- e.g. {"dsa", "system_design", "behavioral"}

    -- ── Timestamps ──
    published_at        TIMESTAMPTZ,                -- Original publish date (nullable, not always available)
    scraped_at          TIMESTAMPTZ NOT NULL,       -- When the scraper collected it
    processed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- When preprocessing completed

    -- ── Scrape Lineage ──
    scrape_batch_id     TEXT NOT NULL,
    source_metadata     JSONB DEFAULT '{}',         -- Platform-specific fields (upvotes, tags, etc.)

    -- ── Preprocessing Lineage ──
    preprocessing_version   TEXT NOT NULL,           -- e.g. "1.0.0" — tracks pipeline version
    extraction_confidence   JSONB DEFAULT '{}'       -- Per-field confidence scores from extraction
                                                     -- e.g. {"company": 0.95, "role": 0.72}
);
'''