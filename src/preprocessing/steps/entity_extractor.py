"""
Step 5: Entity Extraction

Extracts structured fields from cleaned interview content using
regex patterns and curated lookup dictionaries. No ML/NER models —
purely deterministic, fast, and predictable.

Extracts:
    1. company        — Normalized company name (MSFT → Microsoft)
    2. role           — Normalized job role (SDE → Software Development Engineer)
    3. job_level      — Intern / Entry / Mid / Senior / Staff / Leadership
    4. interview_type — Phone Screen / Onsite / OA / Virtual / Walk-in
    5. num_rounds     — Count of interview rounds mentioned
    6. topics         — DSA, System Design, Behavioral, ML, SQL, etc.
    7. outcome        — Offer / Reject / Pending / Unknown
    8. difficulty     — Easy / Medium / Hard / Unknown

All patterns and dictionaries loaded from:
    src/preprocessing/resources/entity_extraction.yaml

Input:  doc dict with "cleaned_content" and "title" (from Steps 1-4)
Output: Same dict enriched with all extracted fields.
        This step never filters — every document passes through.
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Set

import yaml
import spacy

from src.preprocessing.steps.base import PreprocessingStep


# ── Resource loading ──

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
CONFIG_FILE = RESOURCES_DIR / "entity_extraction.yaml"


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Entity extraction config not found: {CONFIG_FILE}"
        )
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_alias_map(aliases: dict) -> Dict[str, str]:
    """
    Build a lowercased lookup from alias → canonical name.

    Input YAML format:
        Microsoft: ["MSFT", "microsoft corp", "microsoft corporation"]

    Output:
        {"msft": "Microsoft", "microsoft corp": "Microsoft", ...}
    """
    lookup = {}
    for canonical, alias_list in aliases.items():
        # Map canonical name to itself (lowercased key)
        lookup[canonical.lower()] = canonical
        for alias in alias_list:
            lookup[alias.lower()] = canonical
    return lookup


def _build_keyword_patterns(
    keywords: Dict[str, List[str]],
) -> Dict[str, re.Pattern]:
    """
    Build a single regex per category from keyword lists.

    Input: {"dsa": ["dsa", "data structure", "algorithm", ...]}
    Output: {"dsa": re.compile(r"\b(?:dsa|data structure|algorithm|...)\b", ...)}
    """
    patterns = {}
    for category, words in keywords.items():
        # Sort by length descending so longer phrases match first
        sorted_words = sorted(words, key=len, reverse=True)
        escaped = [re.escape(w) for w in sorted_words]
        pattern_str = r"\b(?:" + "|".join(escaped) + r")\b"
        patterns[category] = re.compile(pattern_str, re.IGNORECASE)
    return patterns


# Roman numeral → digit conversion
ROMAN_MAP = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
}

# ── Load config once ──
_CONFIG = _load_config()

COMPANY_ALIASES = _build_alias_map(_CONFIG.get("company_aliases", {}))
ROLE_ALIASES = _build_alias_map(_CONFIG.get("role_aliases", {}))
TOPIC_PATTERNS = _build_keyword_patterns(_CONFIG.get("topic_keywords", {}))

# Pre-compile regex for round detection
ROUND_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in _CONFIG.get("round_patterns", [])
]

# Job level keywords: {"intern": ["intern", "internship", ...], ...}
LEVEL_KEYWORDS = _CONFIG.get("level_keywords", {})

# Interview type keywords
TYPE_KEYWORDS = _CONFIG.get("interview_type_keywords", {})

# Outcome keywords
OUTCOME_KEYWORDS = _CONFIG.get("outcome_keywords", {})

# Difficulty keywords
DIFFICULTY_KEYWORDS = _CONFIG.get("difficulty_keywords", {})

# Company extraction config
COMPANY_CONFIG = _CONFIG.get("company_extraction", {})

# Title regex patterns for company extraction
TITLE_COMPANY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in COMPANY_CONFIG.get("title_patterns", [])
]

# NER config for company extraction
COMPANY_NER_CONFIG = COMPANY_CONFIG.get("ner", {})

# Words that spaCy ORG NER commonly misclassifies as companies
COMPANY_NER_BLOCKLIST: Set[str] = set(
    w.lower() for w in COMPANY_NER_CONFIG.get("blocklist", [])
)


class EntityExtractor(PreprocessingStep):
    """
    Extracts structured entities from interview content.

    Company extraction uses a 3-tier approach:
        1. Title regex patterns (highest confidence)
        2. Alias dictionary lookup (known companies)
        3. spaCy ORG NER on title + first 500 chars (catch-all)

    This step never filters — every document passes through,
    enriched with whatever entities could be extracted.
    Fields that couldn't be extracted are set to None or "unknown".
    """

    def __init__(
        self,
        spacy_model: str = None,
        ner_enabled: bool = True,
    ):
        self._spacy_model_name = (
            spacy_model
            or COMPANY_NER_CONFIG.get("model", "en_core_web_sm")
        )
        self._ner_enabled = ner_enabled
        self._nlp = None
        super().__init__()

    def _get_nlp(self):
        """Lazy-load spaCy model on first use. Only NER pipe enabled."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(
                    self._spacy_model_name,
                    disable=["parser", "lemmatizer", "tagger"],
                )
                self.logger.info(
                    f"Loaded spaCy model: {self._spacy_model_name}"
                )
            except OSError:
                self.logger.warning(
                    f"spaCy model '{self._spacy_model_name}' not found. "
                    f"NER-based company extraction disabled."
                )
                self._ner_enabled = False
        return self._nlp
    @property
    def name(self) -> str:
        return "entity_extractor"

    def process(self, doc: dict) -> Optional[dict]:
        title = doc.get("preprocessing", "").get("pii_remover","").get("title","")
        content = doc.get("preprocessing", "").get("pii_remover","").get("content","")

        if "entity_extractor" not in doc["preprocessing"]:
            doc["preprocessing"]["entity_extractor"] = {}

        # Combine title + content for extraction.
        # Title is checked first for company/role as it's
        # the most reliable source (e.g., "Google SDE Interview")
        combined = f"{title}\n{content}"
        combined_lower = combined.lower()

        # ── 1. Company ──
        doc["preprocessing"]["entity_extractor"]["company"] = self._extract_company(title, content)

        # ── 2. Role ──
        doc["preprocessing"]["entity_extractor"]["role"] = self._extract_role(title, content)

        # ── 3. Job Level ──
        doc["preprocessing"]["entity_extractor"]["experience_level"] = self._extract_level(combined_lower)

        # ── 4. Interview Type ──
        doc["preprocessing"]["entity_extractor"]["interview_types"] = self._extract_interview_types(combined_lower)

        # ── 5. Number of Rounds ──
        doc["preprocessing"]["entity_extractor"]["num_rounds"] = self._extract_num_rounds(combined)

        # ── 6. Topics ──
        doc["preprocessing"]["entity_extractor"]["topics"] = self._extract_topics(combined_lower)

        # ── 7. Outcome ──
        doc["preprocessing"]["entity_extractor"]["interview_outcome"] = self._extract_outcome(combined_lower)

        # ── 8. Difficulty ──
        doc["preprocessing"]["entity_extractor"]["difficulty"] = self._extract_difficulty(combined_lower)

        doc["preprocessing"]["entity_extractor"]["completed"] = True

        return doc

    # ── 1. Company Extraction (3-tier) ──

    def _extract_company(self, title: str, content: str) -> Optional[str]:
        """
        Extract and normalize company name using 3-tier approach:

            Tier 1: Title regex patterns
                Titles like "Google Interview Experience for SDE" or
                "My Interview at IBM" follow predictable formats.
                Highest confidence — extracts whatever sits in the
                company slot regardless of any dictionary.

            Tier 2: Alias dictionary lookup
                Scan title → first 500 chars → full content for
                known company names/aliases. Normalizes shortforms
                (MSFT → Microsoft). Handles known companies reliably.

            Tier 3: spaCy ORG NER
                Run ONLY on title + first 500 chars to catch companies
                not in the dictionary. Only ORG entities are extracted
                (never PERSON). Blocklist filters out false positives.

        The extracted name is always normalized through the alias
        dictionary if a match exists, so "MSFT" from any tier
        becomes "Microsoft".
        """
        # ── Tier 1: Title regex patterns ──
        company = self._extract_company_from_title(title)
        if company:
            # print("Company extracted from title")
            return self._normalize_company(company)

        # ── Tier 2: Alias dictionary scan ──
        for text in [title, content[:500], content]:
            company = self._find_company_in_text(text)
            if company:
                # print("Company extracted from Alias dictionary scan")
                return company  # Already normalized by alias lookup

        # ── Tier 3: spaCy ORG NER (title + first 500 chars only) ──
        if self._ner_enabled:
            company = self._extract_company_via_ner(title, content[:500])
            if company:
                # print("Company extracted from spaCy ORG NER")
                return self._normalize_company(company)

        return None

    def _extract_company_from_title(self, title: str) -> Optional[str]:
        """
        Tier 1: Extract company name from title using regex patterns.

        Patterns match common interview title formats:
            - "[Company] Interview Experience for [Role]"
            - "Interview Experience at [Company]"
            - "My [Company] Interview"

        Returns raw extracted name (not yet normalized).
        """
        for pattern in TITLE_COMPANY_PATTERNS:
            match = pattern.search(title)
            if match:
                company = match.group("company").strip()
                # Basic validation — reject if too short or too long
                if 2 <= len(company) <= 50:
                    return company
        return None

    def _find_company_in_text(self, text: str) -> Optional[str]:
        """
        Tier 2: Scan text for any known company alias.
        Returns canonical (normalized) company name or None.
        """
        text_lower = text.lower()
        for alias in sorted(COMPANY_ALIASES.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                return COMPANY_ALIASES[alias]
        return None

    def _extract_company_via_ner(
        self, title: str, content_head: str
    ) -> Optional[str]:
        """
        Tier 3: Use spaCy NER to extract ORG entities.

        Only runs on title + first 500 chars to limit noise.
        Only considers ORG label — never PERSON, GPE, etc.
        Applies blocklist to filter common false positives.

        Returns the first valid ORG entity found (title entities
        are prioritized by processing title first).
        """
        nlp = self._get_nlp()
        if nlp is None:
            return None

        # Process title first (higher signal), then content head
        for text in [title, content_head]:
            if not text.strip():
                continue
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ != "ORG":
                    continue

                name = ent.text.strip()

                # Skip very short or very long entities
                if len(name) < 2 or len(name) > 50:
                    continue

                # Skip blocklisted terms
                if name.lower() in COMPANY_NER_BLOCKLIST:
                    continue

                return name

        return None

    def _normalize_company(self, raw_name: str) -> str:
        """
        Normalize a raw company name through the alias dictionary.

        If the raw name (or its lowercase form) exists as an alias,
        return the canonical form. Otherwise return the raw name
        with cleaned-up casing (title case).
        """
        canonical = COMPANY_ALIASES.get(raw_name.lower())
        if canonical:
            return canonical
        # Not in dictionary — return with title casing
        return raw_name.strip().title()

    # ── 2. Role Extraction ──

    def _extract_role(self, title: str, content: str) -> Optional[str]:
        """
        Extract and normalize job role.

        Same priority as company: title → first paragraph → full scan.
        Also converts roman numerals (SDE II → SDE 2).
        """
        for text in [title, content[:500], content]:
            match = self._find_role_in_text(text)
            if match:
                return self._normalize_roman_numerals(match)
        return None

    def _find_role_in_text(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for alias in sorted(ROLE_ALIASES.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                return ROLE_ALIASES[alias]
        return None

    def _normalize_roman_numerals(self, role: str) -> str:
        """Convert roman numerals in role to digits: SDE II → SDE 2."""
        def replace_roman(match):
            roman = match.group(0).lower()
            return ROMAN_MAP.get(roman, match.group(0))

        # Match standalone roman numerals (I, II, III, IV, V, etc.)
        return re.sub(
            r"\b(I{1,3}|IV|VI{0,3}|IX|X)\b",
            replace_roman,
            role,
        )

    # ── 3. Job Level ──

    def _extract_level(self, text: str) -> str:
        """
        Detect job level/seniority from text.

        Returns first match by priority order defined in YAML.
        Order matters — "senior" should be checked before generic
        terms to avoid "senior" content matching "entry".
        """
        for level, keywords in LEVEL_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return level
        return "unknown"

    # ── 4. Interview Type ──

    def _extract_interview_types(self, text: str) -> List[str]:
        """
        Extract all interview types mentioned.

        Returns a list since a single experience can mention
        multiple types (e.g., OA + phone screen + onsite).
        """
        found = []
        for itype, keywords in TYPE_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    found.append(itype)
                    break  # One match per type is enough
        return found if found else ["unknown"]

    # ── 5. Number of Rounds ──

    def _extract_num_rounds(self, text: str) -> Optional[int]:
        """
        Count interview rounds mentioned in text.

        Two strategies:
            1. Explicit mentions: "3 rounds", "4 rounds of interview"
            2. Round headers: "Round 1", "Round 2", "Technical Round",
               "HR Round" — count unique round references
        Uses whichever gives the higher count.
        """
        # Strategy 1: Explicit round count mentions
        explicit = re.findall(
            r"(\d+)\s*(?:rounds?|stages?)\s*(?:of)?\s*(?:interview)?",
            text,
            re.IGNORECASE,
        )
        explicit_count = max((int(n) for n in explicit), default=0)

        # Strategy 2: Count unique round headers/markers
        round_markers = set()
        for pattern in ROUND_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                # Normalize to avoid counting "Round 1" and "round 1" separately
                normalized = m.lower().strip() if isinstance(m, str) else str(m).lower().strip()
                round_markers.add(normalized)

        marker_count = len(round_markers)

        best = max(explicit_count, marker_count)
        return best if best > 0 else None

    # ── 6. Topics ──

    def _extract_topics(self, text: str) -> List[str]:
        """
        Identify question topics discussed in the interview.

        Uses pre-compiled regex per topic category.
        A topic is included if ANY of its keywords appear.
        """
        found = []
        for topic, pattern in TOPIC_PATTERNS.items():
            if pattern.search(text):
                found.append(topic)
        return sorted(found) if found else []

    # ── 7. Outcome ──

    def _extract_outcome(self, text: str) -> str:
        """
        Detect interview outcome.

        Checks last 30% of text first — outcome is usually
        mentioned near the end ("I got the offer", "finally selected").
        Falls back to full text scan.
        """
        # Check tail first (where outcomes are usually stated)
        tail_start = int(len(text) * 0.7)
        tail = text[tail_start:]

        for outcome, keywords in OUTCOME_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", tail):
                    return outcome

        # Fallback: full text
        for outcome, keywords in OUTCOME_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return outcome

        return "unknown"

    # ── 8. Difficulty ──

    def _extract_difficulty(self, text: str) -> str:
        """
        Estimate interview difficulty from language used.

        Looks for explicit difficulty mentions ("it was tough",
        "easy interview", "very challenging questions").
        """
        for level, keywords in DIFFICULTY_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return level
        return "unknown"

# if __name__ == "__main__":
#     from preprocessing.steps.content_normalizer import ContentNormalizer
#     content_normalizer = ContentNormalizer()
#     from preprocessing.steps.pii_remover import PIIRemover
#     pii_remover = PIIRemover()
#     from preprocessing.steps.quality_filter import QualityFilter
#     quality_filter = QualityFilter()
#
#     doc = {
#       "document_id": "leetcode_d52d8899feb5712ed525ac1cfe01e6257e104ba0bb343bfa59b27ba8d98f5e2a",
#       "source_platform": "leetcode",
#       "source_url": "https://leetcode.com/discuss/post/7571535/cleartrip-flipkart-sde3-interview-experi-yjlk/",
#       "title": "ClearTrip (Flipkart) SDE3 Interview Experience",
#       "raw_content": "Hi everyone,\nI have studied a lot of interview experiences here and after appearing for multiple companies I have literally come to appreciate what this community is. I just want to give back.\nSo recently I got selected for the SDE3 role at ClearTrip (Flipkart) and I want to share my interview experience.\n- There were in total 5 rounds, 1 online test and then 4 rounds on-site in a single day, all 4 rounds were elimination rounds. I got a call through recruiter, I have found myself to be particularly lucky that I get calls as my resume gets shortlisted. I try to be active on LinkedIn, Naukri and InstaHyre, I try to apply as many jobs as possible and also added OpenToWork on my LinkedIn profile, was actively replying to messages on LinkedIn from recruiters and also commenting on job related posts\n- **ROUND 1 - DSA online test (1 hr),** I don't remember the exact questions but they were easy to average medium difficulty, for someone preparing DSA for interviews, this test should not be a problem.\n- **ON-SITE LOOP 1 - Machine coding round,** we were all given the same question and had to present our solution afterwards to an interviewer. We were asked ClearFit Demo application question which has already been given in full by other interview experiences, you have around 1:30-2:00 hrs to complete and then 1:00 hr to explain and answer questions on the same.\n- **ON-SITE LOOP 2 - DSA round (1 hr),** I was asked 3 questions, 2 questions related to monotonic stack (most famous questions on this topic) and 1 question similar to koko banana or flower bouquet problem mixed with some arithmetic\n- **ON-SITE LOOP 3 - HLD (1 hr),** I was asked to design a metrics capture and visualization tool, something like prometheus and grafana. Focus was on streaming millions of data points and how to handle such big load. Also went into discussions around the choice of databases both OLAP and OLTP and some depth of OLAP. Also discussed on how we would want to reduce latency in our visualization dashboards. You can talk about data validity, PII, and tiered storage as well.\n- **ON-SITE LOOP 4 - HM (Behavioral) (1 hr),** It can be mostly anything from technical to managerial to techno-managerial, mine was mostly technical and 10 mins of managerial discussion, main focus was on the most big project and why do you want to join CT and why did you leave your last company.\n\nThe process took a complete day and was on a Saturday. Hope this post helps someone, like others have helped me.",
#       "published_at": "2026-02-11T15:02:06.667317+00:00",
#       "scraped_at": "2026-02-11T15:30:00Z",
#       "scrape_type": "bulk",
#       "scrape_batch_id": "2026-02-11_bulk",
#       "source_metadata": {
#         "topic_id": "7571535",
#         "slug": "cleartrip-flipkart-sde3-interview-experi-yjlk",
#         "tags": [
#           "Interview"
#         ],
#         "tag_slugs": [
#           "interview"
#         ],
#         "company": "Flipkart",
#         "author_username": "yash_10",
#         "hit_count": 27,
#         "comment_count": 0,
#         "reactions": {
#           "UPVOTE": 1
#         },
#         "created_at": "2026-02-11T15:02:06.667317+00:00",
#         "updated_at": "2026-02-11T15:04:46.141895+00:00",
#         "comments": []
#       },
#       "content_hash": "e0ef84b94941d43ace566dd5add8e9c127563fb07cab4b54858dd10a536a0250"
#     }
#
#
#     doc = content_normalizer.process(doc)
#     doc = pii_remover.process(doc)
#     doc = quality_filter.process(doc)
#
#     # from preprocessing.steps.deduplicator import Deduplicator
#     # # At pipeline start — load existing hashes from DB
#     # # existing = {row.content_hash for row in db.query("SELECT content_hash_exact FROM processed_documents")}
#     # existing = {"e9fbc11657ac125b1dcb6f2bb5dda80ffb894dafd944d07be8f32d8a84cf91ba"}
#     # dedup = Deduplicator(
#     #     existing_hashes=existing,
#     #     # state_dir="/tmp/pipeline_state"
#     # )
#     # # After batch
#     # surviving_docs = dedup.process(doc)
#     # dedup.save_state()
#
#     from preprocessing.steps.entity_extractor import EntityExtractor
#     entity_extractor = EntityExtractor()
#     doc = entity_extractor.process(doc)
#     print(doc)