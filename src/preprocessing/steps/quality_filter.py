"""
Step 3: Quality Filtering

Drops documents that are too short, non-English, or consist entirely
of boilerplate/error content that slipped through scraping and
normalization.

Five quality checks (all configurable via quality_filter.yaml):
    1. Minimum word count — reject docs below threshold
    2. Maximum word count — reject abnormally long docs (likely scrape errors)
    3. Language detection — reject non-English content
    4. Boilerplate/error page detection — reject known junk page patterns
    5. Content signal check — reject docs with too little interview-relevant content

Input:  doc dict with "cleaned_content" and "word_count" (from Step 1)
Output: Same doc dict if it passes all checks, None if filtered.
        Adds "quality_score" (0.0 - 1.0) for downstream use.
"""

import re
from pathlib import Path
from typing import Optional, List, Set

import yaml
from langdetect import detect, DetectorFactory, LangDetectException

from src.preprocessing.steps.base import PreprocessingStep


# ── Make langdetect deterministic ──
DetectorFactory.seed = 0

# ── Resource loading ──

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
CONFIG_FILE = RESOURCES_DIR / "quality_filters.yaml"


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Quality filter config not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG = _load_config()


class QualityFilter(PreprocessingStep):
    """
    Filters out low-quality documents based on configurable thresholds.

    This step only filters — it never modifies document content.
    Documents that pass all checks get a quality_score added.

    All thresholds and keyword lists are loaded from quality_filter.yaml.
    Constructor args override YAML defaults when provided.
    """

    def __init__(
        self,
        min_word_count: int = None,
        max_word_count: int = None,
        min_signal_ratio: float = None,
        allowed_languages: List[str] = None,
    ):
        thresholds = _CONFIG.get("thresholds", {})
        lang_config = _CONFIG.get("language", {})

        self.min_word_count = (
            min_word_count
            if min_word_count is not None
            else thresholds.get("min_word_count", 50)
        )
        self.max_word_count = (
            max_word_count
            if max_word_count is not None
            else thresholds.get("max_word_count", 15000)
        )
        self.min_signal_ratio = (
            min_signal_ratio
            if min_signal_ratio is not None
            else thresholds.get("min_signal_ratio", 0.02)
        )
        self.allowed_languages = (
            allowed_languages
            if allowed_languages is not None
            else lang_config.get("allowed", ["en"])
        )
        self.min_lang_detect_chars = lang_config.get("min_chars_for_detection", 50)

        # Load keyword sets from config
        self._signal_keywords: Set[str] = set(
            w.lower() for w in _CONFIG.get("signal_keywords", [])
        )
        self._error_patterns: List[re.Pattern] = [
            re.compile(p["pattern"], re.IGNORECASE)
            for p in _CONFIG.get("error_page_patterns", [])
        ]

        super().__init__()

    @property
    def name(self) -> str:
        return "quality_filter"

    def process(self, doc: dict) -> Optional[dict]:
        """
        Run all quality checks. Returns None on first failure.

        Check order is cheapest-first to fail fast:
            1. Word count (instant)
            2. Error page detection (regex scan)
            3. Language detection (statistical model)
            4. Content signal ratio (keyword scan)
        """
        cleaned = doc.get("preprocessing", "").get("pii_remover","").get("content","")
        word_count = doc.get("preprocessing",{}).get("content_normalizer", {}).get("word_count", 0)

        # ── Check 1: Word count bounds ──
        if word_count < self.min_word_count:
            doc["_filter_reason"] = f"too_short_{word_count}_words"
            return None

        if word_count > self.max_word_count:
            doc["_filter_reason"] = f"too_long_{word_count}_words"
            return None

        # ── Check 2: Error / junk page detection ──
        if self._is_error_page(cleaned):
            doc["_filter_reason"] = "error_or_junk_page"
            return None

        # ── Check 3: Language detection ──
        if not self._is_allowed_language(cleaned):
            doc["_filter_reason"] = "non_english"
            return None

        # ── Check 4: Interview content signal ──
        signal_ratio = self._compute_signal_ratio(cleaned)
        if signal_ratio < self.min_signal_ratio:
            doc["_filter_reason"] = "low_interview_signal"
            return None

        if "quality_filter" not in doc["preprocessing"]:
            doc["preprocessing"]["quality_filter"] = {}

        # ── All checks passed — compute quality score ──
        doc["preprocessing"]["quality_filter"]["quality_score"] = self._compute_quality_score(
            word_count, signal_ratio
        )

        doc["preprocessing"]["quality_filter"]["completed"] = True

        return doc

    def _is_error_page(self, text: str) -> bool:
        """
        Check if text matches known error/junk page patterns.

        Catches pages like:
            - "404 Not Found"
            - "Access Denied"
            - "Page has been removed"
            - Login walls that slipped through
        """
        # Check first 500 chars only — error messages are at the top
        head = text[:500]
        for pattern in self._error_patterns:
            if pattern.search(head):
                return True
        return False

    def _is_allowed_language(self, text: str) -> bool:
        """
        Detect language and check against allowed list.

        Uses langdetect library. Short texts (< min_chars_for_detection)
        skip detection and are assumed English — langdetect is unreliable
        on short strings and interview content often has mixed-language
        technical terms that confuse detection.
        """
        if len(text) < self.min_lang_detect_chars:
            return True

        try:
            # Use first 1000 chars for speed — enough for reliable detection
            detected = detect(text[:1000])
            return detected in self.allowed_languages
        except LangDetectException:
            # If detection fails, let the doc through — don't drop
            # content because of a detection error
            self.logger.debug("Language detection failed, allowing doc through")
            return True

    def _compute_signal_ratio(self, text: str) -> float:
        """
        Compute what fraction of words are interview-relevant signal.

        A document about interview experiences should contain keywords
        like "interview", "round", "asked", "coding", "offer", etc.
        Documents with near-zero signal ratio are likely off-topic
        content that matched scraper URLs but aren't interview experiences.

        Returns float between 0.0 and 1.0.
        """
        words = text.lower().split()
        if not words:
            return 0.0

        signal_count = sum(1 for w in words if w in self._signal_keywords)
        return signal_count / len(words)

    def _compute_quality_score(
        self, word_count: int, signal_ratio: float
    ) -> float:
        """
        Compute a composite quality score (0.0 - 1.0) for the document.

        Factors:
            - Length score: prefers medium-length docs (200-2000 words)
            - Signal score: higher interview keyword density = better

        This score is NOT used for filtering (thresholds handle that).
        It's attached to the doc for downstream ranking/prioritization.
        """
        # Length score: ramp up to 1.0 between min and 200 words,
        # stay at 1.0 up to 2000, then gradually decay
        if word_count < 200:
            length_score = word_count / 200.0
        elif word_count <= 2000:
            length_score = 1.0
        else:
            length_score = max(0.5, 1.0 - (word_count - 2000) / 10000.0)

        # Signal score: cap at 1.0, scale up from threshold
        signal_score = min(1.0, signal_ratio / 0.10)

        # Weighted combination
        return round(0.5 * length_score + 0.5 * signal_score, 3)

# if __name__ == "__main__":
#     from preprocessing.steps.content_normalizer import ContentNormalizer
#     content_normalizer = ContentNormalizer()
#     from preprocessing.steps.pii_remover import PIIRemover
#     pii_remover = PIIRemover()
#     from preprocessing.steps.quality_filter import QualityFilter
#     quality_filter = QualityFilter()
#
#     doc = {
#       "document_id": "medium_6c84cb815d8654cde2cb531ba87eafaec9fb6328ef227da3af9eca6d59f3efdd",
#       "source_platform": "medium",
#       "source_url": "https://medium.com/@sumitpardhiya/john-deere-my-interview-experience-ccb2e763ffe5",
#       "title": "🚜 John Deere — My Interview Experience",
#       "raw_content": "### 🚜 John Deere — My Interview Experience\n\nRecently, I had the opportunity to interview with John Deere for the position of AI/ML Engineer. The process began when I received a Naukri invite from their hiring team. After I applied, a third-party recruiter contacted me, mentioning that he had gone through my profile and was quite impressed. He requested my updated resume to proceed further with the process.\n\nAfter about three days, he reached out again to inform me that my resume was shortlisted, and they wanted to schedule my first interview round.\n\n#### Preparation Phase (Or Lack of It!)\n\nSince it had been almost two years since my last interview, I wasn’t fully confident in my preparation. I told them that I’d need some time to brush up on my concepts. However, since they were moving fast, I was given just three days to prepare.\n\nUnfortunately, due to a small communication gap, I thought my interview was scheduled for the 31st, but it was actually on the 30th. So, when they contacted me that morning to confirm, I realized my mistake — and honestly, I wasn’t as prepared as I wanted to be.\n\n#### Round 1 — Technical Interview\n\nThe interview began with the usual introduction and questions related to my past projects. That part went quite smoothly — I was confident and able to explain my work clearly.\n\nNext, the interviewer moved on to Machine Learning and Deep Learning questions. The initial ones were quite manageable — topics like model evaluation, bias-variance, and regularization. But as the discussion went deeper into advanced ML concepts, I started struggling. I knew the topics but couldn’t recall some of the details properly because of my rushed preparation.\n\nAfter that, they asked a DSA (Data Structures and Algorithms) question — a relatively easy one — which I was able to solve quickly and correctly.\n\nThe round lasted about an hour in total.\n\n#### Result & Reflection\n\nTwo days later, I received a call informing me that I hadn’t cleared the round. Honestly, I wasn’t surprised — I knew I hadn’t given my 100%. But I took it positively because the questions they asked were very relevant and insightful, and similar ones started repeating in other interviews later.\n\nIt reminded me how important consistent revision and practice are, especially in a field that evolves as fast as AI/ML.\n\n### 💡 Final Thoughts\n\nEven though I didn’t make it through, the John Deere interview was a great learning experience. The process was smooth, the interviewers were professional and polite, and it helped me understand where I stood technically and what areas I needed to improve.\n\nIf I’d had just one more day of preparation, I genuinely believe I could have cracked it. But more importantly, it reminded me that every interview adds something valuable to your journey — whether it’s success or learning.",
#       "published_at": "2025-11-10T02:58:01.902Z",
#       "scraped_at": "2026-02-12T17:40:57Z",
#       "scrape_type": "bulk",
#       "scrape_batch_id": "2026-02-12_bulk",
#       "source_metadata": {
#         "description": "Recently, I had the opportunity to interview with John Deere for the position of AI/ML Engineer. The process began when I received a…",
#         "reading_time": "2 min read",
#         "tags": [
#           "Interview",
#           "Interview Questions",
#           "Interview Experience",
#           "Jobs",
#           "AI"
#         ],
#         "featured_image": "https://miro.medium.com/v2/da:true/bc1f8416df0cad099e43cda2872716e5864f18a73bda2a7547ea082aca9b5632",
#         "canonical_url": "https://medium.com/@sumitpardhiya/john-deere-my-interview-experience-ccb2e763ffe5"
#       },
#       "content_hash": "fe5824be821df04f830c866f368cf657a9c8b90d5c227609c25f9cf2d50f93be"
#     }
#     doc = content_normalizer.process(doc)
#     doc = pii_remover.process(doc)
#     doc = quality_filter.process(doc)
#     print(doc)