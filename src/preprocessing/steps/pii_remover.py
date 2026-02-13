"""
Step 2: PII Removal

Scrubs personally identifiable information from cleaned content.
Runs early so no downstream step (entity extraction, embedding,
DB storage) accidentally indexes or persists PII.

Two-pass approach:
    Pass 1 — Regex: Structured PII (emails, phones, personal URLs,
             ID numbers). Patterns loaded from pii_patterns.yaml.

Redaction replaces PII with typed placeholders:
    john.doe@gmail.com      → [EMAIL]
    +91-9876543210          → [PHONE]
    linkedin.com/in/johnd   → [PERSONAL_URL]
    "interviewer Priya"     → "interviewer [PERSON]"

Input:  doc dict with "cleaned_content" and "title" (from Step 1)
Output: Same dict with PII scrubbed from "cleaned_content" and "title".
        Adds "pii_removed": True and "pii_counts": {"email": 2, ...}
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set

import yaml

from src.preprocessing.steps.base import PreprocessingStep


# ── Resource loading ──

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
PII_CONFIG_FILE = RESOURCES_DIR / "pii_patterns.yaml"


def _load_pii_config() -> dict:
    """Load PII patterns YAML config."""
    if not PII_CONFIG_FILE.exists():
        raise FileNotFoundError(f"PII config not found: {PII_CONFIG_FILE}")
    with open(PII_CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compile_pii_patterns(
    entries: list,
) -> List[Tuple[re.Pattern, str, str]]:
    """
    Compile PII pattern entries from YAML.

    Returns:
        List of (compiled_pattern, placeholder, pii_type) tuples.
        e.g. (re.compile(...), "[EMAIL]", "email")
    """
    flag_map = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
    }
    compiled = []
    for entry in entries:
        flags = 0
        for f in entry.get("flags", []):
            flags |= flag_map.get(f.upper(), 0)
        compiled.append((
            re.compile(entry["pattern"], flags),
            entry["placeholder"],
            entry["type"],
        ))
    return compiled


# ── Load config and compile patterns once at module import ──

_PII_CONFIG = _load_pii_config()
PII_PATTERNS = _compile_pii_patterns(_PII_CONFIG.get("patterns", []))


class PIIRemover(PreprocessingStep):
    """
    Removes PII from cleaned_content and title.

    This step never filters documents — it only redacts.
    Every document passes through with PII replaced by placeholders.
    """

    def __init__(self,):
        super().__init__()

    @property
    def name(self) -> str:
        return "pii_remover"

    def process(self, doc: dict) -> Optional[dict]:
        """
        Scrub PII from cleaned_content and title.

        This step never returns None — all documents survive,
        just with PII redacted.
        """
        cleaned = doc.get("preprocessing", "").get("content_normalizer","").get("content","")
        title = doc.get("preprocessing", "").get("content_normalizer","").get("title","")

        if not cleaned:
            print("Cleaned Content not found. Please pass it through Content Normalizer first")
            return doc

        pii_counts: Dict[str, int] = {}

        # ── Pass 1: Regex-based redaction ──
        cleaned, regex_counts = self._regex_scrub(cleaned)
        title, _ = self._regex_scrub(title)
        _merge_counts(pii_counts, regex_counts)

        if "pii_remover" not in doc["preprocessing"]:
            doc["preprocessing"]["pii_remover"] = {}

        doc["preprocessing"]["pii_remover"]["content"] = cleaned
        doc["preprocessing"]["pii_remover"]["title"] = title
        doc["preprocessing"]["pii_remover"]["pii_counts"] = pii_counts
        if pii_counts:
            self.logger.debug(
                f"Doc '{doc.get('document_id', 'unknown')}' — "
                f"PII found: {pii_counts}"
            )

        doc["preprocessing"]["pii_remover"]["completed"] = True

        return doc

    def _regex_scrub(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Apply all regex PII patterns to text.

        Returns (scrubbed_text, counts_by_type).
        """
        counts: Dict[str, int] = {}

        for pattern, placeholder, pii_type in PII_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                counts[pii_type] = counts.get(pii_type, 0) + len(matches)
                text = pattern.sub(placeholder, text)

        return text, counts


def _merge_counts(target: Dict[str, int], source: Dict[str, int]) -> None:
    """Merge source counts into target dict in place."""
    for key, val in source.items():
        target[key] = target.get(key, 0) + val

# if __name__ == "__main__":
#     from preprocessing.steps.content_normalizer import ContentNormalizer
#     content_normalizer = ContentNormalizer()
#     from preprocessing.steps.pii_remover import PIIRemover
#     pii_remover = PIIRemover()
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
#     print(doc)