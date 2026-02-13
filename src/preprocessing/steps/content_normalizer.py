"""
Step 1: Content Normalization

Cleans raw scraped content into plain, uniform text.


Pipeline order:
    1. Encoding fixes (HTML entities, mojibake, special chars)
    2. HTML tag stripping (BeautifulSoup)
    3. Markdown artifact removal
    4. Boilerplate removal (generic + platform-specific)
    5. Emoji removal
    6. Unicode normalization (NFC)
    7. Whitespace standardization

All configurable regex patterns are loaded from:
    src/preprocessing/resources/normalization_patterns.yaml

Input:  doc dict from ScrapedInterviewDocument.to_dict()
Output: Same dict with added "cleaned_content" and "word_count"
        Original "raw_content" is preserved for lineage.
"""

import re
import unicodedata
import html as html_lib
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import yaml
from bs4 import BeautifulSoup, Comment

from src.preprocessing.steps.base import PreprocessingStep


# ── Resource loading ──

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
CONFIG_FILE = RESOURCES_DIR / "normalization_patterns.yaml"


def _load_config() -> dict:
    """Load the normalization YAML config once."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Normalization config not found: {CONFIG_FILE}"
        )
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compile_patterns(entries: list) -> List[re.Pattern]:
    """Compile a list of pattern entries from YAML into re.Pattern objects."""
    flag_map = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }
    compiled = []
    for entry in entries:
        flags = 0
        for f in entry.get("flags", []):
            flags |= flag_map.get(f.upper(), 0)
        compiled.append(re.compile(entry["pattern"], flags))
    return compiled


def _compile_replacement_patterns(
    entries: list,
) -> List[Tuple[re.Pattern, str]]:
    """Compile pattern entries that include replacement strings."""
    flag_map = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }
    compiled = []
    for entry in entries:
        flags = 0
        for f in entry.get("flags", []):
            flags |= flag_map.get(f.upper(), 0)
        compiled.append(
            (re.compile(entry["pattern"], flags), entry.get("replacement", ""))
        )
    return compiled


# ── Load and compile all patterns once at module import ──

_CONFIG = _load_config()

MARKDOWN_PATTERNS = _compile_replacement_patterns(
    _CONFIG.get("markdown", [])
)
GENERIC_BOILERPLATE = _compile_patterns(
    _CONFIG.get("boilerplate_generic", [])
)

# Platform-specific boilerplate: { "gfg": [...], "medium": [...], ... }
PLATFORM_BOILERPLATE: Dict[str, List[re.Pattern]] = {}
for platform, entries in _CONFIG.get("boilerplate_platform", {}).items():
    PLATFORM_BOILERPLATE[platform] = _compile_patterns(entries)

# Platform alias mapping from config: { "gfg": "gfg", "geeksforgeeks": "gfg" }
PLATFORM_ALIASES: Dict[str, str] = _CONFIG.get("platform_aliases", {})

# Encoding replacements
ENCODING_REPLACEMENTS: Dict[str, str] = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\ufeff": "",
    "\x00": "",
}

# Emoji pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)


class ContentNormalizer(PreprocessingStep):
    """
    Single normalizer that handles all platforms.

    Platform-specific boilerplate is selected automatically
    from the YAML config based on doc["source_platform"].
    """

    def __init__(self, min_cleaned_length: int = 20):
        self.min_cleaned_length = min_cleaned_length
        super().__init__()

    @property
    def name(self) -> str:
        return "content_normalizer"

    def process(self, doc: dict) -> Optional[dict]:
        raw = doc.get("raw_content")
        if not raw or not raw.strip():
            doc["_filter_reason"] = "empty_raw_content"
            return None

        platform = self._resolve_platform(doc.get("source_platform", ""))
        cleaned = self._normalize(raw, platform)

        if len(cleaned) < self.min_cleaned_length:
            doc["_filter_reason"] = "empty_after_cleaning"
            return None

        if "preprocessing" not in doc:
            doc["preprocessing"] = {}
        if "content_normalizer" not in doc["preprocessing"]:
            doc["preprocessing"]["content_normalizer"] = {}

        doc["preprocessing"]["content_normalizer"]["content"] = cleaned
        doc["preprocessing"]["content_normalizer"]["title"] = self._clean_title(doc.get("title", ""))
        doc["preprocessing"]["content_normalizer"]["word_count"] = len(cleaned.split())

        doc["preprocessing"]["content_normalizer"]["completed"] = True

        return doc

    # ── Main pipeline ──

    def _normalize(self, text: str, platform: str) -> str:
        text = self._fix_encoding(text)
        text = self._strip_html(text)
        text = self._strip_markdown(text)
        text = self._remove_boilerplate(text, platform)
        text = self._remove_emojis(text)
        text = self._normalize_unicode(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def _clean_title(self, title: str) -> str:
        title = self._fix_encoding(title)
        title = self._strip_html(title)
        title = self._remove_emojis(title)
        title = self._normalize_unicode(title)
        return re.sub(r"\s+", " ", title).strip()

    # ── Individual stages ──

    def _resolve_platform(self, raw: str) -> str:
        return PLATFORM_ALIASES.get(raw.lower().strip(), raw.lower().strip())

    def _fix_encoding(self, text: str) -> str:
        text = html_lib.unescape(text)
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        for old, new in ENCODING_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text

    def _strip_html(self, text: str) -> str:
        if "<" not in text:
            return text
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup.find_all(
            ["script", "style", "nav", "footer", "header", "aside", "noscript"]
        ):
            tag.decompose()
        for comment in soup.find_all(
            string=lambda s: isinstance(s, Comment)
        ):
            comment.extract()
        for tag in soup.find_all(
            ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"]
        ):
            tag.insert_before("\n")
        return soup.get_text()

    def _strip_markdown(self, text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", " [code block removed] ", text)
        for pattern, replacement in MARKDOWN_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _remove_boilerplate(self, text: str, platform: str) -> str:
        # Platform-specific first (more targeted, avoids false positives)
        for pattern in PLATFORM_BOILERPLATE.get(platform, []):
            text = pattern.sub("", text)
        # Then generic
        for pattern in GENERIC_BOILERPLATE:
            text = pattern.sub("", text)
        return text

    def _remove_emojis(self, text: str) -> str:
        return EMOJI_PATTERN.sub("", text)

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n\s+\n", "\n\n", text)
        text = re.sub(r" +\n", "\n", text)
        return text

# if __name__ == "__main__":
#     normalizer = ContentNormalizer()
#     doc = {
#       "document_id": "gfg_65dfe7e09bc4ddb75c73fc86b3cde137b5e1410508c577b56299f0ca46633442",
#       "source_platform": "gfg",
#       "source_url": "https://www.geeksforgeeks.org/interview-experiences/unthinkable-solution-interview-experience-for-software-developer/",
#       "title": "Unthinkable Solution Interview Experience for Software Developer",
#       "raw_content": "Unthinkable Solution Interview Experience for Software Developer\nLast Updated :\n15 Sep, 2025\nCandidate Information\nStatus:\nFinal-year student, seeking full-time opportunities\nExperience:\nAcademic and internship projects (MERN stack, DSA practice)\nTarget Position:\nSoftware Engineer (Entry-level)\nLocation\n: Gurgaon, Haryana, India\nInterview Date:\n13 September 2025\nRound 1: Online Assessment (Eliminatory)\nDuration:\n90 minutes\nMethod:\nOn-site at Gurgaon (the company’s sandbox platform)\nFocus Areas:\nData Structures & Algorithms (Strings, Dynamic Programming, Backtracking).\nObstacles Faced\nThe company’s compiler (sandbox environment) repeatedly crashed/auto-refreshed, erasing progress.\nSupport staff only advised to “refresh again” instead of resolving the issue.\nI solved Q1 completely but could not make final submissions for Q2 and Q3 due to these platform problems.\nFurther Rounds (Managerial/Technical/HR)\nMentioned as part of the process, but I could not proceed beyond Round 1 because of its eliminatory structure.\nIn total, Unthinkable Solutions conducts 6 rounds (all eliminatory).\nPost-Interview Reflections\nCompany Culture Insights:\nThe process appeared to be rigid and process-driven, with less focus on creating a candidate-friendly experience.\nWork Environment:\nAll rounds were conducted onsite at Gurgaon, requiring self-travel, with no travel reimbursement or support provided.\nBenefits Highlight:\nPackage options: 5, 6, 8, and 10 LPA, depending on candidate performance.\nNo notable perks or benefits were discussed in early stages.\nEvaluator Feedback:\nSince the first round was purely technical and eliminatory, no direct feedback was shared.\nSuggestions for Improvement:\nEnsure stable and reliable coding platforms — frequent crashes affect fairness.\nProvide on-site technical support staff during coding tests.\nConsider whether LeetCode Hard questions in Round 1 are proportional to the package being offered.\nShare at least brief feedback with candidates to help them improve.\nAdditional Information\nTimeline: Applied through referral. Round 1 conducted within 2 weeks of application.\nNext Steps: Could not advance due to submission issues caused by the platform.\nTravel: Candidates were required to travel to Gurgaon at their own expense.\nClosing Note\nThe difficulty level of the first round was comparable to what top product-based companies (Amazon, Adobe, Microsoft, etc.) usually ask, but the candidate experience was disappointing due to poor platform reliability.\nIf you are preparing for Unthinkable Solutions, expect Dynamic Programming (Hard), Backtracking, and String + HashMap problems in the very first round itself, and be prepared to deal with potential technical glitches during the test.\nComment\nArticle Tags:\nArticle Tags:\nInterview Experiences\nUnthinkable Solutions\nExperiences-QnA",
#       "published_at": "15 Sep, 2025",
#       "scraped_at": "2026-02-10T16:44:14Z",
#       "scrape_type": "bulk",
#       "scrape_batch_id": "2026-02-10_bulk",
#       "source_metadata": {
#         "tags": [
#           "Interview Experiences",
#           "Unthinkable Solutions",
#           "Experiences-QnA"
#         ],
#         "experience_type": "unknown"
#       },
#       "content_hash": "3d71ec8ab7631251c215f7b8ac0bac1308fd4e85170fb30b5d914deadc79b3dc"
#     }
#     print("*"*50)
#     print(normalizer.process(doc))