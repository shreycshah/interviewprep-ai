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
        cleaned = doc.get("cleaned_content", "")
        title = doc.get("cleaned_title", "")

        if not cleaned:
            print("Cleaned Content not found. Please pass it through Content Normalizer first")
            return doc

        pii_counts: Dict[str, int] = {}

        # ── Pass 1: Regex-based redaction ──
        cleaned, regex_counts = self._regex_scrub(cleaned)
        title, _ = self._regex_scrub(title)
        _merge_counts(pii_counts, regex_counts)

        doc["cleaned_content"] = cleaned
        doc["cleaned_title"] = title
        doc["pii_removed"] = True
        doc["pii_counts"] = pii_counts

        if pii_counts:
            self.logger.debug(
                f"Doc '{doc.get('document_id', 'unknown')}' — "
                f"PII found: {pii_counts}"
            )

        doc["preprocessing_steps"]["2_pii_remover"] = True

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
#     pii_remover = PIIRemover()
#     doc = {'document_id': 'gfg_65dfe7e09bc4ddb75c73fc86b3cde137b5e1410508c577b56299f0ca46633442', 'source_platform': 'gfg', 'source_url': 'https://www.geeksforgeeks.org/interview-experiences/unthinkable-solution-interview-experience-for-software-developer/', 'title': 'Unthinkable Solution Interview Experience for Software Developer', 'raw_content': 'Unthinkable Solution Interview Experience for Software Developer\nLast Updated :\n15 Sep, 2025\nCandidate Information\nStatus:\nFinal-year student, seeking full-time opportunities\nExperience:\nAcademic and internship projects (MERN stack, DSA practice)\nTarget Position:\nSoftware Engineer (Entry-level)\nLocation\n: Gurgaon, Haryana, India\nInterview Date:\n13 September 2025\nRound 1: Online Assessment (Eliminatory)\nDuration:\n90 minutes\nMethod:\nOn-site at Gurgaon (the company’s sandbox platform)\nFocus Areas:\nData Structures & Algorithms (Strings, Dynamic Programming, Backtracking).\nObstacles Faced\nThe company’s compiler (sandbox environment) repeatedly crashed/auto-refreshed, erasing progress.\nSupport staff only advised to “refresh again” instead of resolving the issue.\nI solved Q1 completely but could not make final submissions for Q2 and Q3 due to these platform problems.\nFurther Rounds (Managerial/Technical/HR)\nMentioned as part of the process, but I could not proceed beyond Round 1 because of its eliminatory structure.\nIn total, Unthinkable Solutions conducts 6 rounds (all eliminatory).\nPost-Interview Reflections\nCompany Culture Insights:\nThe process appeared to be rigid and process-driven, with less focus on creating a candidate-friendly experience.\nWork Environment:\nAll rounds were conducted onsite at Gurgaon, requiring self-travel, with no travel reimbursement or support provided.\nBenefits Highlight:\nPackage options: 5, 6, 8, and 10 LPA, depending on candidate performance.\nNo notable perks or benefits were discussed in early stages.\nEvaluator Feedback:\nSince the first round was purely technical and eliminatory, no direct feedback was shared.\nSuggestions for Improvement:\nEnsure stable and reliable coding platforms — frequent crashes affect fairness.\nProvide on-site technical support staff during coding tests.\nConsider whether LeetCode Hard questions in Round 1 are proportional to the package being offered.\nShare at least brief feedback with candidates to help them improve.\nAdditional Information\nTimeline: Applied through referral. Round 1 conducted within 2 weeks of application.\nNext Steps: Could not advance due to submission issues caused by the platform.\nTravel: Candidates were required to travel to Gurgaon at their own expense.\nClosing Note\nThe difficulty level of the first round was comparable to what top product-based companies (Amazon, Adobe, Microsoft, etc.) usually ask, but the candidate experience was disappointing due to poor platform reliability.\nIf you are preparing for Unthinkable Solutions, expect Dynamic Programming (Hard), Backtracking, and String + HashMap problems in the very first round itself, and be prepared to deal with potential technical glitches during the test.\nComment\nArticle Tags:\nArticle Tags:\nInterview Experiences\nUnthinkable Solutions\nExperiences-QnA', 'published_at': '15 Sep, 2025', 'scraped_at': '2026-02-10T16:44:14Z', 'scrape_type': 'bulk', 'scrape_batch_id': '2026-02-10_bulk', 'source_metadata': {'tags': ['Interview Experiences', 'Unthinkable Solutions', 'Experiences-QnA'], 'experience_type': 'unknown'}, 'content_hash': '3d71ec8ab7631251c215f7b8ac0bac1308fd4e85170fb30b5d914deadc79b3dc', 'cleaned_content': 'Shrey Shah --- Unthinkable Solution Interview Experience for Software Developer\n\nCandidate Information\nStatus:\nFinal-year student, seeking full-time opportunities\nExperience:\nAcademic and internship projects (MERN stack, DSA practice)\nTarget Position:\nSoftware Engineer (Entry-level)\nLocation\n: Gurgaon, Haryana, India\nInterview Date:\n13 September 2025\nRound 1: Online Assessment (Eliminatory)\nDuration:\n90 minutes\nMethod:\nOn-site at Gurgaon (the company\'s sandbox platform)\nFocus Areas:\nData Structures & Algorithms (Strings, Dynamic Programming, Backtracking).\nObstacles Faced\nThe company\'s compiler (sandbox environment) repeatedly crashed/auto-refreshed, erasing progress.\nSupport staff only advised to "refresh again" instead of resolving the issue.\nI solved Q1 completely but could not make final submissions for Q2 and Q3 due to these platform problems.\nFurther Rounds (Managerial/Technical/HR)\nMentioned as part of the process, but I could not proceed beyond Round 1 because of its eliminatory structure.\nIn total, Unthinkable Solutions conducts 6 rounds (all eliminatory).\nPost-Interview Reflections\nCompany Culture Insights:\nThe process appeared to be rigid and process-driven, with less focus on creating a candidate-friendly experience.\nWork Environment:\nAll rounds were conducted onsite at Gurgaon, requiring self-travel, with no travel reimbursement or support provided.\nBenefits Highlight:\nPackage options: 5, 6, 8, and 10 LPA, depending on candidate performance.\nNo notable perks or benefits were discussed in early stages.\nEvaluator Feedback:\nSince the first round was purely technical and eliminatory, no direct feedback was shared.\nSuggestions for Improvement:\nEnsure stable and reliable coding platforms - frequent crashes affect fairness.\nProvide on-site technical support staff during coding tests.\nConsider whether LeetCode Hard questions in Round 1 are proportional to the package being offered.\nShare at least brief feedback with candidates to help them improve.\nAdditional Information\nTimeline: Applied through referral. Round 1 conducted within 2 weeks of application.\nNext Steps: Could not advance due to submission issues caused by the platform.\nTravel: Candidates were required to travel to Gurgaon at their own expense.\nClosing Note\nThe difficulty level of the first round was comparable to what top product-based companies (Amazon, Adobe, Microsoft, etc.) usually ask, but the candidate experience was disappointing due to poor platform reliability.\nIf you are preparing for Unthinkable Solutions, expect Dynamic Programming (Hard), Backtracking, and String + HashMap problems in the very first round itself, and be prepared to deal with potential technical glitches during the test.', 'word_count': 372, 'cleaned_title': 'Unthinkable Solution Interview Experience for Software Developer'}
#     print("*"*50)
#     print(pii_remover.process(doc))