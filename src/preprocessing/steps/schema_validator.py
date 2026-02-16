"""
Step 7: Schema Validation

Final gate before database insertion. Validates the enriched doc dict
against the ProcessedInterviewDocument schema and converts it into
a frozen dataclass instance.

Two outcomes:
    - Valid:   Returns doc dict with "processed_document" key containing
               the ProcessedInterviewDocument instance.
    - Invalid: Quarantines the doc to a separate GCS path for manual
               review. Returns None (filtered from pipeline).

Quarantine path format:
    quarantine/schema_failures/{scrape_batch_id}/{document_id}.json

The quarantined JSON includes the original doc dict plus a
"_validation_errors" field describing what failed.

Input:  doc dict enriched by all prior steps (1-6)
Output: Same dict + "processed_document" key, or None if invalid.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict

from src.preprocessing.steps.base import PreprocessingStep
from src.data_models.preprocessed_document import (
    ProcessedInterviewDocument,
    VALID_PLATFORMS,
    VALID_EXPERIENCE,
    VALID_OUTCOMES,
    VALID_DIFFICULTY,
    VALID_INTERVIEW_TYPES,
)


# Platform alias mapping for normalization before validation
PLATFORM_NORMALIZE = {
    "gfg": "geeksforgeeks",
    "geeksforgeeks": "geeksforgeeks",
    "medium": "medium",
    "leetcode": "leetcode",
}


class SchemaValidator(PreprocessingStep):
    """
    Validates enriched doc dicts against ProcessedInterviewDocument
    and quarantines failures to GCS for manual review.

    Args:
        storage_backend: GCS (or local) storage backend for writing
                         quarantined docs. If None, quarantine is
                         logged but not persisted.
        quarantine_prefix: GCS path prefix for quarantined docs.
    """

    def __init__(
        self,
        storage_backend=None,
        quarantine_prefix: str = "quarantine/schema_failures",
    ):
        self._storage = storage_backend
        self._quarantine_prefix = quarantine_prefix
        super().__init__()

    @property
    def name(self) -> str:
        return "schema_validator"

    def process(self, doc: dict) -> ProcessedInterviewDocument | None:
        """
        Validate and convert doc dict → ProcessedInterviewDocument.

        Steps:
            1. Map enriched doc fields to ProcessedInterviewDocument fields
            2. Apply defaults and normalize enum values
            3. Run pre-validation checks (catch errors before dataclass init)
            4. Construct ProcessedInterviewDocument (triggers __post_init__)
            5. On success: attach instance to doc and pass through
            6. On failure: quarantine doc and return None
        """
        errors: List[str] = []

        # ── Step 1: Map fields from pipeline dict to schema fields ──
        mapped = self._map_fields(doc, errors)

        # # ── Step 2: Pre-validation — check required fields exist ──
        # self._validate_required_fields(mapped, errors)

        # ── Step 3: Normalize enum values to valid set ──
        mapped = self._normalize_enums(mapped, errors)

        # If we already have errors from required fields, quarantine early
        if errors:
            self._quarantine(doc, errors)
            doc["_filter_reason"] = "schema_validation_failed"
            return None

        # ── Step 4: Construct ProcessedInterviewDocument ──
        try:
            processed = ProcessedInterviewDocument(**mapped)
            # doc["processed_document"] = processed
            return processed

        except (ValueError, TypeError) as e:
            errors.append(f"Dataclass construction failed: {str(e)}")
            self._quarantine(doc, errors)
            doc["_filter_reason"] = "schema_validation_failed"
            return None

    def _map_fields(self, doc: dict, errors: List[str]) -> dict:
        """
        Map the enriched pipeline dict fields to
        ProcessedInterviewDocument constructor kwargs.

        The pipeline dict has keys from all prior steps:
            Step 1: cleaned_content, word_count, title
            Step 2: pii_removed, pii_counts
            Step 3: quality_score
            Step 4: content_hash
            Step 5: company, role, experience_level, interview_types,
                    num_rounds, topics, interview_outcome, difficulty

        This method maps them to the dataclass field names.
        """
        entity_data = doc.get("preprocessing", {}).get("entity_extractor", {})
        pii_data = doc.get("preprocessing", {}).get("pii_remover", {})
        normalizer_data = doc.get("preprocessing", {}).get("content_normalizer", {})

        return {
            # ── Identity & Lineage ──
            "document_id": doc.get("document_id"),
            "source_platform": self._normalize_platform(
                doc.get("source_platform", "")
            ),
            "source_url": doc.get("source_url"),
            "content_hash": doc.get("content_hash"),

            # ── Content ──
            "title": pii_data.get("title"),
            "content": pii_data.get("content"),
            "word_count": normalizer_data.get("word_count", 0),

            # ── Extracted Entities ──
            "company": entity_data.get("company"),
            "role": entity_data.get("role"),
            "experience_level": entity_data.get("experience_level", "unknown"),

            # ── Extracted Structure ──
            "interview_outcome": entity_data.get("interview_outcome", "unknown"),
            "difficulty": entity_data.get("difficulty", "unknown"),
            "num_rounds": entity_data.get("num_rounds"),
            "interview_types": entity_data.get("interview_types", []),

            # ── Topics ──
            "topics": entity_data.get("topics", []),

            # ── Timestamps ──
            "published_at": doc.get("published_at"),
            "scraped_at": doc.get("scraped_at", ""),
            "preprocessed_at": ProcessedInterviewDocument.now_iso(),

            # ── Scrape Lineage ──
            "scrape_batch_id": doc.get("scrape_batch_id", ""),
            "source_metadata": doc.get("source_metadata", {}),
        }

    def _validate_required_fields(
        self, mapped: dict, errors: List[str]
    ) -> None:
        """Check that non-optional fields are present and non-empty."""
        required = [
            "document_id",
            "source_platform",
            "source_url",
            "content_hash",
            "title",
            "content",
        ]
        for field_name in required:
            value = mapped.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"Missing required field: '{field_name}'")

        # word_count must be positive
        if mapped.get("word_count", 0) <= 0:
            errors.append(
                f"Invalid word_count: {mapped.get('word_count')}"
            )

    def _normalize_enums(
        self, mapped: dict, errors: List[str]
    ) -> dict:
        """
        Normalize enum fields to valid values.

        Strategy: if the value is invalid, try lowercasing and stripping.
        If still invalid, fall back to "unknown" for non-critical fields.
        For source_platform (critical), add an error.
        """
        # source_platform — critical, cannot default to unknown
        platform = mapped.get("source_platform", "")
        if platform not in VALID_PLATFORMS:
            errors.append(
                f"Invalid source_platform: '{platform}'. "
                f"Must be one of {VALID_PLATFORMS}"
            )

        # experience_level — default to unknown
        mapped["experience_level"] = self._safe_enum(
            mapped.get("experience_level"), VALID_EXPERIENCE, "unknown"
        )

        # interview_outcome — default to unknown
        mapped["interview_outcome"] = self._safe_enum(
            mapped.get("interview_outcome"), VALID_OUTCOMES, "unknown"
        )

        # difficulty — default to unknown
        mapped["difficulty"] = self._safe_enum(
            mapped.get("difficulty"), VALID_DIFFICULTY, "unknown"
        )

        # interview_types — filter out invalid values
        raw_types = mapped.get("interview_types", [])
        valid_types = [
            t for t in raw_types
            if t in VALID_INTERVIEW_TYPES
        ]
        # Keep "unknown" only if nothing valid was found
        mapped["interview_types"] = valid_types if valid_types else []

        return mapped

    @staticmethod
    def _safe_enum(value, valid_set, default):
        """Normalize enum value: lowercase + strip, fallback to default."""
        if value is None:
            return default
        cleaned = str(value).lower().strip()
        return cleaned if cleaned in valid_set else default

    @staticmethod
    def _normalize_platform(raw: str) -> str:
        """Map platform aliases to canonical names."""
        return PLATFORM_NORMALIZE.get(raw.lower().strip(), raw.lower().strip())

    # ── Quarantine ──

    def _quarantine(self, doc: dict, errors: List[str]) -> None:
        """
        Write failed doc to quarantine path in GCS for manual review.

        Quarantine path:
            {quarantine_prefix}/{scrape_batch_id}/{document_id}.json

        The quarantined file includes:
            - All original doc fields
            - _validation_errors: list of error descriptions
            - _quarantined_at: timestamp
        """
        doc_id = doc.get("document_id", "unknown")
        batch_id = doc.get("scrape_batch_id", "unknown_batch")

        self.logger.warning(
            f"Quarantining doc '{doc_id}': {errors}"
        )

        # Build quarantine record
        quarantine_record = {
            "_validation_errors": errors,
            "_quarantined_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }

        # Include all original doc fields (exclude internal pipeline keys)
        for key, value in doc.items():
            if not key.startswith("_") and key != "processed_document":
                quarantine_record[key] = value

        # Write to GCS if storage backend is available
        if self._storage:
            path = (
                f"{self._quarantine_prefix}/{batch_id}/{doc_id}.json"
            )
            try:
                self._storage.write_json(path, quarantine_record)
                self.logger.info(f"Quarantined doc written to: {path}")
            except Exception as e:
                self.logger.error(
                    f"Failed to write quarantine for '{doc_id}': {e}"
                )
        else:
            self.logger.warning(
                f"No storage backend — quarantine for '{doc_id}' "
                f"logged but not persisted."
            )


if __name__ == "__main__":
    validator = SchemaValidator()
    doc = {'document_id': 'leetcode_d52d8899feb5712ed525ac1cfe01e6257e104ba0bb343bfa59b27ba8d98f5e2a', 'source_platform': 'leetcode', 'source_url': 'https://leetcode.com/discuss/post/7571535/cleartrip-flipkart-sde3-interview-experi-yjlk/', 'title': 'ClearTrip (Flipkart) SDE3 Interview Experience', 'raw_content': "Hi everyone,\nI have studied a lot of interview experiences here and after appearing for multiple companies I have literally come to appreciate what this community is. I just want to give back.\nSo recently I got selected for the SDE3 role at ClearTrip (Flipkart) and I want to share my interview experience.\n- There were in total 5 rounds, 1 online test and then 4 rounds on-site in a single day, all 4 rounds were elimination rounds. I got a call through recruiter, I have found myself to be particularly lucky that I get calls as my resume gets shortlisted. I try to be active on LinkedIn, Naukri and InstaHyre, I try to apply as many jobs as possible and also added OpenToWork on my LinkedIn profile, was actively replying to messages on LinkedIn from recruiters and also commenting on job related posts\n- **ROUND 1 - DSA online test (1 hr),** I don't remember the exact questions but they were easy to average medium difficulty, for someone preparing DSA for interviews, this test should not be a problem.\n- **ON-SITE LOOP 1 - Machine coding round,** we were all given the same question and had to present our solution afterwards to an interviewer. We were asked ClearFit Demo application question which has already been given in full by other interview experiences, you have around 1:30-2:00 hrs to complete and then 1:00 hr to explain and answer questions on the same.\n- **ON-SITE LOOP 2 - DSA round (1 hr),** I was asked 3 questions, 2 questions related to monotonic stack (most famous questions on this topic) and 1 question similar to koko banana or flower bouquet problem mixed with some arithmetic\n- **ON-SITE LOOP 3 - HLD (1 hr),** I was asked to design a metrics capture and visualization tool, something like prometheus and grafana. Focus was on streaming millions of data points and how to handle such big load. Also went into discussions around the choice of databases both OLAP and OLTP and some depth of OLAP. Also discussed on how we would want to reduce latency in our visualization dashboards. You can talk about data validity, PII, and tiered storage as well.\n- **ON-SITE LOOP 4 - HM (Behavioral) (1 hr),** It can be mostly anything from technical to managerial to techno-managerial, mine was mostly technical and 10 mins of managerial discussion, main focus was on the most big project and why do you want to join CT and why did you leave your last company.\n\nThe process took a complete day and was on a Saturday. Hope this post helps someone, like others have helped me.", 'published_at': '2026-02-11T15:02:06.667317+00:00', 'scraped_at': '2026-02-11T15:30:00Z', 'scrape_type': 'bulk', 'scrape_batch_id': '2026-02-11_bulk', 'source_metadata': {'topic_id': '7571535', 'slug': 'cleartrip-flipkart-sde3-interview-experi-yjlk', 'tags': ['Interview'], 'tag_slugs': ['interview'], 'company': 'Flipkart', 'author_username': 'yash_10', 'hit_count': 27, 'comment_count': 0, 'reactions': {'UPVOTE': 1}, 'created_at': '2026-02-11T15:02:06.667317+00:00', 'updated_at': '2026-02-11T15:04:46.141895+00:00', 'comments': []}, 'content_hash': 'e0ef84b94941d43ace566dd5add8e9c127563fb07cab4b54858dd10a536a0250', 'preprocessing': {'content_normalizer': {'content': "Hi everyone,\nI have studied a lot of interview experiences here and after appearing for multiple companies I have literally come to appreciate what this community is. I just want to give back.\nSo recently I got selected for the SDE3 role at ClearTrip (Flipkart) and I want to share my interview experience.\nThere were in total 5 rounds, 1 online test and then 4 rounds on-site in a single day, all 4 rounds were elimination rounds. I got a call through recruiter, I have found myself to be particularly lucky that I get calls as my resume gets shortlisted. I try to be active on LinkedIn, Naukri and InstaHyre, I try to apply as many jobs as possible and also added OpenToWork on my LinkedIn profile, was actively replying to messages on LinkedIn from recruiters and also commenting on job related posts\nROUND 1 - DSA online test (1 hr), I don't remember the exact questions but they were easy to average medium difficulty, for someone preparing DSA for interviews, this test should not be a problem.\nON-SITE LOOP 1 - Machine coding round, we were all given the same question and had to present our solution afterwards to an interviewer. We were asked ClearFit Demo application question which has already been given in full by other interview experiences, you have around 1:30-2:00 hrs to complete and then 1:00 hr to explain and answer questions on the same.\nON-SITE LOOP 2 - DSA round (1 hr), I was asked 3 questions, 2 questions related to monotonic stack (most famous questions on this topic) and 1 question similar to koko banana or flower bouquet problem mixed with some arithmetic\nON-SITE LOOP 3 - HLD (1 hr), I was asked to design a metrics capture and visualization tool, something like prometheus and grafana. Focus was on streaming millions of data points and how to handle such big load. Also went into discussions around the choice of databases both OLAP and OLTP and some depth of OLAP. Also discussed on how we would want to reduce latency in our visualization dashboards. You can talk about data validity, PII, and tiered storage as well.\nON-SITE LOOP 4 - HM (Behavioral) (1 hr), It can be mostly anything from technical to managerial to techno-managerial, mine was mostly technical and 10 mins of managerial discussion, main focus was on the most big project and why do you want to join CT and why did you leave your last company.\n\nThe process took a complete day and was on a Saturday. Hope this post helps someone, like others have helped me.", 'title': 'ClearTrip (Flipkart) SDE3 Interview Experience', 'word_count': 435, 'completed': True}, 'pii_remover': {'content': "Hi everyone,\nI have studied a lot of interview experiences here and after appearing for multiple companies I have literally come to appreciate what this community is. I just want to give back.\nSo recently I got selected for the SDE3 role at ClearTrip (Flipkart) and I want to share my interview experience.\nThere were in total 5 rounds, 1 online test and then 4 rounds on-site in a single day, all 4 rounds were elimination rounds. I got a call through recruiter, I have found myself to be particularly lucky that I get calls as my resume gets shortlisted. I try to be active on LinkedIn, Naukri and InstaHyre, I try to apply as many jobs as possible and also added OpenToWork on my LinkedIn profile, was actively replying to messages on LinkedIn from recruiters and also commenting on job related posts\nROUND 1 - DSA online test (1 hr), I don't remember the exact questions but they were easy to average medium difficulty, for someone preparing DSA for interviews, this test should not be a problem.\nON-SITE LOOP 1 - Machine coding round, we were all given the same question and had to present our solution afterwards to an interviewer. We were asked ClearFit Demo application question which has already been given in full by other interview experiences, you have around 1:30-2:00 hrs to complete and then 1:00 hr to explain and answer questions on the same.\nON-SITE LOOP 2 - DSA round (1 hr), I was asked 3 questions, 2 questions related to monotonic stack (most famous questions on this topic) and 1 question similar to koko banana or flower bouquet problem mixed with some arithmetic\nON-SITE LOOP 3 - HLD (1 hr), I was asked to design a metrics capture and visualization tool, something like prometheus and grafana. Focus was on streaming millions of data points and how to handle such big load. Also went into discussions around the choice of databases both OLAP and OLTP and some depth of OLAP. Also discussed on how we would want to reduce latency in our visualization dashboards. You can talk about data validity, PII, and tiered storage as well.\nON-SITE LOOP 4 - HM (Behavioral) (1 hr), It can be mostly anything from technical to managerial to techno-managerial, mine was mostly technical and 10 mins of managerial discussion, main focus was on the most big project and why do you want to join CT and why did you leave your last company.\n\nThe process took a complete day and was on a Saturday. Hope this post helps someone, like others have helped me.", 'title': 'ClearTrip (Flipkart) SDE3 Interview Experience', 'pii_counts': {}, 'completed': True}, 'quality_filter': {'quality_score': 1.0, 'completed': True}, 'entity_extractor': {'company': 'Flipkart', 'role': 'Software Development Engineer', 'experience_level': 'senior', 'interview_types': ['onsite', 'online_assessment', 'off_campus'], 'num_rounds': 5, 'topics': ['behavioral', 'dsa', 'os', 'sql_dbms', 'system_design'], 'interview_outcome': 'offer', 'difficulty': 'unknown', 'completed': True}}}

    result = validator.process(doc)
    print(result)
    print("********")
    # if result:
    #     processed = result["processed_document"]  # ProcessedInterviewDocument instance
    #     db.insert(processed.to_dict())