
"""
Step 7: Schema Validation

Final gate before database insertion. Maps the enriched doc dict
to ProcessedInterviewDocument constructor kwargs and attempts
construction. All validation lives inside the dataclass itself.

Two outcomes:
    - Valid:   Returns ProcessedInterviewDocument instance.
    - Invalid: Logs the error, marks the doc as filtered, returns None.

Input:  doc dict enriched by all prior steps (1-6)
Output: ProcessedInterviewDocument instance, or None if invalid.
"""

from datetime import datetime, timezone
from typing import Optional, List

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

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "schema_validator"

    def process(self, doc: dict) -> Optional[ProcessedInterviewDocument]:
        """
        Map fields → construct ProcessedInterviewDocument.

        On success: return the frozen dataclass instance.
        On failure: mark doc with _filter_reason and return None.
        """
        mapped = self._map_fields(doc)

        try:
            return ProcessedInterviewDocument(**mapped)

        except (ValueError, TypeError) as e:
            self.logger.warning(
                f"Schema validation failed for '{doc.get('document_id', 'unknown')}': {e}"
            )
            doc["_filter_reason"] = f"schema_validation_failed: {e}"
            return None

    def _map_fields(self, doc: dict) -> dict:
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
        deduplicator_data = doc.get("preprocessing", {}).get("deduplicator", {})
        normalizer_data = doc.get("preprocessing", {}).get("content_normalizer", {})

        return {
            # ── Identity & Lineage ──
            "document_id": doc.get("document_id"),
            "source_platform": doc.get("source_platform"),
            "source_url": doc.get("source_url"),
            "content_hash": deduplicator_data.get("content_hash"),

            # ── Content ──
            "title": pii_data.get("title"),
            "content": pii_data.get("content"),
            "word_count": normalizer_data.get("word_count", 0),

            # ── Extracted Entities ──
            "company": entity_data.get("company"),
            "role": entity_data.get("role"),
            "experience_level": entity_data.get("experience_level", None),

            # ── Extracted Structure ──
            "interview_outcome": entity_data.get("interview_outcome", None),
            "difficulty": entity_data.get("difficulty", None),
            "num_rounds": entity_data.get("num_rounds", None),
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

# if __name__ == "__main__":
#     validator = SchemaValidator()
#     doc = {'document_id': 'leetcode_d52d8899feb5712ed525ac1cfe01e6257e104ba0bb343bfa59b27ba8d98f5e2a', 'source_platform': 'leetcode', 'source_url': 'https://leetcode.com/discuss/post/7571535/cleartrip-flipkart-sde3-interview-experi-yjlk/', 'title': 'ClearTrip (Flipkart) SDE3 Interview Experience', 'raw_content': "Hi everyone,\nI have studied a lot of interview experiences here and after appearing for multiple companies I have literally come to appreciate what this community is. I just want to give back.\nSo recently I got selected for the SDE3 role at ClearTrip (Flipkart) and I want to share my interview experience.\n- There were in total 5 rounds, 1 online test and then 4 rounds on-site in a single day, all 4 rounds were elimination rounds. I got a call through recruiter, I have found myself to be particularly lucky that I get calls as my resume gets shortlisted. I try to be active on LinkedIn, Naukri and InstaHyre, I try to apply as many jobs as possible and also added OpenToWork on my LinkedIn profile, was actively replying to messages on LinkedIn from recruiters and also commenting on job related posts\n- **ROUND 1 - DSA online test (1 hr),** I don't remember the exact questions but they were easy to average medium difficulty, for someone preparing DSA for interviews, this test should not be a problem.\n- **ON-SITE LOOP 1 - Machine coding round,** we were all given the same question and had to present our solution afterwards to an interviewer. We were asked ClearFit Demo application question which has already been given in full by other interview experiences, you have around 1:30-2:00 hrs to complete and then 1:00 hr to explain and answer questions on the same.\n- **ON-SITE LOOP 2 - DSA round (1 hr),** I was asked 3 questions, 2 questions related to monotonic stack (most famous questions on this topic) and 1 question similar to koko banana or flower bouquet problem mixed with some arithmetic\n- **ON-SITE LOOP 3 - HLD (1 hr),** I was asked to design a metrics capture and visualization tool, something like prometheus and grafana. Focus was on streaming millions of data points and how to handle such big load. Also went into discussions around the choice of databases both OLAP and OLTP and some depth of OLAP. Also discussed on how we would want to reduce latency in our visualization dashboards. You can talk about data validity, PII, and tiered storage as well.\n- **ON-SITE LOOP 4 - HM (Behavioral) (1 hr),** It can be mostly anything from technical to managerial to techno-managerial, mine was mostly technical and 10 mins of managerial discussion, main focus was on the most big project and why do you want to join CT and why did you leave your last company.\n\nThe process took a complete day and was on a Saturday. Hope this post helps someone, like others have helped me.", 'published_at': '2026-02-11T15:02:06.667317+00:00', 'scraped_at': '2026-02-11T15:30:00Z', 'scrape_type': 'bulk', 'scrape_batch_id': '2026-02-11_bulk', 'source_metadata': {'topic_id': '7571535', 'slug': 'cleartrip-flipkart-sde3-interview-experi-yjlk', 'tags': ['Interview'], 'tag_slugs': ['interview'], 'company': 'Flipkart', 'author_username': 'yash_10', 'hit_count': 27, 'comment_count': 0, 'reactions': {'UPVOTE': 1}, 'created_at': '2026-02-11T15:02:06.667317+00:00', 'updated_at': '2026-02-11T15:04:46.141895+00:00', 'comments': []}, 'content_hash': 'e0ef84b94941d43ace566dd5add8e9c127563fb07cab4b54858dd10a536a0250', 'preprocessing': {'content_normalizer': {'content': "Hi everyone,\nI have studied a lot of interview experiences here and after appearing for multiple companies I have literally come to appreciate what this community is. I just want to give back.\nSo recently I got selected for the SDE3 role at ClearTrip (Flipkart) and I want to share my interview experience.\nThere were in total 5 rounds, 1 online test and then 4 rounds on-site in a single day, all 4 rounds were elimination rounds. I got a call through recruiter, I have found myself to be particularly lucky that I get calls as my resume gets shortlisted. I try to be active on LinkedIn, Naukri and InstaHyre, I try to apply as many jobs as possible and also added OpenToWork on my LinkedIn profile, was actively replying to messages on LinkedIn from recruiters and also commenting on job related posts\nROUND 1 - DSA online test (1 hr), I don't remember the exact questions but they were easy to average medium difficulty, for someone preparing DSA for interviews, this test should not be a problem.\nON-SITE LOOP 1 - Machine coding round, we were all given the same question and had to present our solution afterwards to an interviewer. We were asked ClearFit Demo application question which has already been given in full by other interview experiences, you have around 1:30-2:00 hrs to complete and then 1:00 hr to explain and answer questions on the same.\nON-SITE LOOP 2 - DSA round (1 hr), I was asked 3 questions, 2 questions related to monotonic stack (most famous questions on this topic) and 1 question similar to koko banana or flower bouquet problem mixed with some arithmetic\nON-SITE LOOP 3 - HLD (1 hr), I was asked to design a metrics capture and visualization tool, something like prometheus and grafana. Focus was on streaming millions of data points and how to handle such big load. Also went into discussions around the choice of databases both OLAP and OLTP and some depth of OLAP. Also discussed on how we would want to reduce latency in our visualization dashboards. You can talk about data validity, PII, and tiered storage as well.\nON-SITE LOOP 4 - HM (Behavioral) (1 hr), It can be mostly anything from technical to managerial to techno-managerial, mine was mostly technical and 10 mins of managerial discussion, main focus was on the most big project and why do you want to join CT and why did you leave your last company.\n\nThe process took a complete day and was on a Saturday. Hope this post helps someone, like others have helped me.", 'title': 'ClearTrip (Flipkart) SDE3 Interview Experience', 'word_count': 435, 'completed': True}, 'pii_remover': {'content': "Hi everyone,\nI have studied a lot of interview experiences here and after appearing for multiple companies I have literally come to appreciate what this community is. I just want to give back.\nSo recently I got selected for the SDE3 role at ClearTrip (Flipkart) and I want to share my interview experience.\nThere were in total 5 rounds, 1 online test and then 4 rounds on-site in a single day, all 4 rounds were elimination rounds. I got a call through recruiter, I have found myself to be particularly lucky that I get calls as my resume gets shortlisted. I try to be active on LinkedIn, Naukri and InstaHyre, I try to apply as many jobs as possible and also added OpenToWork on my LinkedIn profile, was actively replying to messages on LinkedIn from recruiters and also commenting on job related posts\nROUND 1 - DSA online test (1 hr), I don't remember the exact questions but they were easy to average medium difficulty, for someone preparing DSA for interviews, this test should not be a problem.\nON-SITE LOOP 1 - Machine coding round, we were all given the same question and had to present our solution afterwards to an interviewer. We were asked ClearFit Demo application question which has already been given in full by other interview experiences, you have around 1:30-2:00 hrs to complete and then 1:00 hr to explain and answer questions on the same.\nON-SITE LOOP 2 - DSA round (1 hr), I was asked 3 questions, 2 questions related to monotonic stack (most famous questions on this topic) and 1 question similar to koko banana or flower bouquet problem mixed with some arithmetic\nON-SITE LOOP 3 - HLD (1 hr), I was asked to design a metrics capture and visualization tool, something like prometheus and grafana. Focus was on streaming millions of data points and how to handle such big load. Also went into discussions around the choice of databases both OLAP and OLTP and some depth of OLAP. Also discussed on how we would want to reduce latency in our visualization dashboards. You can talk about data validity, PII, and tiered storage as well.\nON-SITE LOOP 4 - HM (Behavioral) (1 hr), It can be mostly anything from technical to managerial to techno-managerial, mine was mostly technical and 10 mins of managerial discussion, main focus was on the most big project and why do you want to join CT and why did you leave your last company.\n\nThe process took a complete day and was on a Saturday. Hope this post helps someone, like others have helped me.", 'title': 'ClearTrip (Flipkart) SDE3 Interview Experience', 'pii_counts': {}, 'completed': True}, 'quality_filter': {'quality_score': 1.0, 'completed': True}, 'entity_extractor': {'company': 'Flipkart', 'role': 'Software Development Engineer', 'experience_level': 'senior', 'interview_types': ['onsite', 'online_assessment', 'off_campus'], 'num_rounds': 5, 'topics': ['behavioral', 'dsa', 'os', 'sql_dbms', 'system_design'], 'interview_outcome': 'offer', 'difficulty': 'unknown', 'completed': True}}}
#
#     result = validator.process(doc)
#     print(result)