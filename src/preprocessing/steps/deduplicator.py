"""
Step 4: Deduplication

Two-tier dedup to catch both identical and near-identical documents
across platforms (e.g., same experience posted on Reddit and GFK).

    Tier 1 — Exact Dedup:
        SHA256 hash on normalized cleaned_content. Catches verbatim
        reposts and scraper double-fetches. O(1) lookup per doc.

    Tier 2 — Near Dedup (MinHash + LSH):
        MinHash signatures with Locality-Sensitive Hashing to catch
        paraphrased or lightly edited cross-posts. Catches cases like
        same interview experience with minor formatting differences,
        added intro/outro, or platform-specific edits.

Dedup state must persist across batches — we check new docs against
ALL previously processed docs, not just the current batch. State is
stored as:
    - Exact hashes:   Set of SHA256 strings (in-memory + DB lookup)
    - MinHash sigs:   datasketch MinHashLSH index (serialized to disk)

Input:  doc dict with "cleaned_content" (from Steps 1-3)
Output: Same doc dict if unique, None if duplicate.
        Adds "content_hash_exact" and "is_near_duplicate_of" (if near-dup).

All thresholds configurable via dedup_config.yaml.
"""

import hashlib
import re
import pickle
from pathlib import Path
from typing import Optional, Set, Dict

import yaml
from datasketch import MinHash, MinHashLSH

from src.preprocessing.steps.base import PreprocessingStep


# ── Resource / config loading ──

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
CONFIG_FILE = RESOURCES_DIR / "dedup_configs.yaml"


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Dedup config not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG = _load_config()


class Deduplicator(PreprocessingStep):
    """
    Two-tier deduplication: exact hash + MinHash LSH near-dedup.

    Maintains dedup state across batches via:
        - In-memory sets for current pipeline run
        - Optional state_dir for persisting LSH index between runs

    Args:
        similarity_threshold: Jaccard similarity above which two docs
                              are considered near-duplicates. (0.0-1.0)
        num_perm:             Number of MinHash permutations. Higher =
                              more accurate but slower. 128 is standard.
        shingle_size:         Word n-gram size for shingling. 3-5 works
                              well for interview text.
        state_dir:            Directory to persist LSH index between runs.
                              If None, dedup state is in-memory only
                              (resets each pipeline run).
        existing_hashes:      Set of SHA256 hashes from previously
                              processed docs (loaded from DB at pipeline start).
    """

    def __init__(
        self,
        similarity_threshold: float = None,
        num_perm: int = None,
        shingle_size: int = None,
        state_dir: str = None,
        existing_hashes: Set[str] = None,
    ):
        lsh_config = _CONFIG.get("lsh", {})
        exact_config = _CONFIG.get("exact", {})

        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else lsh_config.get("similarity_threshold", 0.80)
        )
        self.num_perm = (
            num_perm
            if num_perm is not None
            else lsh_config.get("num_perm", 128)
        )
        self.shingle_size = (
            shingle_size
            if shingle_size is not None
            else lsh_config.get("shingle_size", 3)
        )
        self.min_shingles = lsh_config.get("min_shingles", 10)

        # ── Exact dedup state ──
        # Start with hashes from DB (previously processed docs)
        self._exact_hashes: Set[str] = (
            set(existing_hashes) if existing_hashes else set()
        )

        # ── Near-dedup state ──
        self._state_dir = Path(state_dir) if state_dir else None
        self._lsh = self._load_or_create_lsh()

        # Track doc_id → minhash for LSH index
        self._doc_signatures: Dict[str, MinHash] = {}

        super().__init__()

    @property
    def name(self) -> str:
        return "deduplicator"

    def process(self, doc: dict) -> Optional[dict]:
        """
        Check document against both tiers.

        Tier 1 (exact) runs first — it's O(1) and catches the
        most common case (verbatim duplicates).

        Tier 2 (near-dedup) only runs if Tier 1 passes.
        """
        cleaned = doc.get("preprocessing", "").get("pii_remover","").get("content","")
        doc_id = doc.get("document_id", "unknown")

        if "deduplicator" not in doc["preprocessing"]:
            doc["preprocessing"]["deduplicator"] = {}

        if not cleaned.strip():
            doc["_filter_reason"] = "empty_content_at_dedup"
            return None

        # ── Tier 1: Exact dedup ──
        content_hash = self._compute_exact_hash(cleaned)

        if content_hash in self._exact_hashes:
            doc["_filter_reason"] = "exact_duplicate"
            return None

        # ── Tier 2: Near dedup (MinHash LSH) ──
        shingles = self._create_shingles(cleaned)

        # Skip near-dedup for very short docs — not enough
        # shingles for reliable similarity estimation
        if len(shingles) >= self.min_shingles:
            minhash = self._compute_minhash(shingles)
            near_dup_id = self._check_near_duplicate(minhash)

            if near_dup_id is not None:
                doc["_filter_reason"] = "near_duplicate"
                doc["is_near_duplicate_of"] = near_dup_id
                return None

            # Not a duplicate — add to LSH index for future checks
            self._lsh.insert(doc_id, minhash)
            self._doc_signatures[doc_id] = minhash

        # ── Unique — register hash and pass through ──
        self._exact_hashes.add(content_hash)

        doc["preprocessing"]["deduplicator"]["content_hash_exact"] = content_hash
        doc["preprocessing"]["deduplicator"]["completed"] = True

        return doc

    # ── Tier 1: Exact hashing ──

    def _compute_exact_hash(self, text: str) -> str:
        """
        SHA256 hash on normalized text.

        Normalization: lowercase, collapse whitespace, strip.
        This ensures formatting-only differences don't produce
        different hashes (e.g., extra newlines between platforms).
        """
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ── Tier 2: MinHash + LSH ──

    def _create_shingles(self, text: str) -> set:
        """
        Create word-level n-gram shingles from text.

        Word shingles (vs character shingles) work better for
        interview content where word choice matters more than
        character-level patterns.

        Example (shingle_size=3):
            "the interviewer asked about trees"
            → {"the interviewer asked", "interviewer asked about",
               "asked about trees"}
        """
        words = re.sub(r"\s+", " ", text.lower().strip()).split()
        if len(words) < self.shingle_size:
            return {" ".join(words)}
        return {
            " ".join(words[i : i + self.shingle_size])
            for i in range(len(words) - self.shingle_size + 1)
        }

    def _compute_minhash(self, shingles: set) -> MinHash:
        """Compute MinHash signature from shingle set."""
        mh = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))
        return mh

    def _check_near_duplicate(self, minhash: MinHash) -> Optional[str]:
        """
        Query LSH index for near-duplicates.

        Returns the doc_id of the first matching duplicate found,
        or None if unique.
        """
        candidates = self._lsh.query(minhash)
        if candidates:
            # Return the first match — for logging/traceability
            return candidates[0]
        return None

    # ── State persistence ──

    def _load_or_create_lsh(self) -> MinHashLSH:
        """
        Load persisted LSH index from disk, or create a new one.

        Persistence allows dedup state to survive across pipeline
        runs — new batches are checked against all previously
        processed documents.
        """
        lsh_path = self._get_lsh_path()

        if lsh_path and lsh_path.exists():
            try:
                with open(lsh_path, "rb") as f:
                    lsh = pickle.load(f)
                return lsh
            except Exception:
                pass  # Corrupted file — start fresh

        return MinHashLSH(
            threshold=self.similarity_threshold,
            num_perm=self.num_perm,
        )

    def save_state(self) -> None:
        """
        Persist LSH index to disk after batch processing.

        Called by the pipeline orchestrator after run_batch completes.
        Not called automatically — the orchestrator controls when
        state is checkpointed.
        """
        lsh_path = self._get_lsh_path()
        if lsh_path:
            lsh_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lsh_path, "wb") as f:
                pickle.dump(self._lsh, f)

    def _get_lsh_path(self) -> Optional[Path]:
        if self._state_dir:
            return self._state_dir / "lsh_index.pkl"
        return None

    @property
    def stats(self) -> dict:
        """Current dedup state stats for monitoring."""
        return {
            "exact_hashes_tracked": len(self._exact_hashes),
            "lsh_signatures_tracked": len(self._doc_signatures),
            "similarity_threshold": self.similarity_threshold,
        }

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
#     }
#     doc = content_normalizer.process(doc)
#     doc = pii_remover.process(doc)
#     doc = quality_filter.process(doc)
#
#     from preprocessing.steps.deduplicator import Deduplicator
#     # At pipeline start — load existing hashes from DB
#     # existing = {row.content_hash for row in db.query("SELECT content_hash_exact FROM processed_documents")}
#     existing = {"e9fbc11657ac125b1dcb6f2bb5dda80ffb894dafd944d07be8f32d8a84cf91ba"}
#     dedup = Deduplicator(
#         existing_hashes=existing,
#         # state_dir="/tmp/pipeline_state"
#     )
#     # After batch
#     surviving_docs = dedup.process(doc)
#     dedup.save_state()
#     print(doc)