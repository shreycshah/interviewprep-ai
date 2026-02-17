"""
Preprocessing Pipeline Orchestrator

Reads raw scraped documents from GCS, runs them through a configurable
sequence of PreprocessingSteps, and writes ProcessedInterviewDocuments
back to GCS. Supports checkpointing after expensive steps, batch
processing, and resume-from-checkpoint on failure.

Design decisions:
    - Config-driven: Step ordering, toggles, and batch sizes live in
      pipeline_config.yaml — no code changes to reorder or skip steps.
    - Single-process: All steps run in one process. Airflow calls this
      as a single task — parallelism is across documents, not steps.
    - Checkpoint-based resilience: After expensive steps (dedup, NER),
      intermediate state is persisted to GCS. On crash, the pipeline
      resumes from the last checkpoint instead of re-running everything.
    - Quarantine, don't drop: Docs that fail schema validation go to a
      quarantine bucket path for manual review — nothing is silently lost.

Usage:
    from src.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    pipeline.run(batch_id="scrape_2025-02-16")

    # Or resume from a checkpoint after a crash:
    pipeline.run(batch_id="scrape_2025-02-16", resume=True)
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

import yaml

from src.preprocessing.steps.base import PreprocessingStep
from src.preprocessing.registry import _STEP_REGISTRY
from src.storage.gcs_backend import GCSBackend


# ── Resource loading ──

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
CONFIG_FILE = RESOURCES_DIR / "pipeline_config.yaml"

logger = logging.getLogger("preprocessing.pipeline")


def _load_config() -> dict:
    """Load the pipeline YAML config once at module level."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Pipeline config not found: {CONFIG_FILE}"
        )
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG = _load_config()


# ─────────────────────────────────────────────────
# Step Registry
# ─────────────────────────────────────────────────

# Maps step names (from YAML config) to their classes.
# Each step module registers itself here on import.
# This avoids a giant if/elif chain and lets new steps
# plug in by adding one entry.

def _build_steps(
    step_configs: List[dict],
    step_kwargs: Dict[str, dict] = None,
) -> List[PreprocessingStep]:
    """
    Instantiate the ordered list of steps from YAML config.

    Skips steps with enabled: false. Raises if a step name
    isn't found in the registry.

    Args:
        step_configs: List of step dicts from pipeline_config.yaml.
        step_kwargs:  Optional dict mapping step names to constructor
                      kwargs. Used for steps that need runtime data
                      (e.g., deduplicator needs existing_hashes from DB).
                      Example: {"deduplicator": {"existing_hashes": {…}}}
    """
    step_kwargs = step_kwargs or {}
    steps = []
    for cfg in step_configs:
        name = cfg["name"]
        if not cfg.get("enabled", True):
            logger.info(f"Step '{name}' is disabled — skipping")
            continue
        if name not in _STEP_REGISTRY:
            raise ValueError(
                f"Unknown step '{name}' in pipeline config. "
                f"Registered steps: {list(_STEP_REGISTRY.keys())}"
            )
        kwargs = step_kwargs.get(name, {})
        steps.append(_STEP_REGISTRY[name](**kwargs))
    return steps


# ─────────────────────────────────────────────────
# Pipeline Run Report
# ─────────────────────────────────────────────────


@dataclass
class PipelineReport:
    """
    Summary of a full pipeline run.

    Written to GCS alongside processed output for audit trail.
    """

    batch_id: str
    started_at: str = ""
    completed_at: str = ""
    total_duration_seconds: float = 0.0
    input_count: int = 0
    output_count: int = 0
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    resumed_from_step: Optional[str] = None
    status: str = "pending"  # pending | completed | failed

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────────


class CheckpointManager:
    """
    Persists intermediate pipeline state to GCS after expensive steps.

    Checkpoint format:
        checkpoints/{batch_id}/{step_name}.jsonl
        checkpoints/{batch_id}/_meta.json  (tracks last completed step)

    On resume, the orchestrator reads _meta.json to find which step
    completed last, loads that step's checkpoint, and continues from
    the next step.
    """

    def __init__(self, storage: GCSBackend, prefix: str, batch_id: str):
        self.storage = storage
        self.base_path = f"{prefix}{batch_id}"

    @property
    def meta_path(self) -> str:
        return f"{self.base_path}/_meta.json"

    def save(self, step_name: str, docs: List[dict]) -> None:
        """Write docs as JSONL and update the meta pointer."""
        # Write docs
        jsonl_path = f"{self.base_path}/{step_name}.jsonl"
        lines = [json.dumps(doc, ensure_ascii=False) for doc in docs]
        content = "\n".join(lines)

        blob = self.storage.bucket.blob(jsonl_path)
        blob.upload_from_string(content, content_type="application/jsonl")
        logger.info(
            f"Checkpoint saved: {jsonl_path} ({len(docs)} docs)"
        )

        # Update meta
        meta = {"last_completed_step": step_name, "doc_count": len(docs)}
        self.storage.write_json(self.meta_path, meta)

    def load_latest(self) -> tuple[Optional[str], List[dict]]:
        """
        Load the most recent checkpoint.

        Returns:
            (step_name, docs) if a checkpoint exists, else (None, []).
        """
        meta = self.storage.read_json(self.meta_path)
        if meta is None:
            return None, []

        step_name = meta["last_completed_step"]
        jsonl_path = f"{self.base_path}/{step_name}.jsonl"

        blob = self.storage.bucket.blob(jsonl_path)
        if not blob.exists():
            logger.warning(
                f"Meta points to '{step_name}' but checkpoint file missing"
            )
            return None, []

        content = blob.download_as_text()
        docs = [json.loads(line) for line in content.strip().split("\n") if line]
        logger.info(
            f"Loaded checkpoint from step '{step_name}' ({len(docs)} docs)"
        )
        return step_name, docs

    def cleanup(self) -> None:
        """Delete all checkpoint files for this batch."""
        blobs = list(
            self.storage.client.list_blobs(
                self.storage.bucket_name, prefix=self.base_path
            )
        )
        if blobs:
            self.storage.bucket.delete_blobs(blobs)
            logger.info(
                f"Cleaned up {len(blobs)} checkpoint files"
            )


# ─────────────────────────────────────────────────
# Main Pipeline Orchestrator
# ─────────────────────────────────────────────────


class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing pipeline.

    Responsibilities:
        1. Load raw docs from GCS (by batch_id)
        2. Build the step chain from YAML config
        3. Run docs through each step sequentially
        4. Checkpoint after configured steps
        5. Quarantine docs that fail schema validation
        6. Write processed docs + run report to GCS

    This class owns the *flow*. Individual steps own their *logic*.
    """

    def __init__(self):
        """
        Initialize pipeline from module-level config.

        Config is loaded once from resources/pipeline_config.yaml
        at module import time, consistent with how all other
        preprocessing steps load their configs.
        """
        self.config = _CONFIG

        # GCS backend
        gcs_cfg = self.config["gcs"]
        self.storage = GCSBackend(bucket_name=gcs_cfg["bucket_name"],
                                  project_id=gcs_cfg["project_id"],
                                  secret_name=gcs_cfg["secret_name"])

        # Paths
        self.raw_prefix = gcs_cfg["raw_prefix"]
        self.processed_prefix = gcs_cfg["processed_prefix"]
        self.checkpoint_prefix = gcs_cfg["checkpoint_prefix"]
        self.quarantine_prefix = gcs_cfg["quarantine_prefix"]

        # Build step chain
        self.step_configs = self.config["steps"]
        step_kwargs = self._build_step_kwargs()
        self.steps = _build_steps(self.step_configs, step_kwargs)
        self.steps = _build_steps(self.step_configs)

        # Checkpoint settings
        cp_cfg = self.config.get("checkpoint", {})
        self.checkpointing_enabled = cp_cfg.get("enabled", True)
        self.cleanup_checkpoints = cp_cfg.get("cleanup_on_success", True)

        # Batch settings
        batch_cfg = self.config.get("batch", {})
        self.batch_size = batch_cfg.get("size", 500)
        self.fail_fast = batch_cfg.get("fail_fast", False)

        # Build set of step names that need checkpointing
        self.checkpoint_steps = {
            cfg["name"]
            for cfg in self.step_configs
            if cfg.get("checkpoint", False) and cfg.get("enabled", True)
        }

        logger.info(
            f"Pipeline initialized with {len(self.steps)} steps: "
            f"{[s.name for s in self.steps]}"
        )

    # ── Step kwargs builder ──

    def _build_step_kwargs(self) -> Dict[str, dict]:
        """
        Build runtime constructor kwargs for steps that need them.

        Steps like the deduplicator require data that can only be
        fetched at pipeline start (e.g., existing hashes from DB).
        This method centralizes that logic.
        """
        kwargs = {}

        # Deduplicator needs existing content hashes from DB
        # to avoid re-accepting previously processed docs
        if self._is_step_enabled("deduplicator"):
            kwargs["deduplicator"] = {
                "existing_hashes": self._load_existing_hashes()
            }

        return kwargs

    def _is_step_enabled(self, step_name: str) -> bool:
        """Check if a step is enabled in the config."""
        return any(
            cfg["name"] == step_name and cfg.get("enabled", True)
            for cfg in self.step_configs
        )

    def _load_existing_hashes(self) -> set:
        """
        Load content hashes of previously processed documents from DB.

        Used by the deduplicator to avoid re-accepting docs that
        were processed in prior batches.
        """
        # try:
        #     from src.database.connection import get_db_connection
        #
        #     conn = get_db_connection()
        #     cursor = conn.cursor()
        #     cursor.execute(
        #         "SELECT content_hash FROM processed_documents"
        #     )
        #     hashes = {row[0] for row in cursor.fetchall()}
        #     cursor.close()
        #     conn.close()
        #     logger.info(
        #         f"Loaded {len(hashes)} existing hashes from DB"
        #     )
        #     return hashes
        # except Exception as e:
        #     logger.warning(
        #         f"Could not load existing hashes from DB: {e}. "
        #         f"Deduplicator will only check within current batch."
        #     )
        #     return set()
        print("Loading existing hashes.....")
        return set()

    # ── Public API ──

    def run(self, batch_id: str, resume: bool = False) -> PipelineReport:
        """
        Execute the full pipeline for a scrape batch.

        Args:
            batch_id: Identifies which raw docs to process
                      (e.g., "scrape_2025-02-16"). Maps to
                      GCS prefix: raw/{platform}/{date}/.
            resume:   If True, attempt to resume from the last
                      checkpoint instead of starting from scratch.

        Returns:
            PipelineReport with per-step stats and final counts.
        """
        report = PipelineReport(
            batch_id=batch_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        checkpoint_mgr = CheckpointManager(
            self.storage, self.checkpoint_prefix, batch_id
        )

        try:
            # Determine starting point
            docs, start_step_idx = self._resolve_start(
                batch_id, resume, checkpoint_mgr
            )

            if start_step_idx > 0:
                report.resumed_from_step = self.steps[start_step_idx].name

            report.input_count = len(docs)
            logger.info(
                f"Pipeline starting: {len(docs)} docs, "
                f"from step '{self.steps[start_step_idx].name}'"
            )

            # Run each step
            for step in self.steps[start_step_idx:]:
                docs, step_result = step.run_batch(docs)
                report.step_results.append(asdict(step_result))

                if self.fail_fast and step_result.error_count > 0:
                    raise RuntimeError(
                        f"fail_fast enabled: {step_result.error_count} "
                        f"errors in step '{step.name}'"
                    )

                # Checkpoint if configured
                if (
                    self.checkpointing_enabled
                    and step.name in self.checkpoint_steps
                    and docs  # Don't checkpoint empty batches
                ):
                    checkpoint_mgr.save(step.name, docs)

                if not docs:
                    logger.warning(
                        f"All docs filtered out after step '{step.name}' "
                        f"— pipeline ending early"
                    )
                    break

            # Separate valid docs from quarantined
            # valid_docs, quarantined_docs = self._separate_quarantined(docs)
            valid_docs = docs

            # Write outputs
            self._write_processed(batch_id, valid_docs)
            # self._write_quarantined(batch_id, quarantined_docs)
            self._write_report(batch_id, report)

            report.output_count = len(valid_docs)
            # report.quarantined_count = len(quarantined_docs)

            # Cleanup checkpoints on success
            if self.cleanup_checkpoints and self.checkpointing_enabled:
                checkpoint_mgr.cleanup()

            report.status = "completed"

        except Exception as e:
            report.status = "failed"
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            # Still write the partial report for debugging
            self._write_report(batch_id, report)
            raise

        finally:
            report.completed_at = datetime.now(timezone.utc).isoformat()
            elapsed = (
                datetime.fromisoformat(report.completed_at)
                - datetime.fromisoformat(report.started_at)
            ).total_seconds()
            report.total_duration_seconds = elapsed

        logger.info(
            f"Pipeline {report.status}: "
            f"in={report.input_count} "
            f"out={report.output_count} "
            f"quarantined={report.quarantined_count} "
            f"time={report.total_duration_seconds:.1f}s"
        )
        return report

    # ── Private helpers ──

    def _resolve_start(
        self,
        batch_id: str,
        resume: bool,
        checkpoint_mgr: CheckpointManager,
    ) -> tuple[List[dict], int]:
        """
        Determine where to start the pipeline.

        If resume=True and a checkpoint exists, load docs from
        the checkpoint and skip to the step after it.

        Otherwise, load raw docs from GCS.

        Returns:
            (docs, start_step_index)
        """
        if resume and self.checkpointing_enabled:
            last_step, docs = checkpoint_mgr.load_latest()
            if last_step and docs:
                # Find the index of the step AFTER the checkpointed one
                step_names = [s.name for s in self.steps]
                if last_step in step_names:
                    next_idx = step_names.index(last_step) + 1
                    if next_idx < len(self.steps):
                        logger.info(
                            f"Resuming after step '{last_step}' "
                            f"(step {next_idx}/{len(self.steps)})"
                        )
                        return docs, next_idx

                logger.warning(
                    f"Checkpoint step '{last_step}' not found in "
                    f"current config — starting from scratch"
                )

        # Fresh start: load raw docs from GCS
        docs = self._load_raw_docs(batch_id)
        return docs, 0

    def _load_raw_docs(self, batch_id: str) -> List[dict]:
        """
        Load all raw JSON documents for a batch from GCS.

        Scans all platform subdirectories under raw/ for files
        matching the batch date. Each JSON file is one scraped doc.

        GCS layout:
            raw/2025-02-16_bulk/gfg/doc1.json
            raw/2025-02-16_bulk/medium/doc2.json
            raw/2025-02-16_bulk/leetcode/doc3.json
        """
        # Extract date from batch_id (e.g., "scrape_2025-02-16" → "2025-02-16")
        # batch_date = batch_id.replace("scrape_", "")

        docs = []
        platforms = ["gfg", "medium", "leetcode"]

        for platform in platforms:
            prefix = f"{self.raw_prefix}/{batch_id}/{platform}"
            files = self.storage.list_files(prefix=prefix, suffix=".json")

            for filepath in files:
                try:
                    doc = self.storage.read_json(filepath)
                    if doc:
                        # Tag with source path for lineage
                        doc["_source_gcs_path"] = filepath
                        docs.append(doc)
                except Exception as e:
                    logger.error(f"Failed to read {filepath}: {e}")

        logger.info(f"Loaded {len(docs)} raw docs for batch '{batch_id}'")
        return docs

    def _separate_quarantined(
        self, docs: List[dict]
    ) -> tuple[List[dict], List[dict]]:
        """
        Split docs into valid and quarantined.

        The schema_validator step marks invalid docs with
        _quarantine_reason. This method separates them.
        """
        valid = []
        quarantined = []
        for doc in docs:
            if "_quarantine_reason" in doc.keys():
            # if doc.get("_quarantine_reason"):
                quarantined.append(doc)
            else:
                valid.append(doc)
        return valid, quarantined

    def _write_processed(self, batch_id: str, docs: List[dict]) -> None:
        """
        Write processed documents to GCS as individual JSON files.

        Output path:
            processed/{batch_date}/{document_id}.json
        """
        if not docs:
            return

        count = 0

        # for doc in docs:
        #     doc_id = doc.get("document_id", f"unknown_{count}")
        #     path = f"{self.processed_prefix}/{batch_id}/{doc_id}.json"
        #
        #     # Strip internal pipeline metadata before writing
        #     clean_doc = {
        #         k: v for k, v in doc.items() if not k.startswith("_")
        #     }
        #     self.storage.write_json(path, clean_doc)
        #     count += 1
        #
        # logger.info(f"Wrote {count} processed docs to GCS")

        for doc in docs:
            doc_id = getattr(doc, "document_id", f"unknown_{count}")
            clean_doc = doc.to_dict()

            path = f"{self.processed_prefix}/{batch_id}/{doc_id}.json"
            self.storage.write_json(path, clean_doc)
            count += 1

        logger.info(f"Wrote {count} processed docs to GCS")

    def _write_quarantined(self, batch_id: str, docs: List[dict]) -> None:
        """Write quarantined docs to a separate GCS path for review."""
        if not docs:
            return

        batch_date = batch_id.replace("scrape_", "")

        for doc in docs:
            doc_id = doc.get("document_id", "unknown")
            reason = doc.get("_quarantine_reason", "unknown")
            path = (
                f"{self.quarantine_prefix}{batch_date}/"
                f"{reason}/{doc_id}.json"
            )
            self.storage.write_json(path, doc)

        logger.info(f"Quarantined {len(docs)} docs for review")

    def _write_report(self, batch_id: str, report: PipelineReport) -> None:
        """Write the pipeline run report to GCS."""
        path = f"{self.processed_prefix}{batch_id}_report.json"
        self.storage.write_json(path, report.to_dict())


####### RUNNING PIPELINE run.py #############
# import logging
# import sys
#
# # Import steps package to trigger registration
# import src.preprocessing.steps  # noqa: F401
#
# from src.preprocessing.pipeline import PreprocessingPipeline
#
#
# def setup_logging(level: str = "INFO"):
#     """Configure structured logging for the pipeline."""
#     logging.basicConfig(
#         level=getattr(logging, level.upper()),
#         format=(
#             "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
#         ),
#         datefmt="%Y-%m-%d %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
#
#
# def main():
#     setup_logging("INFO")
#
#     # Build pipeline
#     pipeline = PreprocessingPipeline()
#
#     # Run
#     report = pipeline.run(batch_id="2026-02-16_bulk", resume=False)
#
#     # Exit code based on result
#     if report.status == "completed":
#         print(
#             f"\nPipeline completed successfully. "
#             f"{report.output_count}/{report.input_count} docs processed, "
#         )
#         sys.exit(0)
#     else:
#         print(f"\nPipeline failed. Check logs for details.")
#         sys.exit(1)
#
#
# if __name__ == "__main__":
#     main()