"""
Abstract base class for all preprocessing steps.

Every preprocessing step in the pipeline implements this interface,
enabling consistent orchestration, logging, and testing across steps.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import logging
import time


@dataclass
class StepResult:
    """
    Aggregated stats from running a preprocessing step over a batch.

    Provides observability into how each step affects the pipeline —
    how many docs passed, how many were filtered, and why.
    """

    step_name: str
    input_count: int = 0
    output_count: int = 0
    filtered_count: int = 0
    error_count: int = 0
    duration_seconds: float = 0.0
    filter_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.output_count / self.input_count


class PreprocessingStep(ABC):
    """
    Base interface for a single preprocessing step.

    Each step receives a document dict and either:
        - Returns the transformed document (possibly modified in place)
        - Returns None to signal the document should be dropped

    Subclasses must implement:
        - `name`: A unique identifier for the step (used in logs/metrics)
        - `process`: The core transformation logic for a single document

    The `run_batch` method handles iteration, error handling, and stats
    collection so subclasses only focus on single-document logic.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"preprocessing.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this step (e.g., 'pii_removal', 'quality_filter')."""
        ...

    @abstractmethod
    def process(self, doc: dict) -> Optional[dict]:
        """
        Process a single document.

        Args:
            doc: Document dict conforming to the pipeline's intermediate schema.
                 At minimum contains fields from ScrapedInterviewDocument,
                 progressively enriched by earlier steps.

        Returns:
            The transformed document dict, or None if the document
            should be filtered out of the pipeline.
        """
        ...

    def run_batch(self, docs: List[dict]) -> tuple[List[dict], StepResult]:
        """
        Run this step over a batch of documents.

        Handles iteration, error catching, and metrics collection.
        Subclasses should NOT override this — implement `process` instead.

        Args:
            docs: List of document dicts to process.

        Returns:
            A tuple of (surviving_docs, step_result_stats).
        """
        result = StepResult(step_name=self.name, input_count=len(docs))
        output: List[dict] = []
        start = time.time()

        for doc in docs:
            doc_id = doc.get("document_id", "unknown")
            try:
                processed = self.process(doc)
                if processed is not None:
                    output.append(processed)
                    result.output_count += 1
                else:
                    result.filtered_count += 1
                    reason = doc.get("_filter_reason", "unspecified")
                    result.filter_reasons[reason] = (
                        result.filter_reasons.get(reason, 0) + 1
                    )
                    self.logger.debug(
                        f"Filtered doc '{doc_id}' — reason: {reason}"
                    )
            except Exception as e:
                result.error_count += 1
                self.logger.error(
                    f"Error processing doc '{doc_id}' in step "
                    f"'{self.name}': {e}",
                    exc_info=True,
                )

        result.duration_seconds = time.time() - start
        self.logger.info(
            f"[{self.name}] "
            f"in={result.input_count} "
            f"out={result.output_count} "
            f"filtered={result.filtered_count} "
            f"errors={result.error_count} "
            f"time={result.duration_seconds:.2f}s"
        )
        return output, result