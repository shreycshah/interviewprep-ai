"""
LLM-as-Judge Relevance Scorer (GPT-4o-mini)
=============================================
Reads the pooled retrieval_labeling_dataset.csv from GCS, sends each
(query_text, chunk_text) pair to GPT-4o-mini, and writes back a
scored CSV with a `relevance` column (0 / 1 / 2).

Estimated cost: ~$0.40 for 5,000 pairs.

Usage:
    1. Set OPENAI_API_KEY env var or hardcode in LLM_PARAMS
    2. Run: python llm_relevance_judge.py

Prerequisites:
    pip install openai pandas tqdm google-cloud-storage google-cloud-secret-manager

Author: InterviewPrep AI Team
"""

import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from tqdm import tqdm

from src.storage.gcs_backend import GCSBackend


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

GCS_PARAMS = {
    "bucket_name": "interviewprep-ai-data",
    "project_id": "professorbot-dovbsg",
    "secret_name": "gcs-service-account-key",
}

IO_PARAMS = {
    "input_gcs_path": "eval_queries/labeling_dataset_264.csv",
    "output_gcs_path": "eval_queries/labeled_dataset_264.csv",
}

LLM_PARAMS = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY", ""),   # or hardcode here
    "temperature": 0.0,
    "max_tokens": 64,
    "max_chunk_chars": 1500,       # truncate long chunks
    "max_retries": 3,
    "retry_delay": 2,              # seconds between retries
    "concurrent_workers": 5,       # parallel API calls
}

BATCH_PARAMS = {
    "save_every": 200,             # checkpoint to GCS every N rows
}

LOG_LEVEL = "INFO"


# ═════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a relevance judge for an interview preparation search system.

Given a user QUERY and a retrieved CHUNK from an interview experience database, rate the relevance.

Scoring:
0 = NOT RELEVANT — The chunk does not help answer the query. Wrong company, wrong role, or completely off-topic.
1 = PARTIALLY RELEVANT — The chunk has some useful information but doesn't directly answer the query. Right company but wrong role, or right topic but different context.
2 = HIGHLY RELEVANT — The chunk directly addresses the query. Right company, right role, and contains information the user is looking for.

Base judgment ONLY on the provided QUERY and CHUNK text – Do not use external knowledge or assumptions.

Respond with ONLY a JSON object in this exact format, no markdown fences:
{"relevance": <0 or 1 or 2>, "reason": "<one sentence explanation>"}"""

USER_TEMPLATE = """QUERY: {query_text}

CHUNK: {chunk_text}"""


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("llm_judge")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%H:%M:%S")
    )
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# ─── GCS CSV Helpers ──────────────────────────────────────────────────────────

def read_csv_from_gcs(gcs: GCSBackend, gcs_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Reading CSV from gs://%s/%s", gcs.bucket_name, gcs_path)
    blob = gcs.bucket.blob(gcs_path)
    if not blob.exists():
        raise FileNotFoundError(f"CSV not found at gs://{gcs.bucket_name}/{gcs_path}")
    csv_text = blob.download_as_text(encoding="utf-8")
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = df.columns.str.strip()
    logger.info("Read %d rows from GCS.", len(df))
    return df


def write_csv_to_gcs(
    gcs: GCSBackend,
    gcs_path: str,
    df: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    logger.info("Writing %d rows to gs://%s/%s", len(df), gcs.bucket_name, gcs_path)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    blob = gcs.bucket.blob(gcs_path)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")
    logger.info("Upload complete.")


# ─── OpenAI Client ────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = LLM_PARAMS["api_key"]
    if not api_key:
        raise ValueError(
            "OpenAI API key not set. Either:\n"
            "  1. export OPENAI_API_KEY=sk-...\n"
            "  2. Hardcode in LLM_PARAMS['api_key']"
        )
    return OpenAI(api_key=api_key)


# ─── LLM Judge ────────────────────────────────────────────────────────────────

def truncate_chunk(chunk_text: str) -> str:
    limit = LLM_PARAMS["max_chunk_chars"]
    if len(chunk_text) <= limit:
        return chunk_text
    return chunk_text[:limit] + "..."


def parse_llm_response(response_text: str) -> tuple[int, str]:
    """Parse the JSON response. Returns (relevance, reason)."""
    text = response_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        relevance = int(parsed.get("relevance", -1))
        reason = str(parsed.get("reason", ""))

        if relevance not in (0, 1, 2):
            raise ValueError(f"Invalid relevance value: {relevance}")

        return relevance, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: look for a bare digit
        for char in text:
            if char in ("0", "1", "2"):
                return int(char), "parse_fallback"
        raise ValueError(f"Could not parse LLM response: {text[:200]}")


def judge_pair(
    client: OpenAI,
    query_text: str,
    chunk_text: str,
    logger: logging.Logger,
) -> tuple[int, str]:
    """
    Send a single (query, chunk) pair to GPT-4o-mini.
    Returns (relevance, reason). Retries on transient errors.
    """
    truncated = truncate_chunk(chunk_text)
    user_msg = USER_TEMPLATE.format(query_text=query_text, chunk_text=truncated)

    for attempt in range(LLM_PARAMS["max_retries"]):
        try:
            response = client.chat.completions.create(
                model=LLM_PARAMS["model"],
                temperature=LLM_PARAMS["temperature"],
                max_tokens=LLM_PARAMS["max_tokens"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content
            relevance, reason = parse_llm_response(raw)
            return relevance, reason

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            delay = LLM_PARAMS["retry_delay"] * (2 ** attempt)
            logger.warning(
                "API error (attempt %d/%d): %s. Retrying in %ds...",
                attempt + 1, LLM_PARAMS["max_retries"], type(e).__name__, delay,
            )
            time.sleep(delay)

        except ValueError as e:
            if attempt < LLM_PARAMS["max_retries"] - 1:
                logger.debug("Parse failed (attempt %d): %s", attempt + 1, e)
                continue
            else:
                logger.warning("All parse attempts failed. Error: %s", e)
                return -1, "parse_failed"

        except Exception as e:
            logger.error("Unexpected error: %s", e)
            return -1, f"error: {str(e)[:100]}"

    return -1, "max_retries_exceeded"


def judge_pair_wrapper(args):
    """Wrapper for ThreadPoolExecutor — unpacks args and calls judge_pair."""
    idx, client, query_text, chunk_text, logger = args
    relevance, reason = judge_pair(client, query_text, chunk_text, logger)
    return idx, relevance, reason


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("LLM-as-Judge Relevance Scorer (GPT-4o-mini)")
    logger.info("=" * 60)
    logger.info("Model: %s | Workers: %d", LLM_PARAMS["model"], LLM_PARAMS["concurrent_workers"])

    # ── Initialize OpenAI client ──────────────────────────────────────────
    client = get_client()

    # Verify connectivity with a tiny test call
    logger.info("Verifying OpenAI API connection...")
    try:
        test = client.chat.completions.create(
            model=LLM_PARAMS["model"],
            max_tokens=5,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        logger.info("API connection verified.")
    except Exception as e:
        logger.error("OpenAI API test failed: %s", e)
        sys.exit(1)

    # ── Initialize GCS ────────────────────────────────────────────────────
    logger.info("Initializing GCS backend...")
    gcs = GCSBackend(
        bucket_name=GCS_PARAMS["bucket_name"],
        project_id=GCS_PARAMS["project_id"],
        secret_name=GCS_PARAMS["secret_name"],
    )

    # ── Load labeling dataset from GCS ────────────────────────────────────
    df = read_csv_from_gcs(gcs, IO_PARAMS["input_gcs_path"], logger)
    total_rows = len(df)
    logger.info("Loaded %d query-chunk pairs to judge.", total_rows)

    # ── Check for existing progress (resume support) ──────────────────────
    output_path = IO_PARAMS["output_gcs_path"]
    scored_so_far = 0

    try:
        existing_df = read_csv_from_gcs(gcs, output_path, logger)
        if "relevance" in existing_df.columns:
            scored_mask = existing_df["relevance"].notna() & (existing_df["relevance"] != "")
            scored_so_far = int(scored_mask.sum())
            if scored_so_far > 0:
                logger.info("Found %d already-scored rows. Resuming...", scored_so_far)
                existing_scores = existing_df[scored_mask][
                    ["query_id", "chunk_id", "relevance", "relevance_reason"]
                ].copy()
                existing_scores["relevance"] = existing_scores["relevance"].astype(int)
                df = df.merge(
                    existing_scores, on=["query_id", "chunk_id"], how="left"
                )
    except FileNotFoundError:
        pass

    if "relevance" not in df.columns:
        df["relevance"] = pd.NA
    if "relevance_reason" not in df.columns:
        df["relevance_reason"] = pd.NA

    # ── Identify pending rows ─────────────────────────────────────────────
    pending_mask = df["relevance"].isna()
    pending_indices = df.index[pending_mask].tolist()
    total_pending = len(pending_indices)

    logger.info("Pairs to judge: %d (skipping %d already scored)", total_pending, scored_so_far)

    if total_pending == 0:
        logger.info("Nothing to do — all rows already scored.")
        return

    # ── Run LLM judging with concurrent workers ──────────────────────────
    t0 = time.time()
    scored_count = 0
    score_dist = {0: 0, 1: 0, 2: 0, -1: 0}

    # Process in batches for checkpointing
    batch_size = BATCH_PARAMS["save_every"]
    batches = [
        pending_indices[i : i + batch_size]
        for i in range(0, total_pending, batch_size)
    ]

    for batch_num, batch_indices in enumerate(batches, start=1):
        logger.info(
            "Batch %d/%d (%d pairs)...",
            batch_num, len(batches), len(batch_indices),
        )

        # Build args for thread pool
        tasks = [
            (idx, client, str(df.at[idx, "query_text"]), str(df.at[idx, "chunk_text"]), logger)
            for idx in batch_indices
        ]

        with ThreadPoolExecutor(max_workers=LLM_PARAMS["concurrent_workers"]) as executor:
            futures = {
                executor.submit(judge_pair_wrapper, task): task[0]
                for task in tasks
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Batch {batch_num}",
                leave=False,
            ):
                try:
                    idx, relevance, reason = future.result()
                    df.at[idx, "relevance"] = relevance
                    df.at[idx, "relevance_reason"] = reason
                    score_dist[relevance] = score_dist.get(relevance, 0) + 1
                    scored_count += 1
                except Exception as e:
                    idx = futures[future]
                    logger.error("Future failed for index %d: %s", idx, e)
                    df.at[idx, "relevance"] = -1
                    df.at[idx, "relevance_reason"] = f"future_error: {str(e)[:100]}"
                    score_dist[-1] += 1
                    scored_count += 1

        # Checkpoint after each batch
        write_csv_to_gcs(gcs, output_path, df, logger)
        elapsed = time.time() - t0
        rate = scored_count / elapsed if elapsed > 0 else 0
        remaining = (total_pending - scored_count) / rate if rate > 0 else 0
        logger.info(
            "  Progress: %d/%d | Rate: %.1f pairs/sec | ETA: %.1f min",
            scored_count, total_pending, rate, remaining / 60,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("DONE — Summary")
    logger.info("=" * 60)
    logger.info("Total pairs judged:    %d", scored_count)
    logger.info("Time elapsed:          %.1f min", elapsed / 60)
    logger.info(
        "Throughput:            %.1f pairs/sec",
        scored_count / elapsed if elapsed > 0 else 0,
    )
    logger.info("")
    logger.info("Relevance distribution:")
    logger.info("  0 (not relevant):    %d (%.1f%%)",
                score_dist[0], 100 * score_dist[0] / max(scored_count, 1))
    logger.info("  1 (partial):         %d (%.1f%%)",
                score_dist[1], 100 * score_dist[1] / max(scored_count, 1))
    logger.info("  2 (highly relevant): %d (%.1f%%)",
                score_dist[2], 100 * score_dist[2] / max(scored_count, 1))
    logger.info("  -1 (failed):         %d (%.1f%%)",
                score_dist[-1], 100 * score_dist[-1] / max(scored_count, 1))
    logger.info("")
    logger.info(
        "Output: gs://%s/%s",
        GCS_PARAMS["bucket_name"], IO_PARAMS["output_gcs_path"],
    )

    # ── Cost estimate ─────────────────────────────────────────────────────
    # GPT-4o-mini: $0.15/1M input, $0.60/1M output
    est_input_tokens = scored_count * 500
    est_output_tokens = scored_count * 30
    est_cost = (est_input_tokens * 0.15 / 1_000_000) + (est_output_tokens * 0.60 / 1_000_000)
    logger.info("Estimated API cost:    $%.2f", est_cost)

    # ── Quality warning ───────────────────────────────────────────────────
    fail_rate = score_dist[-1] / max(scored_count, 1)
    if fail_rate > 0.05:
        logger.warning(
            "Failure rate is %.1f%% — review rows with relevance=-1.",
            fail_rate * 100,
        )


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()