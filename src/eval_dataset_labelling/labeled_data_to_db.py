"""
Golden Dataset Loader
======================
Reads the LLM-scored retrieval_labeling_dataset_scored.csv from GCS
and inserts relevant records (relevance 1 or 2) into the
evaluation_golden_dataset SQL table.

Usage:
    python insert_golden_dataset.py

Prerequisites:
    pip install psycopg2-binary pandas google-cloud-storage google-cloud-secret-manager

Author: InterviewPrep AI Team
"""

import io
import logging
import sys

import pandas as pd
import psycopg2
import psycopg2.extras

from src.storage.gcs_backend import GCSBackend


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

DB_PARAMS = {
    "host": "34.148.0.165",
    "port": 5432,
    "dbname": "interviewprep-ai-database",
    "user": "postgres",
    "password": "admin",
}

GCS_PARAMS = {
    "bucket_name": "interviewprep-ai-data",
    "project_id": "professorbot-dovbsg",
    "secret_name": "gcs-service-account-key",
}

IO_PARAMS = {
    "input_gcs_path": "eval_queries/labeled_dataset_264.csv",
}

TABLE_NAME = "eval_dataset"
BATCH_SIZE = 500                # rows per INSERT batch
LOG_LEVEL = "INFO"


# ═════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION
# ═════════════════════════════════════════════════════════════════════════════

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME}
    (query_id, query_text, relevant_chunk_id, relevant_doc_id, query_category, relevance, relevance_reason)
VALUES
    (%(query_id)s, %(query_text)s, %(relevant_chunk_id)s, %(relevant_doc_id)s,
     %(query_category)s, %(relevance)s, %(relevance_reason)s)
ON CONFLICT (query_id, relevant_chunk_id)
DO UPDATE SET
    relevance        = EXCLUDED.relevance,
    relevance_reason = EXCLUDED.relevance_reason,
    created_at       = NOW();
"""


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("golden_dataset_loader")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%H:%M:%S")
    )
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# ─── GCS CSV Helper ──────────────────────────────────────────────────────────

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


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Golden Dataset Loader")
    logger.info("=" * 60)

    # ── Initialize GCS ────────────────────────────────────────────────────
    logger.info("Initializing GCS backend...")
    gcs = GCSBackend(
        bucket_name=GCS_PARAMS["bucket_name"],
        project_id=GCS_PARAMS["project_id"],
        secret_name=GCS_PARAMS["secret_name"],
    )

    # ── Load scored dataset from GCS ──────────────────────────────────────
    df = read_csv_from_gcs(gcs, IO_PARAMS["input_gcs_path"], logger)

    # ── Validate required columns ─────────────────────────────────────────
    required = {"query_id", "query_text", "chunk_id", "document_id",
                "query_category", "relevance", "relevance_reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in scored CSV: {missing}")

    # ── Filter to relevant pairs only ─────────────────────────────────────
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")
    df = df.dropna(subset=["relevance"])
    df["relevance"] = df["relevance"].astype(int)

    total_after = len(df)

    # Log distribution
    dist = df["relevance"].value_counts().sort_index()
    for score, count in dist.items():
        logger.info("  relevance=%d: %d rows", score, count)

    if total_after == 0:
        logger.warning("No rows to insert. Exiting.")
        return

    # ── Prepare records ───────────────────────────────────────────────────
    records = []
    for _, row in df.iterrows():
        records.append({
            "query_id": str(row["query_id"]),
            "query_text": str(row["query_text"]),
            "relevant_chunk_id": str(row["chunk_id"]),
            "relevant_doc_id": str(row["document_id"]),
            "query_category": str(row["query_category"]),
            "relevance": int(row["relevance"]),
            "relevance_reason": str(row.get("relevance_reason", "")),
        })

    # ── Connect to DB ─────────────────────────────────────────────────────
    logger.info(
        "Connecting to PostgreSQL at %s:%s/%s",
        DB_PARAMS["host"], DB_PARAMS["port"], DB_PARAMS["dbname"],
    )
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # ── Batch insert ──────────────────────────────────────────────────────
    total_inserted = 0
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = records[start:end]

        try:
            psycopg2.extras.execute_batch(cur, INSERT_SQL, batch, page_size=BATCH_SIZE)
            conn.commit()
            total_inserted += len(batch)
            logger.info(
                "  Batch %d/%d: inserted %d rows (%d/%d total)",
                batch_num + 1, total_batches, len(batch), total_inserted, len(records),
            )
        except Exception as e:
            conn.rollback()
            logger.error("Batch %d failed: %s", batch_num + 1, e)
            # Try row-by-row for this batch to identify bad records
            row_errors = 0
            for record in batch:
                try:
                    cur.execute(INSERT_SQL, record)
                    conn.commit()
                    total_inserted += 1
                except Exception as row_e:
                    conn.rollback()
                    row_errors += 1
                    logger.warning(
                        "  Skipped query_id=%s chunk_id=%s: %s",
                        record["query_id"], record["relevant_chunk_id"], row_e,
                    )
            if row_errors:
                logger.warning("  %d rows failed in batch %d", row_errors, batch_num + 1)

    # ── Verify ────────────────────────────────────────────────────────────
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    db_count = cur.fetchone()[0]

    cur.execute(
        f"SELECT relevance, COUNT(*) FROM {TABLE_NAME} GROUP BY relevance ORDER BY relevance"
    )
    db_dist = cur.fetchall()

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DONE — Summary")
    logger.info("=" * 60)
    logger.info("Records inserted/updated: %d", total_inserted)
    logger.info("Total rows in table:      %d", db_count)
    logger.info("")
    logger.info("Table distribution:")
    for score, count in db_dist:
        logger.info("  relevance=%d: %d rows", score, count)

    # ── Unique query coverage ─────────────────────────────────────────────
    cur.execute(f"SELECT COUNT(DISTINCT query_id) FROM {TABLE_NAME}")
    unique_queries = cur.fetchone()[0]

    cur.execute(
        f"SELECT query_category, COUNT(DISTINCT query_id) FROM {TABLE_NAME} "
        f"GROUP BY query_category ORDER BY COUNT(DISTINCT query_id) DESC"
    )
    cat_dist = cur.fetchall()

    logger.info("")
    logger.info("Unique queries covered:   %d", unique_queries)
    logger.info("By category:")
    for cat, count in cat_dist:
        logger.info("  %-30s %d queries", cat, count)

    cur.close()
    conn.close()
    logger.info("Connection closed.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()