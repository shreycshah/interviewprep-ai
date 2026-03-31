"""
Gold Standard Evaluation Dataset Generator
===========================================
Generates a labeling dataset for retriever evaluation by running
three retrieval strategies (Vector, BM25, Hybrid) against a
PostgreSQL+pgvector database and pooling the results.

Each unique chunk per query appears ONCE in the output, with a
`sources` column listing all retrieval methods that found it
(e.g. "vector,hybrid") and per-source scores + ranks preserved.

Reads queries CSV from GCS and writes output CSV back to GCS
using the project's GCSBackend class.

Usage:
    python generate_labeling_dataset.py

Prerequisites:
    pip install psycopg2-binary pgvector sentence-transformers rank-bm25 pandas numpy tqdm
    pip install google-cloud-storage google-cloud-secret-manager

Author: InterviewPrep AI Team
"""

import csv
import io
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.storage.gcs_backend import GCSBackend


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these values to match your environment
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

RETRIEVAL_PARAMS = {
    "top_k": 20,                         # chunks per retrieval method
    "hybrid_alpha": 0.5,                 # 0 = pure BM25, 1 = pure vector
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "query_batch_size": 32,              # batch size for encoding queries
}

BM25_PARAMS = {
    "k1": 1.5,
    "b": 0.75,
}

IO_PARAMS = {
    "queries_gcs_path": "eval_queries/queries.csv", # input from GCS
    "output_gcs_path": "eval_queries/retrieval_labeling_dataset.csv", # output to GCS
}

CORPUS_PARAMS = {
    "table_name": "document_chunks",
    "embedding_column": "embeddings_all_minilm_l6_v2",
    "fetch_batch_size": 5000,            # rows per DB cursor fetch
}

SEED = 42
LOG_LEVEL = "INFO"


# ═════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION — No need to edit below this line
# ═════════════════════════════════════════════════════════════════════════════


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("eval_dataset_gen")
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
    """Download a CSV from GCS and return as a DataFrame."""
    logger.info("Reading CSV from gs://%s/%s", gcs.bucket_name, gcs_path)
    blob = gcs.bucket.blob(gcs_path)
    if not blob.exists():
        raise FileNotFoundError(
            f"CSV not found at gs://{gcs.bucket_name}/{gcs_path}"
        )
    csv_text = blob.download_as_text(encoding="utf-8")
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = df.columns.str.strip()
    logger.info("Read %d rows from GCS.", len(df))
    return df


def write_csv_to_gcs(
    gcs: GCSBackend,
    gcs_path: str,
    rows: list[dict],
    fieldnames: list[str],
    logger: logging.Logger,
) -> None:
    """Write a list of dicts as CSV to GCS."""
    logger.info("Writing %d rows to gs://%s/%s", len(rows), gcs.bucket_name, gcs_path)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    blob = gcs.bucket.blob(gcs_path)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")
    logger.info("Upload complete.")


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Query:
    query_id: str
    query_text: str
    query_category: str


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    chunk_text: str
    raw_text: str
    word_count: int
    char_start_offset: int
    char_end_offset: int
    strategy: Optional[str]
    round_label: Optional[str]
    embedding: Optional[np.ndarray] = None


@dataclass
class PooledChunkResult:
    """A single chunk's aggregated retrieval info across all sources for one query."""
    query_id: str
    query_text: str
    query_category: str
    chunk_id: str
    document_id: str
    chunk_text: str
    # Per-source tracking
    sources: list[str] = field(default_factory=list)
    vector_rank: Optional[int] = None
    vector_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    hybrid_rank: Optional[int] = None
    hybrid_score: Optional[float] = None
    # Best score across all sources (for sorting)
    best_score: float = 0.0


# ─── Database Layer ───────────────────────────────────────────────────────────

class ChunkStore:
    """Handles all PostgreSQL + pgvector interactions."""

    COLUMNS = [
        "chunk_id", "document_id", "chunk_index", "total_chunks",
        "chunk_text", "raw_text", "word_count",
        "char_start_offset", "char_end_offset",
        "strategy", "round_label", CORPUS_PARAMS["embedding_column"],
    ]

    def __init__(self, logger: logging.Logger):
        self.log = logger
        self.conn = None

    def connect(self):
        self.log.info(
            "Connecting to PostgreSQL at %s:%s/%s",
            DB_PARAMS["host"], DB_PARAMS["port"], DB_PARAMS["dbname"],
        )
        self.conn = psycopg2.connect(**DB_PARAMS)
        register_vector(self.conn)
        self.log.info("Connected and pgvector registered.")

    def close(self):
        if self.conn:
            self.conn.close()

    def fetch_all_chunks(self) -> list[Chunk]:
        """Load entire corpus into memory for BM25 indexing."""
        table = CORPUS_PARAMS["table_name"]
        emb_col = CORPUS_PARAMS["embedding_column"]

        self.log.info("Fetching all chunks from %s...", table)
        chunks = []
        with self.conn.cursor(
            name="chunk_cursor", cursor_factory=psycopg2.extras.DictCursor
        ) as cur:
            cur.itersize = CORPUS_PARAMS["fetch_batch_size"]
            cur.execute(f"SELECT {', '.join(self.COLUMNS)} FROM {table}")
            for row in cur:
                emb = row[emb_col]
                chunks.append(Chunk(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    chunk_index=row["chunk_index"],
                    total_chunks=row["total_chunks"],
                    chunk_text=row["chunk_text"] or "",
                    raw_text=row["raw_text"] or "",
                    word_count=row["word_count"] or 0,
                    char_start_offset=row["char_start_offset"] or 0,
                    char_end_offset=row["char_end_offset"] or 0,
                    strategy=row["strategy"],
                    round_label=row["round_label"],
                    embedding=np.array(emb, dtype=np.float32) if emb is not None else None,
                ))
        self.log.info("Loaded %d chunks.", len(chunks))
        return chunks

    def vector_search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """
        pgvector cosine distance search.
        Returns list of (chunk_id, cosine_similarity) sorted descending.
        """
        table = CORPUS_PARAMS["table_name"]
        emb_col = CORPUS_PARAMS["embedding_column"]

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id,
                       1 - ({emb_col} <=> %s::vector) AS similarity
                FROM {table}
                WHERE {emb_col} IS NOT NULL
                ORDER BY {emb_col} <=> %s::vector
                LIMIT %s
                """,
                (query_embedding.tolist(), query_embedding.tolist(), top_k),
            )
            return [(str(row[0]), float(row[1])) for row in cur.fetchall()]


# ─── BM25 Index ───────────────────────────────────────────────────────────────

class BM25Index:
    """In-memory BM25 index over the chunk corpus."""

    def __init__(self, chunks: list[Chunk], logger: logging.Logger):
        self.log = logger
        self.chunks = chunks
        self.chunk_id_list = [c.chunk_id for c in chunks]

        self.log.info("Building BM25 index over %d chunks...", len(chunks))
        tokenized = [self._tokenize(c.chunk_text) for c in chunks]
        self.index = BM25Okapi(
            tokenized, k1=BM25_PARAMS["k1"], b=BM25_PARAMS["b"]
        )
        self.log.info("BM25 index ready.")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, bm25_score) sorted descending."""
        tokens = self._tokenize(query)
        scores = self.index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.chunk_id_list[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


# ─── Score Normalization ──────────────────────────────────────────────────────

def normalize_scores(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Min-max normalize scores to [0, 1]. Handles edge cases."""
    if not results:
        return results

    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    spread = max_s - min_s

    if spread == 0:
        return [(cid, 1.0) for cid, _ in results]

    return [(cid, (s - min_s) / spread) for cid, s in results]


# ─── Hybrid Fusion ────────────────────────────────────────────────────────────

def hybrid_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    alpha: float,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Weighted linear combination of normalized scores.
    alpha = weight for vector, (1 - alpha) = weight for BM25.
    """
    vec_norm = dict(normalize_scores(vector_results))
    bm25_norm = dict(normalize_scores(bm25_results))

    all_chunk_ids = set(vec_norm.keys()) | set(bm25_norm.keys())

    combined = []
    for cid in all_chunk_ids:
        v_score = vec_norm.get(cid, 0.0)
        b_score = bm25_norm.get(cid, 0.0)
        hybrid_score = alpha * v_score + (1 - alpha) * b_score
        combined.append((cid, hybrid_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]


# ─── Query Loader ─────────────────────────────────────────────────────────────

def load_queries(gcs: GCSBackend, logger: logging.Logger) -> list[Query]:
    """Load queries from CSV on GCS: query_id, query_text, query_category."""
    df = read_csv_from_gcs(gcs, IO_PARAMS["queries_gcs_path"], logger)

    required = {"query_id", "query_text", "query_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in queries CSV: {missing}")

    queries = [
        Query(
            query_id=str(row["query_id"]),
            query_text=str(row["query_text"]).strip(),
            query_category=str(row["query_category"]).strip(),
        )
        for _, row in df.iterrows()
    ]
    logger.info(
        "Loaded %d queries across %d categories.",
        len(queries), df["query_category"].nunique(),
    )
    return queries


# ─── Pooling Logic ────────────────────────────────────────────────────────────

def pool_results_for_query(
    query: Query,
    vec_normed: list[tuple[str, float]],
    bm25_normed: list[tuple[str, float]],
    hybrid_normed: list[tuple[str, float]],
    chunk_lookup: dict[str, Chunk],
    logger: logging.Logger,
) -> list[PooledChunkResult]:
    """
    Merge results from all three sources into a single deduplicated list.
    Each chunk appears once with all source attributions preserved.
    """
    pool: dict[str, PooledChunkResult] = {}

    def _merge(source: str, results: list[tuple[str, float]]):
        for rank, (chunk_id, score) in enumerate(results, start=1):
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                logger.warning(
                    "chunk_id %s from %s not in lookup (query %s)",
                    chunk_id, source, query.query_id,
                )
                continue

            if chunk_id not in pool:
                pool[chunk_id] = PooledChunkResult(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    query_category=query.query_category,
                    chunk_id=chunk_id,
                    document_id=chunk.document_id,
                    chunk_text=chunk.chunk_text,
                )

            entry = pool[chunk_id]
            entry.sources.append(source)
            entry.best_score = max(entry.best_score, score)

            if source == "vector":
                entry.vector_rank = rank
                entry.vector_score = round(score, 6)
            elif source == "BM25":
                entry.bm25_rank = rank
                entry.bm25_score = round(score, 6)
            elif source == "hybrid":
                entry.hybrid_rank = rank
                entry.hybrid_score = round(score, 6)

    _merge("vector", vec_normed)
    _merge("BM25", bm25_normed)
    _merge("hybrid", hybrid_normed)

    # Sort pooled results by best score descending
    pooled = sorted(pool.values(), key=lambda x: x.best_score, reverse=True)
    return pooled


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def build_chunk_lookup(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def run_pipeline():
    logger = setup_logging()
    np.random.seed(SEED)

    top_k = RETRIEVAL_PARAMS["top_k"]
    alpha = RETRIEVAL_PARAMS["hybrid_alpha"]

    logger.info("=" * 60)
    logger.info("Gold Standard Evaluation Dataset Generator")
    logger.info("=" * 60)
    logger.info(
        "Config: top_k=%d, hybrid_alpha=%.2f, model=%s",
        top_k, alpha, RETRIEVAL_PARAMS["embedding_model"],
    )

    # ── Initialize GCS ────────────────────────────────────────────────────
    logger.info("Initializing GCS backend...")
    gcs = GCSBackend(
        bucket_name=GCS_PARAMS["bucket_name"],
        project_id=GCS_PARAMS["project_id"],
        secret_name=GCS_PARAMS["secret_name"],
    )

    # ── Load queries from GCS ─────────────────────────────────────────────
    queries = load_queries(gcs, logger)

    # ── Connect to DB ─────────────────────────────────────────────────────
    store = ChunkStore(logger)
    store.connect()

    # ── Load corpus into memory (needed for BM25) ────────────────────────
    t0 = time.time()
    chunks = store.fetch_all_chunks()
    chunk_lookup = build_chunk_lookup(chunks)
    logger.info("Corpus loaded in %.1fs", time.time() - t0)

    # ── Build BM25 index ──────────────────────────────────────────────────
    t0 = time.time()
    bm25 = BM25Index(chunks, logger)
    logger.info("BM25 index built in %.1fs", time.time() - t0)

    # ── Load embedding model ──────────────────────────────────────────────
    logger.info("Loading embedding model: %s", RETRIEVAL_PARAMS["embedding_model"])
    model = SentenceTransformer(RETRIEVAL_PARAMS["embedding_model"])
    logger.info("Embedding model ready.")

    # ── Encode all queries ────────────────────────────────────────────────
    logger.info("Encoding %d queries...", len(queries))
    query_texts = [q.query_text for q in queries]
    query_embeddings = model.encode(
        query_texts,
        batch_size=RETRIEVAL_PARAMS["query_batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # ── Run retrieval + pooling for each query ────────────────────────────
    all_results: list[PooledChunkResult] = []
    stats = {"vector": 0, "BM25": 0, "hybrid": 0}
    overlap_counts = {"vector_only": 0, "bm25_only": 0, "hybrid_only": 0, "multi_source": 0}

    for i, query in enumerate(tqdm(queries, desc="Retrieving")):
        q_emb = query_embeddings[i]

        # 1) Vector search via pgvector
        try:
            vec_raw = store.vector_search(q_emb, top_k)
        except Exception as e:
            logger.warning("Vector search failed for query %s: %s", query.query_id, e)
            vec_raw = []

        # 2) BM25 search
        try:
            bm25_raw = bm25.search(query.query_text, top_k)
        except Exception as e:
            logger.warning("BM25 search failed for query %s: %s", query.query_id, e)
            bm25_raw = []

        # 3) Hybrid fusion
        try:
            hybrid_raw = hybrid_fusion(vec_raw, bm25_raw, alpha, top_k)
        except Exception as e:
            logger.warning("Hybrid fusion failed for query %s: %s", query.query_id, e)
            hybrid_raw = []

        # Normalize scores per source
        vec_normed = normalize_scores(vec_raw)
        bm25_normed = normalize_scores(bm25_raw)
        hybrid_normed = normalize_scores(hybrid_raw)

        # Pool and merge
        pooled = pool_results_for_query(
            query, vec_normed, bm25_normed, hybrid_normed, chunk_lookup, logger
        )
        all_results.extend(pooled)

        # Track source overlap stats
        for entry in pooled:
            for s in entry.sources:
                stats[s] += 1
            if len(entry.sources) > 1:
                overlap_counts["multi_source"] += 1
            elif "vector" in entry.sources:
                overlap_counts["vector_only"] += 1
            elif "BM25" in entry.sources:
                overlap_counts["bm25_only"] += 1
            elif "hybrid" in entry.sources:
                overlap_counts["hybrid_only"] += 1

    # ── Write output CSV to GCS ───────────────────────────────────────────
    output_fields = [
        "query_id", "query_text", "query_category",
        "chunk_id", "document_id", "chunk_text",
        "sources",
        "vector_rank", "vector_score",
        "bm25_rank", "bm25_score",
        "hybrid_rank", "hybrid_score",
        "best_score",
    ]

    output_rows = []
    for r in all_results:
        output_rows.append({
            "query_id": r.query_id,
            "query_text": r.query_text,
            "query_category": r.query_category,
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "chunk_text": r.chunk_text,
            "sources": ",".join(r.sources),
            "vector_rank": r.vector_rank if r.vector_rank is not None else "",
            "vector_score": r.vector_score if r.vector_score is not None else "",
            "bm25_rank": r.bm25_rank if r.bm25_rank is not None else "",
            "bm25_score": r.bm25_score if r.bm25_score is not None else "",
            "hybrid_rank": r.hybrid_rank if r.hybrid_rank is not None else "",
            "hybrid_score": r.hybrid_score if r.hybrid_score is not None else "",
            "best_score": round(r.best_score, 6),
        })

    write_csv_to_gcs(gcs, IO_PARAMS["output_gcs_path"], output_rows, output_fields, logger)

    # ── Summary ───────────────────────────────────────────────────────────
    num_queries = len(queries)
    total_rows = len(all_results)
    avg_unique = total_rows / max(num_queries, 1)

    logger.info("=" * 60)
    logger.info("DONE — Summary")
    logger.info("=" * 60)
    logger.info("Queries processed:       %d", num_queries)
    logger.info("Total unique chunk rows: %d", total_rows)
    logger.info("Avg unique chunks/query: %.1f", avg_unique)
    logger.info("")
    logger.info("Source hit counts (a chunk can count for multiple):")
    logger.info("  - vector:              %d", stats["vector"])
    logger.info("  - BM25:                %d", stats["BM25"])
    logger.info("  - hybrid:              %d", stats["hybrid"])
    logger.info("")
    logger.info("Overlap analysis:")
    logger.info("  - vector only:         %d", overlap_counts["vector_only"])
    logger.info("  - BM25 only:           %d", overlap_counts["bm25_only"])
    logger.info("  - hybrid only:         %d", overlap_counts["hybrid_only"])
    logger.info("  - multi-source:        %d (%.1f%%)",
                overlap_counts["multi_source"],
                100 * overlap_counts["multi_source"] / max(total_rows, 1))
    logger.info("")
    logger.info(
        "Output: gs://%s/%s",
        GCS_PARAMS["bucket_name"], IO_PARAMS["output_gcs_path"],
    )


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()