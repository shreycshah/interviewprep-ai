"""
Tests for src/eval_dataset_labelling/dataset_generator.py

Covers:
- Query / Chunk / PooledChunkResult dataclass defaults
- normalize_scores(): edge cases (empty, single, zero-spread, normal)
- hybrid_fusion(): alpha weights, top_k truncation
- build_chunk_lookup(): dict keyed by chunk_id
- BM25Index: tokenize, search ranking
- pool_results_for_query(): single-source, multi-source dedup, missing chunk

All DB, SentenceTransformer, and GCS interactions are mocked.

Skipped if pandas or rank_bm25 is not installed.
"""
import pytest

pytest.importorskip("pandas", reason="pandas not installed")
pytest.importorskip("rank_bm25", reason="rank_bm25 not installed")
pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed")
pytest.importorskip("pgvector", reason="pgvector not installed")

import numpy as np
from unittest.mock import MagicMock

from src.eval_dataset_labelling.dataset_generator import (
    Query,
    Chunk,
    PooledChunkResult,
    normalize_scores,
    hybrid_fusion,
    build_chunk_lookup,
    pool_results_for_query,
    BM25Index,
)


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

def _make_chunk(chunk_id="c1", text="chunk text"):
    return Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        total_chunks=1,
        chunk_text=text,
        raw_text=text,
        word_count=len(text.split()),
        char_start_offset=0,
        char_end_offset=len(text),
        strategy="single_chunk",
        round_label=None,
    )


def _make_query(query_id="q1"):
    return Query(
        query_id=query_id,
        query_text="tell me about Google interview",
        query_category="company_specific",
    )


# ─────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────

class TestDataclasses:
    def test_query_fields(self):
        q = Query(query_id="q1", query_text="text", query_category="cat")
        assert q.query_id == "q1"
        assert q.query_text == "text"
        assert q.query_category == "cat"

    def test_chunk_optional_defaults(self):
        c = _make_chunk()
        assert c.embedding is None

    def test_pooled_chunk_result_defaults(self):
        p = PooledChunkResult(
            query_id="q1",
            query_text="text",
            query_category="cat",
            chunk_id="c1",
            document_id="d1",
            chunk_text="text",
        )
        assert p.sources == []
        assert p.vector_rank is None
        assert p.bm25_rank is None
        assert p.hybrid_rank is None
        assert p.best_score == 0.0


# ─────────────────────────────────────────────────
# normalize_scores()
# ─────────────────────────────────────────────────

class TestNormalizeScores:
    def test_empty_list(self):
        assert normalize_scores([]) == []

    def test_single_item(self):
        result = normalize_scores([("c1", 0.5)])
        assert result == [("c1", 1.0)]

    def test_zero_spread(self):
        result = normalize_scores([("c1", 0.5), ("c2", 0.5)])
        assert all(s == 1.0 for _, s in result)

    def test_normal_spread(self):
        result = normalize_scores([("c1", 0.0), ("c2", 0.5), ("c3", 1.0)])
        scores = {cid: s for cid, s in result}
        assert scores["c1"] == pytest.approx(0.0)
        assert scores["c2"] == pytest.approx(0.5)
        assert scores["c3"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────
# hybrid_fusion()
# ─────────────────────────────────────────────────

class TestHybridFusion:
    def test_alpha_one_pure_vector(self):
        vec = [("c1", 1.0), ("c2", 0.5)]
        bm25 = [("c3", 1.0)]
        result = hybrid_fusion(vec, bm25, alpha=1.0, top_k=10)
        ids = [cid for cid, _ in result]
        assert "c1" in ids
        assert "c2" in ids

    def test_alpha_zero_pure_bm25(self):
        vec = [("c1", 1.0)]
        bm25 = [("c2", 1.0), ("c3", 0.5)]
        result = hybrid_fusion(vec, bm25, alpha=0.0, top_k=10)
        ids = [cid for cid, _ in result]
        assert "c2" in ids
        assert "c3" in ids

    def test_top_k_truncation(self):
        vec = [("c1", 1.0), ("c2", 0.8)]
        bm25 = [("c3", 1.0), ("c4", 0.8)]
        result = hybrid_fusion(vec, bm25, alpha=0.5, top_k=2)
        assert len(result) == 2

    def test_mixed_alpha(self):
        vec = [("c1", 1.0)]
        bm25 = [("c1", 1.0)]
        result = hybrid_fusion(vec, bm25, alpha=0.5, top_k=5)
        assert len(result) == 1
        assert result[0][1] == pytest.approx(1.0)

    def test_empty_inputs(self):
        result = hybrid_fusion([], [], alpha=0.5, top_k=10)
        assert result == []


# ─────────────────────────────────────────────────
# build_chunk_lookup()
# ─────────────────────────────────────────────────

class TestBuildChunkLookup:
    def test_keyed_by_chunk_id(self):
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        lookup = build_chunk_lookup(chunks)
        assert "c1" in lookup
        assert "c2" in lookup
        assert lookup["c1"].chunk_id == "c1"

    def test_empty_list(self):
        assert build_chunk_lookup([]) == {}


# ─────────────────────────────────────────────────
# BM25Index
# ─────────────────────────────────────────────────

class TestBM25Index:
    def test_tokenize_lowercase_split(self):
        tokens = BM25Index._tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]

    def test_search_returns_ranked_results(self):
        chunks = [
            _make_chunk("c1", "google interview system design round"),
            _make_chunk("c2", "amazon behavioral leadership principles"),
            _make_chunk("c3", "google phone screen coding round"),
        ]
        logger = MagicMock()
        idx = BM25Index(chunks, logger)
        results = idx.search("google interview", top_k=5)
        ids = [cid for cid, _ in results]
        assert "c1" in ids or "c3" in ids

    def test_search_respects_top_k(self):
        chunks = [_make_chunk(f"c{i}", f"word{i} interview") for i in range(20)]
        logger = MagicMock()
        idx = BM25Index(chunks, logger)
        results = idx.search("interview", top_k=3)
        assert len(results) <= 3

    def test_search_no_match_returns_empty(self):
        chunks = [_make_chunk("c1", "hello world")]
        logger = MagicMock()
        idx = BM25Index(chunks, logger)
        results = idx.search("xyznonexistent", top_k=5)
        assert results == []


# ─────────────────────────────────────────────────
# pool_results_for_query()
# ─────────────────────────────────────────────────

class TestPoolResultsForQuery:
    def test_single_source(self):
        query = _make_query()
        chunk = _make_chunk("c1")
        lookup = {"c1": chunk}
        logger = MagicMock()

        pooled = pool_results_for_query(
            query,
            vec_normed=[("c1", 0.9)],
            bm25_normed=[],
            hybrid_normed=[],
            chunk_lookup=lookup,
            logger=logger,
        )
        assert len(pooled) == 1
        assert pooled[0].sources == ["vector"]
        assert pooled[0].vector_rank == 1
        assert pooled[0].vector_score == pytest.approx(0.9, abs=1e-5)

    def test_multi_source_dedup(self):
        query = _make_query()
        chunk = _make_chunk("c1")
        lookup = {"c1": chunk}
        logger = MagicMock()

        pooled = pool_results_for_query(
            query,
            vec_normed=[("c1", 0.9)],
            bm25_normed=[("c1", 0.7)],
            hybrid_normed=[("c1", 0.8)],
            chunk_lookup=lookup,
            logger=logger,
        )
        assert len(pooled) == 1
        assert set(pooled[0].sources) == {"vector", "BM25", "hybrid"}

    def test_missing_chunk_in_lookup_skipped(self):
        query = _make_query()
        lookup = {}
        logger = MagicMock()

        pooled = pool_results_for_query(
            query,
            vec_normed=[("missing_id", 0.9)],
            bm25_normed=[],
            hybrid_normed=[],
            chunk_lookup=lookup,
            logger=logger,
        )
        assert len(pooled) == 0
        logger.warning.assert_called()

    def test_sorted_by_best_score_descending(self):
        query = _make_query()
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")
        lookup = {"c1": c1, "c2": c2}
        logger = MagicMock()

        pooled = pool_results_for_query(
            query,
            vec_normed=[("c1", 0.3), ("c2", 0.9)],
            bm25_normed=[],
            hybrid_normed=[],
            chunk_lookup=lookup,
            logger=logger,
        )
        assert pooled[0].chunk_id == "c2"
        assert pooled[1].chunk_id == "c1"
