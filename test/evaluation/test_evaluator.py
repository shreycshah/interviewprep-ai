"""
Tests for src/evaluation/evaluator.py

Covers:
- Dataclasses: ModelConfig, BM25Config, HybridConfig, EvalQuery, RetrievalResult, RetrievalMode
- Config loader: load_eval_config()
- EvalDatasetLoader: grouping and filtering by threshold
- MetricsCalculator: mrr_at_k, recall_at_k, precision_at_k, ndcg_at_k, score_distribution, bias_report
- compute_selection_score(): weighted composite
- Retrieval strategies: VectorStrategy.embedding_column_name, build_strategy dispatch
- HybridStrategy: RRF fusion
- Decision gate: _apply_decision_gate

All DB, SentenceTransformer, and MLflow interactions are mocked.

Skipped if sentence_transformers or mlflow is not installed.
"""
import pytest

pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed")
pytest.importorskip("mlflow", reason="mlflow not installed")

import numpy as np
from unittest.mock import MagicMock, patch

from src.evaluation.evaluator import (
    EvalQuery,
    RetrievalResult,
    RetrievalMode,
    BM25Config,
    HybridConfig,
    ModelConfig,
    load_eval_config,
    EvalDatasetLoader,
    MetricsCalculator,
    compute_selection_score,
    VectorStrategy,
    BM25Strategy,
    HybridStrategy,
    build_strategy,
)


# ── Helpers ───────────────────────────────────────

def _make_query(query_id=1, grades=None, relevant_ids=None, category="general"):
    grades = grades or {"c1": 2, "c2": 1, "c3": 0}
    relevant_ids = relevant_ids or ["c1", "c2"]
    return EvalQuery(
        query_id=query_id,
        query_text=f"query {query_id}",
        category=category,
        relevance_grades=grades,
        relevant_chunk_ids=relevant_ids,
        relevant_doc_ids=["d1"],
    )


def _make_result(query_id=1, chunk_ids=None, scores=None):
    chunk_ids = chunk_ids or ["c1", "c2", "c3"]
    scores = scores or [0.9, 0.5, 0.1]
    return RetrievalResult(query_id=query_id, retrieved_chunk_ids=chunk_ids, scores=scores)


def _make_config(**overrides):
    defaults = {
        "model_name": "test-model",
        "embedding_dim": 384,
        "chunk_size": 512,
        "overlap_size": 64,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ── Dataclasses ───────────────────────────────────

class TestModelConfig:
    def test_run_name_vector(self):
        cfg = _make_config(retrieval_mode=RetrievalMode.VECTOR)
        assert "vector" in cfg.run_name
        assert "test-model" in cfg.run_name

    def test_run_name_hybrid(self):
        cfg = _make_config(retrieval_mode=RetrievalMode.HYBRID)
        assert "hybrid" in cfg.run_name

    def test_model_name_column_value_defaults(self):
        cfg = _make_config()
        assert cfg.model_name_column_value == "test-model"

    def test_to_params_dict_base(self):
        cfg = _make_config()
        params = cfg.to_params_dict()
        assert params["model_name"] == "test-model"
        assert params["embedding_dim"] == 384
        assert "rrf_k" not in params

    def test_to_params_dict_hybrid_includes_rrf(self):
        cfg = _make_config(retrieval_mode=RetrievalMode.HYBRID)
        params = cfg.to_params_dict()
        assert "rrf_k" in params
        assert "vector_weight" in params


class TestRetrievalMode:
    def test_enum_values(self):
        assert RetrievalMode.VECTOR == "vector"
        assert RetrievalMode.BM25 == "bm25"
        assert RetrievalMode.HYBRID == "hybrid"


class TestBM25Config:
    def test_defaults(self):
        cfg = BM25Config()
        assert cfg.search_column == "chunk_text"
        assert cfg.table == "document_chunks"


class TestHybridConfig:
    def test_defaults(self):
        cfg = HybridConfig()
        assert cfg.rrf_k == 60
        assert cfg.vector_weight == 0.5
        assert cfg.bm25_weight == 0.5


# ── Config Loader ─────────────────────────────────

class TestLoadEvalConfig:
    def test_loads_configs_from_yaml(self):
        result = load_eval_config()
        assert "configs" in result
        assert "max_k" in result
        assert "relevance_threshold" in result
        assert len(result["configs"]) >= 1

    def test_configs_are_model_config_instances(self):
        result = load_eval_config()
        for cfg in result["configs"]:
            assert isinstance(cfg, ModelConfig)

    def test_mlflow_tracking_uri_present(self):
        result = load_eval_config()
        assert "mlflow_tracking_uri" in result


# ── EvalDatasetLoader ─────────────────────────────

class TestEvalDatasetLoader:
    def test_groups_by_query_id(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            (1, "query 1", "c1", "d1", "general", 2),
            (1, "query 1", "c2", "d1", "general", 1),
            (2, "query 2", "c3", "d2", "company", 0),
        ]
        mock_conn.cursor.return_value = mock_cursor

        loader = EvalDatasetLoader(mock_conn, relevance_threshold=1)
        queries = loader.load()

        assert len(queries) == 2
        q1 = next(q for q in queries if q.query_id == 1)
        assert len(q1.relevance_grades) == 2
        assert q1.relevant_chunk_ids == ["c1", "c2"]

    def test_threshold_filters_relevance(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            (1, "query", "c1", "d1", "general", 2),
            (1, "query", "c2", "d1", "general", 0),
        ]
        mock_conn.cursor.return_value = mock_cursor

        loader = EvalDatasetLoader(mock_conn, relevance_threshold=2)
        queries = loader.load()

        q = queries[0]
        assert q.relevant_chunk_ids == ["c1"]

    def test_empty_result(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        loader = EvalDatasetLoader(mock_conn)
        assert loader.load() == []


# ── MetricsCalculator ─────────────────────────────

class TestMetricsCalculator:
    def setup_method(self):
        self.calc = MetricsCalculator(relevance_threshold=1)
        self.query = _make_query()
        self.qmap = {self.query.query_id: self.query}

    def test_mrr_at_k_first_hit(self):
        result = _make_result(chunk_ids=["c1", "c3", "c2"])
        mrr = self.calc.mrr_at_k([result], self.qmap, k=10)
        assert mrr == pytest.approx(1.0)

    def test_mrr_at_k_second_hit(self):
        result = _make_result(chunk_ids=["c3", "c1", "c2"])
        mrr = self.calc.mrr_at_k([result], self.qmap, k=10)
        assert mrr == pytest.approx(0.5)

    def test_mrr_at_k_no_hit(self):
        result = _make_result(chunk_ids=["c3", "c99"])
        mrr = self.calc.mrr_at_k([result], self.qmap, k=10)
        assert mrr == pytest.approx(0.0)

    def test_recall_at_k_full(self):
        result = _make_result(chunk_ids=["c1", "c2", "c3"])
        recall = self.calc.recall_at_k([result], self.qmap, k=10)
        assert recall == pytest.approx(1.0)

    def test_recall_at_k_partial(self):
        result = _make_result(chunk_ids=["c1", "c3"])
        recall = self.calc.recall_at_k([result], self.qmap, k=10)
        assert recall == pytest.approx(0.5)

    def test_precision_at_k(self):
        result = _make_result(chunk_ids=["c1", "c2", "c3"])
        precision = self.calc.precision_at_k([result], self.qmap, k=3)
        assert precision == pytest.approx(2 / 3)

    def test_ndcg_at_k_perfect(self):
        result = _make_result(chunk_ids=["c1", "c2", "c3"])
        ndcg = self.calc.ndcg_at_k([result], self.qmap, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_at_k_reversed(self):
        result = _make_result(chunk_ids=["c3", "c2", "c1"])
        ndcg = self.calc.ndcg_at_k([result], self.qmap, k=3)
        assert ndcg < 1.0

    def test_empty_results(self):
        assert self.calc.mrr_at_k([], self.qmap, 10) == 0.0
        assert self.calc.recall_at_k([], self.qmap, 10) == 0.0
        assert self.calc.precision_at_k([], self.qmap, 10) == 0.0
        assert self.calc.ndcg_at_k([], self.qmap, 10) == 0.0

    def test_score_distribution(self):
        result = _make_result(chunk_ids=["c1", "c2", "c3"], scores=[0.9, 0.5, 0.1])
        flat, buckets = self.calc.score_distribution_by_grade([result], self.qmap)
        assert len(buckets["grade_2"]) == 1
        assert len(buckets["grade_1"]) == 1
        assert len(buckets["grade_0"]) == 1


class TestPerCategoryMetrics:
    def test_splits_by_category(self):
        calc = MetricsCalculator(relevance_threshold=1)
        q1 = _make_query(query_id=1, category="general")
        q2 = _make_query(query_id=2, category="company")
        qmap = {1: q1, 2: q2}
        r1 = _make_result(query_id=1, chunk_ids=["c1"])
        r2 = _make_result(query_id=2, chunk_ids=["c1"])
        breakdown = calc.per_category_metrics([r1, r2], qmap)
        assert "general" in breakdown
        assert "company" in breakdown


class TestBiasReport:
    def test_two_categories(self):
        calc = MetricsCalculator(relevance_threshold=1)
        breakdown = {
            "general": {"count": 5, "ndcg@10": 0.8, "recall@10": 0.7, "mrr@10": 0.6, "precision@10": 0.5},
            "company": {"count": 5, "ndcg@10": 0.5, "recall@10": 0.4, "mrr@10": 0.3, "precision@10": 0.2},
        }
        mlflow_metrics, artifact = calc.compute_bias_report(breakdown)
        assert "bias_disparity_ndcg_at_10" in mlflow_metrics
        assert mlflow_metrics["bias_disparity_ndcg_at_10"] == pytest.approx(0.3)
        assert "underperforming_slices" in artifact

    def test_single_category(self):
        calc = MetricsCalculator(relevance_threshold=1)
        breakdown = {"general": {"count": 5, "ndcg@10": 0.8}}
        mlflow_metrics, artifact = calc.compute_bias_report(breakdown)
        assert mlflow_metrics == {}
        assert "note" in artifact


# ── compute_selection_score ───────────────────────

class TestComputeSelectionScore:
    def test_perfect_metrics(self):
        metrics = {
            "ndcg_at_10": 1.0,
            "recall_at_10": 1.0,
            "mrr_at_5": 1.0,
            "storage_mb": 60,
        }
        score = compute_selection_score(metrics)
        assert 0.9 <= score <= 1.0

    def test_zero_metrics(self):
        metrics = {
            "ndcg_at_10": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_5": 0.0,
            "storage_mb": 1000,
        }
        score = compute_selection_score(metrics)
        assert score < 0.1

    def test_storage_penalty(self):
        base = {"ndcg_at_10": 0.8, "recall_at_10": 0.8, "mrr_at_5": 0.8}
        score_small = compute_selection_score({**base, "storage_mb": 10})
        score_large = compute_selection_score({**base, "storage_mb": 500})
        assert score_small > score_large


# ── VectorStrategy ────────────────────────────────

class TestVectorStrategy:
    def test_embedding_column_name(self):
        result = VectorStrategy.embedding_column_name("sentence-transformers/all-MiniLM-L6-v2")
        assert result == "embeddings_all_minilm_l6_v2"

    def test_embedding_column_name_no_slash(self):
        result = VectorStrategy.embedding_column_name("all-mpnet-base-v2")
        assert result == "embeddings_all_mpnet_base_v2"


# ── build_strategy ────────────────────────────────

class TestBuildStrategy:
    def test_vector_mode(self):
        cfg = _make_config(retrieval_mode=RetrievalMode.VECTOR)
        strategy = build_strategy(MagicMock(), cfg)
        assert isinstance(strategy, VectorStrategy)

    def test_bm25_mode(self):
        cfg = _make_config(retrieval_mode=RetrievalMode.BM25)
        strategy = build_strategy(MagicMock(), cfg)
        assert isinstance(strategy, BM25Strategy)

    def test_hybrid_mode(self):
        cfg = _make_config(retrieval_mode=RetrievalMode.HYBRID)
        strategy = build_strategy(MagicMock(), cfg)
        assert isinstance(strategy, HybridStrategy)

    def test_unknown_mode_raises(self):
        cfg = _make_config()
        cfg.retrieval_mode = "invalid"
        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            build_strategy(MagicMock(), cfg)


# ── HybridStrategy RRF ───────────────────────────

class TestHybridRRF:
    def test_rrf_merges_results(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        cfg = _make_config(retrieval_mode=RetrievalMode.HYBRID)
        strategy = HybridStrategy(mock_conn, cfg)

        strategy.vector.retrieve = MagicMock(return_value=(["c1", "c2"], [0.9, 0.5]))
        strategy.bm25.retrieve = MagicMock(return_value=(["c2", "c3"], [0.8, 0.4]))

        ids, scores = strategy.retrieve("q", np.array([0.1] * 384), k=3)
        assert set(ids) == {"c1", "c2", "c3"}
        assert len(scores) == 3
        assert all(s > 0 for s in scores)

    def test_rrf_ranks_shared_higher(self):
        mock_conn = MagicMock()
        cfg = _make_config(retrieval_mode=RetrievalMode.HYBRID)
        strategy = HybridStrategy(mock_conn, cfg)

        strategy.vector.retrieve = MagicMock(return_value=(["shared", "vec_only"], [0.9, 0.5]))
        strategy.bm25.retrieve = MagicMock(return_value=(["shared", "bm25_only"], [0.8, 0.4]))

        ids, scores = strategy.retrieve("q", np.array([0.1] * 384), k=10)
        assert ids[0] == "shared"


# ── Decision Gate ─────────────────────────────────

class TestDecisionGate:
    def test_open_source_within_threshold(self, capsys):
        from src.evaluation.evaluator import PipelineOrchestrator
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)

        summary = [
            {"config": "openai-text-embedding", "ndcg_at_10": 0.80, "selection_score": 0.9},
            {"config": "minilm-v2", "ndcg_at_10": 0.78, "selection_score": 0.85},
        ]
        orch._apply_decision_gate(summary)

    def test_open_source_below_threshold(self, capsys):
        from src.evaluation.evaluator import PipelineOrchestrator
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)

        summary = [
            {"config": "openai-text-embedding", "ndcg_at_10": 0.90, "selection_score": 0.95},
            {"config": "minilm-v2", "ndcg_at_10": 0.60, "selection_score": 0.70},
        ]
        orch._apply_decision_gate(summary)

    def test_zero_ndcg_exits_early(self):
        from src.evaluation.evaluator import PipelineOrchestrator
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        summary = [{"config": "test", "ndcg_at_10": 0, "selection_score": 0}]
        orch._apply_decision_gate(summary)
