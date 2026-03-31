"""
Tests for DAG task helper functions in dags/chunking_embedding_pipeline.py

Covers:
- _as_bool(): truthy/falsy string parsing
- _as_int(): string-to-int with fallback
- _safe_variable_get(): fallback on Airflow Variable error
- chunking_task(): demo mode patches FETCH_SQL, restores after
- embeddings_task(): demo mode wraps fetch function, restores after

Airflow is conditionally imported; tests skip if not installed.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers to import task functions ──

def _try_import():
    try:
        import airflow  # noqa: F401
    except ImportError:
        return None

    from dags.chunking_embedding_pipeline import (
        _as_bool,
        _as_int,
        _safe_variable_get,
        chunking_task,
        embeddings_task,
    )
    return {
        "_as_bool": _as_bool,
        "_as_int": _as_int,
        "_safe_variable_get": _safe_variable_get,
        "chunking_task": chunking_task,
        "embeddings_task": embeddings_task,
    }


_funcs = _try_import()
if _funcs is None:
    pytest.skip("Airflow not installed, skipping DAG task tests", allow_module_level=True)


_as_bool = _funcs["_as_bool"]
_as_int = _funcs["_as_int"]
_safe_variable_get = _funcs["_safe_variable_get"]
chunking_task = _funcs["chunking_task"]
embeddings_task = _funcs["embeddings_task"]


# ─────────────────────────────────────────────────
# _as_bool()
# ─────────────────────────────────────────────────

class TestAsBool:
    def test_true_values(self):
        for val in ("1", "true", "True", "TRUE", "yes", "Yes", "y", "on", "ON"):
            assert _as_bool(val) is True, f"Expected True for '{val}'"

    def test_false_values(self):
        for val in ("0", "false", "False", "no", "off", "", "random"):
            assert _as_bool(val) is False, f"Expected False for '{val}'"

    def test_whitespace_stripped(self):
        assert _as_bool("  true  ") is True

    def test_non_string_coerced(self):
        assert _as_bool(1) is True
        assert _as_bool(0) is False


# ─────────────────────────────────────────────────
# _as_int()
# ─────────────────────────────────────────────────

class TestAsInt:
    def test_valid_int_string(self):
        assert _as_int("42", 10) == 42

    def test_invalid_falls_back(self):
        assert _as_int("abc", 10) == 10

    def test_none_falls_back(self):
        assert _as_int(None, 5) == 5

    def test_int_value(self):
        assert _as_int(100, 10) == 100


# ─────────────────────────────────────────────────
# _safe_variable_get()
# ─────────────────────────────────────────────────

class TestSafeVariableGet:
    @patch("dags.chunking_embedding_pipeline.Variable")
    def test_returns_variable_value(self, mock_var):
        mock_var.get.return_value = "50"
        assert _safe_variable_get("demo_limit", "10") == "50"

    @patch("dags.chunking_embedding_pipeline.Variable")
    def test_fallback_on_exception(self, mock_var):
        mock_var.get.side_effect = Exception("DB error")
        assert _safe_variable_get("demo_limit", "10") == "10"


# ─────────────────────────────────────────────────
# chunking_task()
# ─────────────────────────────────────────────────

class TestChunkingTask:
    @patch("dags.chunking_embedding_pipeline.run_chunking_pipeline")
    @patch("dags.chunking_embedding_pipeline._safe_variable_get", return_value="false")
    def test_runs_without_demo_mode(self, mock_var, mock_run):
        dag_run = MagicMock()
        dag_run.conf = {}
        chunking_task(dag_run=dag_run)
        mock_run.assert_called_once()

    @patch("dags.chunking_embedding_pipeline.run_chunking_pipeline")
    @patch("dags.chunking_embedding_pipeline._safe_variable_get", return_value="false")
    def test_demo_mode_patches_and_restores_sql(self, mock_var, mock_run):
        import dags.chunking_embedding_pipeline as dag_mod
        from src.chunking import pipeline as chunking_mod

        original_sql = chunking_mod.FETCH_SQL
        dag_run = MagicMock()
        dag_run.conf = {"demo_mode": "true", "demo_limit_chunking": "5"}
        chunking_task(dag_run=dag_run)

        # After the task completes, SQL should be restored
        assert chunking_mod.FETCH_SQL == original_sql
        mock_run.assert_called_once()


# ─────────────────────────────────────────────────
# embeddings_task()
# ─────────────────────────────────────────────────

class TestEmbeddingsTask:
    @patch("dags.chunking_embedding_pipeline.run_embeddings_pipeline")
    @patch("dags.chunking_embedding_pipeline._safe_variable_get", return_value="false")
    def test_runs_without_demo_mode(self, mock_var, mock_run):
        dag_run = MagicMock()
        dag_run.conf = {}
        embeddings_task(dag_run=dag_run)
        mock_run.assert_called_once()

    @patch("dags.chunking_embedding_pipeline.run_embeddings_pipeline")
    @patch("dags.chunking_embedding_pipeline._safe_variable_get", return_value="false")
    def test_demo_mode_wraps_and_restores_fetch(self, mock_var, mock_run):
        from src.embeddings import pipeline as emb_mod

        original_func = emb_mod.fetch_chunks_missing_embeddings
        dag_run = MagicMock()
        dag_run.conf = {"demo_mode": "true", "demo_limit_embeddings": "3"}
        embeddings_task(dag_run=dag_run)

        # After the task completes, the function should be restored
        assert emb_mod.fetch_chunks_missing_embeddings is original_func
        mock_run.assert_called_once()
