"""
Tests for src/embeddings/pipeline.py

Covers:
- embedding_column_name(): model name → column name conversion
- fetch_chunks_missing_embeddings(): SQL generation and result parsing
- update_embeddings_batch(): executemany + commit
- write_manifest(): GCS path and payload
- run(): happy path, no-chunks exit, encode error handling

All DB, SentenceTransformer, and GCS interactions are mocked.

Skipped if sentence_transformers is not installed.
"""
import pytest

pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed")

from unittest.mock import MagicMock, patch, call

from src.embeddings.pipeline import (
    embedding_column_name,
    fetch_chunks_missing_embeddings,
    update_embeddings_batch,
    write_manifest,
)


# ─────────────────────────────────────────────────
# embedding_column_name()
# ─────────────────────────────────────────────────

class TestEmbeddingColumnName:
    def test_dashes_to_underscores(self):
        result = embedding_column_name("all-MiniLM-L6-v2")
        assert result == "embeddings_all_minilm_l6_v2"

    def test_lowercase(self):
        result = embedding_column_name("ALL-MPNET-BASE-V2")
        assert result == "embeddings_all_mpnet_base_v2"

    def test_no_dashes(self):
        result = embedding_column_name("simplemodel")
        assert result == "embeddings_simplemodel"


# ─────────────────────────────────────────────────
# fetch_chunks_missing_embeddings()
# ─────────────────────────────────────────────────

class TestFetchChunksMissingEmbeddings:
    def test_returns_list_of_dicts(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {"chunk_id": "c1", "chunk_text": "text 1"},
            {"chunk_id": "c2", "chunk_text": "text 2"},
        ]
        mock_conn.cursor.return_value = mock_cursor

        result = fetch_chunks_missing_embeddings(mock_conn, ["all-MiniLM-L6-v2"])
        assert len(result) == 2
        assert result[0]["chunk_id"] == "c1"

    def test_empty_result(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        result = fetch_chunks_missing_embeddings(mock_conn, ["all-MiniLM-L6-v2"])
        assert result == []

    def test_sql_uses_or_for_multiple_models(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        fetch_chunks_missing_embeddings(
            mock_conn, ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
        )

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "OR" in executed_sql
        assert "embeddings_all_minilm_l6_v2" in executed_sql
        assert "embeddings_all_mpnet_base_v2" in executed_sql


# ─────────────────────────────────────────────────
# update_embeddings_batch()
# ─────────────────────────────────────────────────

class TestUpdateEmbeddingsBatch:
    def test_calls_executemany_and_commit(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        columns = ["embeddings_all_minilm_l6_v2"]
        rows = [([0.1, 0.2, 0.3], "c1"), ([0.4, 0.5, 0.6], "c2")]

        update_embeddings_batch(mock_conn, columns, rows)

        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()

    def test_sql_set_clause_matches_columns(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        columns = ["embeddings_all_minilm_l6_v2", "embeddings_all_mpnet_base_v2"]
        update_embeddings_batch(mock_conn, columns, [])

        sql = mock_cursor.executemany.call_args[0][0]
        assert "embeddings_all_minilm_l6_v2 = %s" in sql
        assert "embeddings_all_mpnet_base_v2 = %s" in sql


# ─────────────────────────────────────────────────
# write_manifest()
# ─────────────────────────────────────────────────

class TestWriteManifest:
    def test_writes_to_gcs(self):
        mock_gcs = MagicMock()
        stats = {"models": ["m1"], "chunks_embedded": 100, "errors": []}
        write_manifest(mock_gcs, stats)

        mock_gcs.write_json.assert_called_once()
        path = mock_gcs.write_json.call_args[0][0]
        assert path.startswith("embedding_manifests/manifest_")

    def test_payload_matches_stats(self):
        mock_gcs = MagicMock()
        stats = {"models": ["m1", "m2"], "chunks_embedded": 50, "errors": []}
        write_manifest(mock_gcs, stats)

        payload = mock_gcs.write_json.call_args[0][1]
        assert payload == stats


# ─────────────────────────────────────────────────
# run()
# ─────────────────────────────────────────────────

class TestRun:
    @patch("src.embeddings.pipeline.write_manifest")
    @patch("src.embeddings.pipeline.update_embeddings_batch")
    @patch("src.embeddings.pipeline.fetch_chunks_missing_embeddings")
    @patch("src.embeddings.pipeline.SentenceTransformer")
    @patch("src.embeddings.pipeline.GCSBackend")
    @patch("src.embeddings.pipeline.psycopg2")
    def test_no_chunks_exits_early(
        self, mock_pg, mock_gcs_cls, mock_st, mock_fetch, mock_update, mock_manifest
    ):
        from src.embeddings.pipeline import run

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        mock_fetch.return_value = []

        run(models=["test-model"])

        mock_update.assert_not_called()
        mock_manifest.assert_not_called()

    @patch("src.embeddings.pipeline.write_manifest")
    @patch("src.embeddings.pipeline.update_embeddings_batch")
    @patch("src.embeddings.pipeline.fetch_chunks_missing_embeddings")
    @patch("src.embeddings.pipeline.SentenceTransformer")
    @patch("src.embeddings.pipeline.GCSBackend")
    @patch("src.embeddings.pipeline.psycopg2")
    def test_happy_path_embeds_chunks(
        self, mock_pg, mock_gcs_cls, mock_st, mock_fetch, mock_update, mock_manifest
    ):
        from src.embeddings.pipeline import run
        import numpy as np

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
        mock_st.return_value = mock_model

        mock_fetch.return_value = [
            {"chunk_id": "c1", "chunk_text": "text 1"},
            {"chunk_id": "c2", "chunk_text": "text 2"},
        ]

        run(models=["test-model"])

        assert mock_update.call_count >= 1
        mock_manifest.assert_called_once()

    @patch("src.embeddings.pipeline.write_manifest")
    @patch("src.embeddings.pipeline.update_embeddings_batch")
    @patch("src.embeddings.pipeline.fetch_chunks_missing_embeddings")
    @patch("src.embeddings.pipeline.SentenceTransformer")
    @patch("src.embeddings.pipeline.GCSBackend")
    @patch("src.embeddings.pipeline.psycopg2")
    def test_encode_error_logged(
        self, mock_pg, mock_gcs_cls, mock_st, mock_fetch, mock_update, mock_manifest
    ):
        from src.embeddings.pipeline import run

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = RuntimeError("encode failed")
        mock_st.return_value = mock_model

        mock_fetch.return_value = [{"chunk_id": "c1", "chunk_text": "text"}]

        run(models=["test-model"])

        manifest_stats = mock_manifest.call_args[0][1]
        assert len(manifest_stats["errors"]) == 1
