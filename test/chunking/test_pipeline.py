"""
Tests for src/chunking/pipeline.py

Covers:
- fetch_docs(): DB cursor → list of dicts
- insert_chunks(): execute_values call with correct row tuples
- write_manifest(): GCS path and payload
- run(): happy path, no-docs exit, chunk error handling, batch flushing

All DB and GCS interactions are mocked.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from src.data_models.document_chunk import DocumentChunk


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

def _make_chunk(**overrides):
    defaults = {
        "chunk_id": "c1",
        "document_id": "d1",
        "chunk_index": 0,
        "total_chunks": 1,
        "chunk_text": "Part: 1 of 1 | content",
        "raw_text": "content",
        "word_count": 1,
        "char_start_offset": 0,
        "char_end_offset": 7,
        "strategy": "single_chunk",
        "round_label": None,
    }
    defaults.update(overrides)
    return DocumentChunk(**defaults)


def _make_row(doc_id="doc_1", platform="leetcode"):
    return {
        "document_id": doc_id,
        "source_platform": platform,
        "cleaned_content": "Short interview at Google.",
        "word_count": 5,
        "company": "Google",
        "role": "SDE",
        "experience_level": "mid",
        "interview_outcome": "offer",
        "difficulty": "medium",
        "interview_type": "onsite",
        "topics": ["dsa", "system_design"],
    }


# ─────────────────────────────────────────────────
# fetch_docs()
# ─────────────────────────────────────────────────

class TestFetchDocs:
    def test_returns_list_of_dicts(self):
        from src.chunking.pipeline import fetch_docs

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {"document_id": "d1", "source_platform": "leetcode", "cleaned_content": "text"},
        ]
        mock_conn.cursor.return_value = mock_cursor

        docs = fetch_docs(mock_conn)
        assert len(docs) == 1
        assert docs[0]["document_id"] == "d1"

    def test_empty_result(self):
        from src.chunking.pipeline import fetch_docs

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        docs = fetch_docs(mock_conn)
        assert docs == []


# ─────────────────────────────────────────────────
# insert_chunks()
# ─────────────────────────────────────────────────

class TestInsertChunks:
    @patch("src.chunking.pipeline.execute_values")
    def test_calls_execute_values(self, mock_ev):
        from src.chunking.pipeline import insert_chunks

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        chunks = [_make_chunk(chunk_id="c1"), _make_chunk(chunk_id="c2")]
        insert_chunks(mock_conn, chunks)

        mock_ev.assert_called_once()
        rows = mock_ev.call_args[0][2]
        assert len(rows) == 2
        mock_conn.commit.assert_called_once()

    @patch("src.chunking.pipeline.execute_values")
    def test_row_tuple_has_correct_fields(self, mock_ev):
        from src.chunking.pipeline import insert_chunks

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        chunk = _make_chunk(chunk_id="c1", document_id="d1", chunk_index=0)
        insert_chunks(mock_conn, [chunk])

        row = mock_ev.call_args[0][2][0]
        assert row[0] == "c1"   # chunk_id
        assert row[1] == "d1"   # document_id
        assert row[2] == 0      # chunk_index


# ─────────────────────────────────────────────────
# write_manifest()
# ─────────────────────────────────────────────────

class TestWriteManifest:
    def test_writes_to_gcs(self):
        from src.chunking.pipeline import write_manifest

        mock_gcs = MagicMock()
        stats = {
            "docs_processed": 10,
            "chunks_created": 50,
            "errors": [],
            "strategy_counts": {"structural": 3, "fixed_window": 7, "single_chunk": 0},
        }
        write_manifest(mock_gcs, stats)

        mock_gcs.write_json.assert_called_once()
        path = mock_gcs.write_json.call_args[0][0]
        assert path.startswith("chunking_manifests/manifest_")

    def test_manifest_payload_contains_stats(self):
        from src.chunking.pipeline import write_manifest

        mock_gcs = MagicMock()
        stats = {
            "docs_processed": 5,
            "chunks_created": 20,
            "errors": [{"document_id": "d1", "error": "bad"}],
            "strategy_counts": {},
        }
        write_manifest(mock_gcs, stats)

        payload = mock_gcs.write_json.call_args[0][1]
        assert payload["docs_processed"] == 5
        assert payload["chunks_created"] == 20
        assert len(payload["errors"]) == 1


# ─────────────────────────────────────────────────
# run()
# ─────────────────────────────────────────────────

class TestRun:
    @patch("src.chunking.pipeline.write_manifest")
    @patch("src.chunking.pipeline.insert_chunks")
    @patch("src.chunking.pipeline.fetch_docs")
    @patch("src.chunking.pipeline.GCSBackend")
    @patch("src.chunking.pipeline.psycopg2")
    def test_no_docs_exits_early(self, mock_pg, mock_gcs_cls, mock_fetch, mock_insert, mock_manifest):
        from src.chunking.pipeline import run

        mock_fetch.return_value = []
        run()

        mock_insert.assert_not_called()
        mock_manifest.assert_not_called()

    @patch("src.chunking.pipeline.write_manifest")
    @patch("src.chunking.pipeline.insert_chunks")
    @patch("src.chunking.pipeline.fetch_docs")
    @patch("src.chunking.pipeline.GCSBackend")
    @patch("src.chunking.pipeline.psycopg2")
    def test_happy_path_processes_docs(self, mock_pg, mock_gcs_cls, mock_fetch, mock_insert, mock_manifest):
        from src.chunking.pipeline import run

        mock_fetch.return_value = [_make_row("d1"), _make_row("d2")]
        run()

        assert mock_insert.call_count >= 1
        mock_manifest.assert_called_once()

    @patch("src.chunking.pipeline.write_manifest")
    @patch("src.chunking.pipeline.insert_chunks")
    @patch("src.chunking.pipeline.fetch_docs")
    @patch("src.chunking.pipeline.chunk_document")
    @patch("src.chunking.pipeline.GCSBackend")
    @patch("src.chunking.pipeline.psycopg2")
    def test_chunk_error_logged_in_stats(self, mock_pg, mock_gcs_cls, mock_chunk_doc, mock_fetch, mock_insert, mock_manifest):
        from src.chunking.pipeline import run

        mock_fetch.return_value = [_make_row("d1")]
        mock_chunk_doc.side_effect = RuntimeError("chunking failed")
        run()

        manifest_stats = mock_manifest.call_args[0][1]
        assert len(manifest_stats["errors"]) == 1
        assert manifest_stats["errors"][0]["document_id"] == "d1"
