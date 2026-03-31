"""
Tests for src/eval_dataset_labelling/labeled_data_to_db.py

Covers:
- read_csv_from_gcs(): blob exists returns DataFrame, blob missing raises
- run_pipeline(): happy path, missing columns, empty after filter

All DB and GCS interactions are mocked.

Skipped if pandas is not installed.
"""
import io
import pytest

pd = pytest.importorskip("pandas", reason="pandas not installed")

from unittest.mock import MagicMock, patch

from src.eval_dataset_labelling.labeled_data_to_db import (
    read_csv_from_gcs,
    INSERT_SQL,
    TABLE_NAME,
    BATCH_SIZE,
)


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

def _mock_gcs():
    gcs = MagicMock()
    gcs.bucket_name = "test-bucket"
    return gcs


def _make_csv_text(rows):
    if not rows:
        columns = [
            "query_id", "query_text", "chunk_id", "document_id",
            "query_category", "relevance", "relevance_reason"
        ]
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────
# read_csv_from_gcs()
# ─────────────────────────────────────────────────

class TestReadCsvFromGcs:
    def test_blob_exists_returns_dataframe(self):
        gcs = _mock_gcs()
        blob = MagicMock()
        blob.exists.return_value = True
        blob.download_as_text.return_value = "col_a,col_b\n1,2\n3,4\n"
        gcs.bucket.blob.return_value = blob
        logger = MagicMock()

        df = read_csv_from_gcs(gcs, "path/to/file.csv", logger)
        assert len(df) == 2
        assert list(df.columns) == ["col_a", "col_b"]

    def test_blob_missing_raises(self):
        gcs = _mock_gcs()
        blob = MagicMock()
        blob.exists.return_value = False
        gcs.bucket.blob.return_value = blob
        logger = MagicMock()

        with pytest.raises(FileNotFoundError):
            read_csv_from_gcs(gcs, "missing.csv", logger)

    def test_strips_column_whitespace(self):
        gcs = _mock_gcs()
        blob = MagicMock()
        blob.exists.return_value = True
        blob.download_as_text.return_value = " col_a , col_b \n1,2\n"
        gcs.bucket.blob.return_value = blob
        logger = MagicMock()

        df = read_csv_from_gcs(gcs, "file.csv", logger)
        assert list(df.columns) == ["col_a", "col_b"]


# ─────────────────────────────────────────────────
# run_pipeline()
# ─────────────────────────────────────────────────

class TestRunPipeline:
    def _build_csv_text(self, relevance_values):
        rows = []
        for i, rel in enumerate(relevance_values):
            rows.append({
                "query_id": f"q{i}",
                "query_text": f"query {i}",
                "chunk_id": f"c{i}",
                "document_id": f"d{i}",
                "query_category": "general",
                "relevance": rel,
                "relevance_reason": "test",
            })
        return _make_csv_text(rows)

    @patch("src.eval_dataset_labelling.labeled_data_to_db.psycopg2")
    @patch("src.eval_dataset_labelling.labeled_data_to_db.GCSBackend")
    def test_missing_columns_raises(self, mock_gcs_cls, mock_pg):
        from src.eval_dataset_labelling.labeled_data_to_db import run_pipeline

        mock_gcs = _mock_gcs()
        mock_gcs_cls.return_value = mock_gcs
        blob = MagicMock()
        blob.exists.return_value = True
        blob.download_as_text.return_value = "wrong_col\n1\n"
        mock_gcs.bucket.blob.return_value = blob

        with pytest.raises(ValueError, match="Missing columns"):
            run_pipeline()

    @patch("src.eval_dataset_labelling.labeled_data_to_db.psycopg2")
    @patch("src.eval_dataset_labelling.labeled_data_to_db.GCSBackend")
    def test_empty_after_filter_exits(self, mock_gcs_cls, mock_pg):
        from src.eval_dataset_labelling.labeled_data_to_db import run_pipeline

        mock_gcs = _mock_gcs()
        mock_gcs_cls.return_value = mock_gcs
        csv_text = self._build_csv_text([])
        blob = MagicMock()
        blob.exists.return_value = True
        blob.download_as_text.return_value = csv_text
        mock_gcs.bucket.blob.return_value = blob

        # Should return without connecting to DB
        run_pipeline()
        mock_pg.connect.assert_not_called()

    @patch("src.eval_dataset_labelling.labeled_data_to_db.psycopg2")
    @patch("src.eval_dataset_labelling.labeled_data_to_db.GCSBackend")
    def test_happy_path_inserts_records(self, mock_gcs_cls, mock_pg):
        from src.eval_dataset_labelling.labeled_data_to_db import run_pipeline

        mock_gcs = _mock_gcs()
        mock_gcs_cls.return_value = mock_gcs
        csv_text = self._build_csv_text([1, 2, 0])
        blob = MagicMock()
        blob.exists.return_value = True
        blob.download_as_text.return_value = csv_text
        mock_gcs.bucket.blob.return_value = blob

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (3,)
        mock_cursor.fetchall.return_value = [(0, 1), (1, 1), (2, 1)]
        mock_conn.cursor.return_value = mock_cursor
        mock_pg.connect.return_value = mock_conn

        run_pipeline()

        mock_pg.extras.execute_batch.assert_called()
        mock_conn.commit.assert_called()
