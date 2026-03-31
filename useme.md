# How to Run InterviewPrep-AI

## 1. Prerequisites

- Python 3.10+
- Google Cloud SDK (for GCS authentication)
- PostgreSQL client libraries (for `psycopg2`)

---

## 2. Python Dependencies

| Package | Purpose |
|---|---|
| `requests` | HTTP client for GFG and LeetCode scrapers |
| `beautifulsoup4` | HTML parsing for GFG scraper and Medium content extraction |
| `google-cloud-storage` | GCS backend for all storage operations |
| `playwright` | Headless browser automation for Medium scraper |
| `markdownify` | HTML-to-Markdown conversion for Medium content |
| `pyyaml` | YAML config loading for preprocessing steps and pipeline |
| `langdetect` | Language detection in QualityFilter step |
| `datasketch` | MinHash + LSH for near-duplicate detection |
| `spacy` | NER-based company extraction in EntityExtractor |
| `psycopg2-binary` | PostgreSQL driver for the database loader |
| `sentence-transformers` | Bi-encoder embedding models for chunking and retrieval |
| `numpy` | Array operations for embeddings and metric computation |
| `pandas` | Tabular data handling for eval dataset labelling |
| `pgvector` | PostgreSQL vector similarity search extension |
| `rank-bm25` | BM25 scoring for lexical retrieval in eval dataset generation |
| `openai` | LLM-as-a-judge relevance grading for eval dataset |
| `mlflow` | Experiment tracking for retrieval model evaluation |
| `pytest` | Test framework |

---

## 3. Environment Setup

```bash
# 1. Clone the repository
git clone <repo-url> && cd interviewprep-ai

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download spaCy English model (used by EntityExtractor)
python -m spacy download en_core_web_sm

# 5. Install Playwright browsers (used by MediumScraper)
playwright install chromium

# 6. Set environment variables
export GCS_BUCKET_NAME="interviewprep-ai-data"
export GCP_PROJECT_ID="professorbot-dovbsg"
export DB_HOST="34.148.0.165"
export DB_NAME="interviewprep-ai-database"
export DB_USER="postgres"
export DB_PASSWORD="<your-password>"
export DB_PORT="5432"

# 7. Authenticate with GCP (choose one)
# Option A: Application Default Credentials (local dev)
gcloud auth application-default login
# Option B: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# 8. Initialize Airflow (if running locally)
export AIRFLOW_HOME=~/airflow
airflow db init
cp dags/scraping_pipeline.py $AIRFLOW_HOME/dags/
cp dags/chunking_embedding_pipeline.py $AIRFLOW_HOME/dags/

# 9. Set MLflow tracking URI (for evaluation pipeline)
export MLFLOW_TRACKING_URI="http://<mlflow-host>:5000"

# 10. Run tests to verify setup
pytest test/ -v
```

---

## 4. Reproducibility

### Reproducing the Pipeline from Scratch

```bash
# 1. Clone and set up the environment (see above)
git clone <repo-url> && cd interviewprep-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
playwright install chromium

# 2. Configure GCP credentials and environment variables
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
export DB_HOST="..." DB_NAME="..." DB_USER="..." DB_PASSWORD="..." DB_PORT="5432"

# 3. Verify tests pass
pytest test/ -v

# 4. Initialize Airflow
export AIRFLOW_HOME=~/airflow
airflow db init
cp dags/scraping_pipeline.py $AIRFLOW_HOME/dags/
cp dags/chunking_embedding_pipeline.py $AIRFLOW_HOME/dags/

# 5. Trigger the data pipeline
airflow dags trigger interview_scraping_pipeline

# 6. Trigger the chunking + embedding pipeline
airflow dags trigger chunking_embedding_pipeline

# 7. Run retrieval evaluation (requires MLflow + DB access)
export MLFLOW_TRACKING_URI="http://<mlflow-host>:5000"
python src/evaluation/evaluator.py
```

### What Makes It Reproducible

- **Idempotent DB writes** — `ON CONFLICT DO UPDATE` means you can re-run the pipeline on the same batch without duplicating data.
- **Batch ID determinism** — The batch ID is derived from Airflow's `logical_date`, so the same trigger date always produces the same batch ID and GCS paths.
- **Checkpoint recovery** — Preprocessing saves intermediate state to GCS. On retry, it resumes from the last checkpoint rather than re-processing from scratch.
- **Dedup before fetch** — Scrapers check GCS before downloading, so re-running a scrape on the same batch skips already-collected documents.
- **Config-driven pipeline** — Step ordering, thresholds, and toggle switches live in YAML configs, not code. Changing the pipeline behavior doesn't require code changes.
- **Pinned dependencies** — `requirements.txt` locks the dependency set. For full reproducibility, consider generating a `pip freeze` snapshot.
- **Manifest watermarks** — Each scraping run writes a manifest with `last_sitemap_lastmod`, enabling incremental runs that pick up exactly where the last run left off.

---

## 5. Test Suite

### Overview

The project has a comprehensive test suite with **29 test files** across 9 modules, plus 2 shared `conftest.py` fixture files. Everything uses `pytest` with `unittest.mock` — no real GCS, database, or network calls needed to run the tests. Tests with missing optional dependencies (e.g., `sentence_transformers`, `mlflow`, `airflow`) skip gracefully via `pytest.importorskip`.

### What's Covered

- **Preprocessing steps** — Each of the 6 steps has its own test file covering happy paths, filtering behavior, edge cases, and `run_batch()` integration. The deduplicator tests include state persistence and MinHash accuracy. The schema validator tests use parametrized fixtures for required field validation.
- **Pipeline orchestrator** — Tests for `CheckpointManager` (save/load/cleanup), `_build_steps()` (enabled/disabled/unknown), `_resolve_start()` (fresh vs resume), quarantine splitting, and full `run()` flows (success, resume, fail_fast, exception handling).
- **Database** — Loader tests mock both GCS and psycopg2, verifying correct SQL execution, error isolation, and summary generation. Sanitizer tests cover all enum mappings.
- **Scrapers** — Config tests validate static attributes, URL patterns, and path generation. Implementation tests mock HTTP/GraphQL/Playwright calls and verify parsing, dedup, error handling, and manifest creation.
- **Storage** — GCS backend tests cover all auth paths, CRUD operations, and temp key cleanup. The ABC test verifies that partial implementations are rejected.
- **Chunking** — Tests for word counting, sentence splitting, header building, strategy detection, chunk_document, validate_chunks, DB fetch/insert, GCS manifest writing, and pipeline run flow.
- **Embeddings** — Tests for column naming, missing embeddings fetch, batch update, manifest writing, and pipeline run.
- **Eval dataset labelling** — Tests for score normalization, hybrid fusion, BM25 index, result pooling, chunk truncation, LLM response parsing, retry logic, CSV reading, and DB insertion.
- **Evaluation** — Tests for retrieval metrics (MRR, Recall, Precision, NDCG), retrieval strategies, RRF fusion, bias report, config loading, selection score, and decision gate.
- **DAGs** — Tests for task helper functions (`_as_bool`, `_as_int`, `_safe_variable_get`) and task callables with demo mode patching.

### Running Tests

```bash
# Run everything
pytest test/ -v

# Run a specific module
pytest test/preprocessing/ -v
pytest test/database/ -v
pytest test/scrapers/ -v
pytest test/storage/ -v
pytest test/chunking/ -v
pytest test/embeddings/ -v
pytest test/eval_dataset_labelling/ -v
pytest test/evaluation/ -v

# Run with coverage
pytest test/ --cov=src --cov-report=term-missing

# Run tests matching a pattern
pytest test/ -k "test_exact_duplicate" -v
```