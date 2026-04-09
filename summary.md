# InterviewPrep-AI: Architecture Summary

## Overview

InterviewPrep-AI is an end-to-end data engineering pipeline that scrapes interview experiences from GeeksforGeeks, LeetCode, and Medium, processes raw data through a configurable multi-step transformation pipeline, validates output integrity, and loads cleaned documents into PostgreSQL. A chat-based web application (FastAPI backend + Next.js frontend) provides an AI-powered interview preparation assistant using a RAG (Retrieval-Augmented Generation) pipeline with hybrid retrieval (pgvector + BM25 + RRF fusion) and OpenAI generation. An automated evaluation framework benchmarks multiple retrieval strategies, selects the best model, and deploys it via Vertex AI Model Registry. Continuous monitoring detects embedding drift and performance degradation, automatically triggering corpus refresh when thresholds are breached. The pipeline is orchestrated by Apache Airflow with artifacts stored in Google Cloud Storage (GCS), experiments tracked with MLflow, and CI/CD managed by GitHub Actions.

---

## Architecture Layers

```
Scrapers (3 parallel)
  |
Raw documents --> GCS raw/{batch_id}/
  |
Preprocessing Pipeline (6 sequential steps)
  |
Processed documents --> GCS processed/{batch_id}/
  |
Validation (count matching, report existence)
  |
Database Loader --> PostgreSQL
  |
Chunking & Embeddings --> pgvector
  |
Email Notification

        --- ML/Evaluation Layer ---

Eval Dataset Labelling
  |  - Multi-strategy retrieval pooling (Vector, BM25, Hybrid)
  |  - LLM-as-a-judge relevance scoring (GPT-4o-mini)
  |  - Gold standard dataset --> eval_dataset table
  |
Retrieval Model Evaluation (PipelineOrchestrator)
  |  - 5 configs: 2 pure vector, 1 pure BM25, 2 hybrid
  |  - Metrics: MRR, Recall, Precision, NDCG @ k
  |  - Bias analysis (fairness gaps, representation)
  |  - Best model selection --> Vertex AI Model Registry
  |  - Experiment tracking --> MLflow
  |
CI/CD: eval_pipeline.yml (auto-triggered on config changes)

        --- RAG Pipeline Layer ---

Vertex AI Model Registry --> best embedding model
  |
HybridRetriever (vector + BM25 via RRF)
  |
RAGGenerator (OpenAI gpt-4.1-mini)
  |  - Context-grounded answers with source citations
  |  - Token usage tracking

        --- Monitoring Layer ---

Production Query Logs (query_logs table)
  |
Drift Detection (weekly, composite 2-of-3)
  |  - Centroid cosine distance
  |  - Per-dimension P95 shift
  |  - Evidently DataDriftReport (KS test)
  |
Performance Monitor (weekly)
  |  - Golden dataset eval score vs baseline
  |  - Standalone Evidently drift check
  |
Corpus Refresh (triggered on drift/degradation)
  |  - Re-scrape → re-chunk → re-embed → regression gate
  |  - Slack + email notifications

        --- Web Application Layer ---

PostgreSQL (pgvector)
  |
FastAPI Backend (RAG Chat API)
  |  - POST /api/chat: hybrid retrieval + OpenAI generation
  |  - GET /api/health: readiness check
  |  - Query logging to query_logs table (drift monitoring)
  |
Next.js Frontend (Chat Interface)
     - Single-page conversational UI
     - Message bubbles with markdown rendering
     - Source citations with clickable links
     - Suggestion prompts for first-time users
```

---

## File-by-File Breakdown

### 1. Data Models (`src/data_models/`)

| File | Purpose |
|------|---------|
| `scraped_document.py` | Frozen dataclass for raw, unprocessed documents emitted by scrapers. Fields include `document_id`, `source_platform`, `source_url`, `title`, `raw_content`, `published_at`, `scraped_at`, etc. Provides JSON serialization and stable document ID generation via SHA256 hashing of URLs. |
| `preprocessed_document.py` | Frozen dataclass for fully validated documents ready for DB insertion. Implements strict validation including required field checks, platform alias normalization (`geeksforgeeks` -> `gfg`), safe enum coercion, and interview type filtering. |
| `scraping_manifest.py` | Dataclass tracking scraper execution metadata: `scrape_date`, `scrape_type`, `started_at`, `completed_at`, `sources`, `total_files`, `last_sitemap_lastmod`. Uses `StorageBackend` abstraction for persistence. |
| `preprocessing_report.py` | Dataclass summarizing a full pipeline run. Tracks `batch_id`, timing, counts, per-step results, quarantined count, and status. Written to `processed/{batch_id}_report.json`. |
| `db_load_report.py` | Dataclass generated after DB insertion. Tracks total/inserted/skipped counts, lists of successful inserts, read errors, and insert errors. Stored at `gs://{bucket}/db-report/`. |
| `document_chunk.py` | Dataclass for document chunks used by the embedding pipeline. Stores chunk content, metadata (company, role, round, part info), and vector embeddings. |

---

### 2. Web Scrapers (`src/scrapers/`)

All scrapers follow a unified interface: accept a `StorageBackend`, produce `ScrapedInterviewDocument` objects, write manifests for state tracking. Batch IDs follow `{YYYY-MM-DD}_{bulk|weekly}` convention.

| File | Purpose |
|------|---------|
| `gfg.py` | **GeeksforGeeks scraper.** Sitemap-based strategy. Fetches sitemap index, filters to `post/` sitemaps, date-filters (bulk: `>= 2025-01-01`, incremental: last manifest watermark), extracts interview-experience URLs via regex, deduplicates against GCS before fetching. Parses pages with BeautifulSoup. Output: `raw/{batch_id}/gfg/`. |
| `leetcode.py` | **LeetCode scraper.** GraphQL API strategy. Queries `discussPostItems` with `tagSlugs: ["interview"]`, paginating in batches of 50. Fetches full content via `discussPostDetail`. Company extraction: post tags + title matching against 60+ companies. Handles rate limits (60s retry on 429). Output: `raw/{batch_id}/leetcode/`. |
| `medium.py` | **Medium scraper.** Playwright-based (client-side rendering). Launches headless Chromium with realistic fingerprint. Content extraction: primary via `__APOLLO_STATE__` JSON, fallback chain through `<article>` / `postArticle` / `<main>` tags. Paywall detection via multiple signals. Output: `raw/{batch_id}/medium/`. |
| `configs/gfg.py` | Static configuration for GFG scraper: URLs, regex patterns, batch ID generation, path construction. |
| `configs/leetcode.py` | Static configuration for LeetCode scraper: GraphQL queries (captured from browser DevTools), URLs, paths. |
| `configs/medium.py` | Static configuration for Medium scraper: sitemap URLs, browser settings, paths. |

---

### 3. Preprocessing Pipeline (`src/preprocessing/`)

Config-driven, single-process, checkpoint-based resilience. Raw docs flow through 6 sequential steps.

| File | Purpose |
|------|---------|
| `pipeline.py` | **Pipeline orchestrator.** Loads raw docs from GCS, builds step chain from `pipeline_config.yaml`, executes steps iteratively with checkpointing, handles resume-on-retry logic (checks Airflow `try_number`), splits valid docs from quarantined ones, writes final output to `processed/{batch_id}/`. |
| `steps/base.py` | Abstract `PreprocessingStep` class. Defines `name`, `process(doc)`, and `run_batch(docs)` with per-document error isolation, timing, and metrics aggregation. Returns `StepResult`. |
| `steps/content_normalizer.py` | **Step 1.** Normalizes HTML/Markdown to clean plain text. Pipeline: encoding fixes -> HTML stripping (BeautifulSoup) -> Markdown removal -> boilerplate removal -> emoji removal -> Unicode NFC normalization -> whitespace standardization. Filters: `empty_raw_content`, `empty_after_cleaning`. |
| `steps/pii_remover.py` | **Step 2.** Scrubs PII using regex patterns from `pii_patterns.yaml`. Replaces emails with `[EMAIL]`, phones with `[PHONE]`, URLs with `[PERSONAL_URL]`, etc. Never filters documents. |
| `steps/quality_filter.py` | **Step 3.** Drops low-quality docs. Checks (cheapest-first): word count bounds, error page detection, language detection (langdetect), interview signal ratio (2% threshold). Adds `quality_score` (0.0-1.0). |
| `steps/deduplicator.py` | **Step 4.** Two-tier dedup: (1) Exact via SHA256 hash (O(1) lookup), (2) Near via MinHash + LSH (0.80 similarity threshold, 128 permutations, 3-word shingles). State persists across batches. |
| `steps/entity_extractor.py` | **Step 5.** Extracts 8 structured fields via regex + curated dicts + spaCy NER (lazy-loaded). Fields: company, role, experience_level, interview_types, num_rounds, topics, outcome, difficulty. |
| `steps/schema_validator.py` | **Step 6.** Final gate. Maps enriched dict to `ProcessedInterviewDocument` constructor and validates. Filters: `schema_validation_failed`. |
| `registry.py` | Simple `Dict[str, type]` mapping step names to classes. Steps register via `steps/__init__.py`. |

#### Configuration Resources (`src/preprocessing/resources/`)

| File | Purpose |
|------|---------|
| `pipeline_config.yaml` | Master pipeline config: step ordering, checkpointing flags, batch sizes, GCS paths, fail_fast mode. |
| `normalization_patterns.yaml` | Regex patterns for text normalization. |
| `pii_patterns.yaml` | Regex patterns for PII detection (emails, phones, URLs, SSNs, etc.). |
| `quality_filters.yaml` | Thresholds for word count, language, signal ratio; error page patterns; interview keywords. |
| `dedup_configs.yaml` | Exact hash config, LSH similarity threshold, num_perm, shingle_size, state_dir. |
| `entity_extraction.yaml` | Regex patterns, company aliases, role aliases, keyword lists per topic category. |

---

### 4. Storage Layer (`src/storage/`)

| File | Purpose |
|------|---------|
| `storage_backend.py` | Abstract base class defining 4 methods: `write_json`, `read_json`, `file_exists`, `list_files`. Enables test/production swapping (in-memory dict for tests, GCS for production). |
| `gcs_backend.py` | GCS implementation. Auth priority: (1) Secret Manager, (2) Default credentials. Handles JSON read/write, blob listing (sorted newest-first), bucket existence verification. |

---

### 5. Database Layer (`src/database/`)

| File | Purpose |
|------|---------|
| `loader.py` | **Batch DB loader.** Reads processed JSON from GCS in configurable chunks (default 50), deserializes into `ProcessedInterviewDocument`, performs idempotent upserts via `ON CONFLICT DO UPDATE` into `processed_documents` and `interview_metadata` tables. Get-or-create for company/role lookups with in-memory caching. Per-row error isolation. |
| `queries.py` | SQL query constants: `UPSERT_PROCESSED_DOC` and `UPSERT_INTERVIEW_META` with enum casts, JSONB, and `text[]` array handling. |
| `sanitizers.py` | Maps Python model values to valid PostgreSQL enum values. Converts `"unknown"` to `NULL`, handles platform alias mapping. |

---

### 6. Chunking & Embeddings (`src/chunking/` & `src/embeddings/`)

| File | Purpose |
|------|---------|
| `chunking/chunker.py` | Splits documents into semantic chunks for embedding/retrieval. Detects interview round boundaries via regex. Generates `DocumentChunk` objects with context headers (company, role, round, part). |
| `chunking/chunking_config.yaml` | Chunk size in words, overlap, min doc words, round boundary patterns. |
| `embeddings/pipeline.py` | Generates vector embeddings for document chunks. Configurable biencoder models (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`). Updates embedding columns in `document_chunks` table via `pgvector`. Writes manifest to GCS. |
| `embeddings/embeddings_configs.yaml` | Model list, batch sizes for embedding generation. |

---

### 7. RAG Pipeline (`src/rag_pipeline/`)

Retrieval-Augmented Generation system that answers interview prep questions using hybrid retrieval and OpenAI generation, with model selection driven by Vertex AI Model Registry.

| File | Purpose |
|------|---------|
| `config.yaml` | Master RAG config: GCP project settings (project ID, region, Vertex AI model registry name), database connection parameters, MLflow tracking URI, retrieval config (`top_k=8`, RRF parameters, BM25/vector weights), embedding models mapping (`all-MiniLM-L6-v2` dim:384, `all-mpnet-base-v2` dim:768) with DB column names, generation config (OpenAI `gpt-4.1-mini`, temperature 0.3, max_tokens 1024). |
| `retriever.py` | **`HybridRetriever` class.** Loads `SentenceTransformer` model for query encoding. `_vector_search()`: pgvector cosine similarity via dynamic SQL (joins document_chunks, processed_documents, interview_metadata, companies, roles). `_bm25_search()`: full-text search using `to_tsvector`/`plainto_tsquery` with `ts_rank_cd`. `_reciprocal_rank_fusion()`: merges vector and BM25 results using weighted RRF. `retrieve()` returns list of `{id, text, source_url, company, role}`. |
| `generator.py` | **`RAGGenerator` class.** Takes `HybridRetriever` + generation config. Uses OpenAI client (`OPENAI_API_KEY` env var). `generate()`: retrieves chunks, builds prompt, calls OpenAI, returns response with answer, chunks, model, and token usage metrics. |
| `prompt.py` | Prompt construction utilities. `SYSTEM_PROMPT`: instructs GPT to be a technical interview prep assistant — ground claims in context, cite sources inline, handle topic mismatches gracefully, reject non-interview questions. `build_context()`: formats chunks as numbered sections. `build_messages()`: constructs system + user message pair. |
| `model_registry.py` | **Model registry orchestration.** `_get_config_from_registry()`: queries Vertex AI Model Registry for latest model, extracts `best-config` label. `_get_model_name_from_mlflow()`: finds MLflow pipeline parent run, reads `best_config` tag, matches child run, extracts `model_name`. `get_deployed_embedding_model()`: full flow combining both, validates against config, returns `{model_name, embedding_dim, embedding_column}`. |
| `pipeline.py` | **`build_generator()` factory function.** Loads config, resolves deployed embedding model via registry, initializes `HybridRetriever` and `RAGGenerator`. Entry point for the RAG system. |

---

### 8. Evaluation Module (`src/evaluation/`)

Automated retrieval model evaluation framework that benchmarks multiple embedding/retrieval configurations against a gold standard dataset and selects the best model.

| File | Purpose |
|------|---------|
| `retrieval_model_configs.yaml` | Evaluation config: MLflow tracking URI, `relevance_threshold=1`, `max_k=15`, BM25 defaults. Defines 5 retrieval configs: (1) pure vector MiniLM, (2) pure vector mpnet, (3) pure BM25, (4) hybrid MiniLM+BM25 (RRF 0.5/0.5), (5) hybrid mpnet+BM25 (RRF 0.5/0.5). |
| `evaluator.py` | **Core evaluation engine.** `Evaluator` class computes ranking metrics: `metric_mrr_at_k()`, `metric_recall_at_k()`, `metric_precision_at_k()`, `metric_ndcg_at_k()`. `RetrieverStrategy` abstract class with implementations: `VectorRetriever` (pgvector cosine), `BM25Retriever` (full-text lexical), `HybridRetriever` (RRF fusion). `BiasReport` dataclass for fairness metrics (representation, fairness gaps, disparity). `PipelineOrchestrator`: loads gold standard from `eval_dataset` table, runs each config, computes metrics + bias analysis, logs child runs to MLflow per config + parent run with best config determination. |

---

### 9. Eval Dataset Labelling (`src/eval_dataset_labelling/`)

Pipeline for generating gold standard retrieval evaluation datasets using multi-strategy retrieval pooling and LLM-as-a-judge relevance scoring.

| File | Purpose |
|------|---------|
| `dataset_generator.py` | **Retrieval pooling pipeline.** Runs 3 retrieval strategies (Vector, BM25, Hybrid) against PostgreSQL+pgvector. `ChunkStore`: DB layer for pgvector queries. `BM25Index`: in-memory BM25 indexing. Pools results per query with `pool_results_for_query()` to deduplicate chunks while tracking source rankings. Loads queries from GCS CSV. Outputs CSV with columns: query_id, query_text, query_category, chunk_id, document_id, chunk_text, sources, vector/bm25/hybrid ranks and scores. |
| `llm_relevance_judge.py` | **LLM-as-a-judge scorer.** Reads pooled dataset from GCS. Sends each (query, chunk) pair to GPT-4o-mini for relevance scoring (0=not relevant, 1=partially, 2=highly relevant). Features: detailed grading rubric, chunk truncation (1500 chars), 5 concurrent workers, retry with exponential backoff (max 3 retries), resume support (skips scored rows), batch checkpointing (every 200 rows to GCS). |
| `labeled_data_to_db.py` | **Golden dataset loader.** Reads LLM-scored CSV from GCS. Inserts relevant pairs (relevance >= 1) into `eval_dataset` table via batch upserts (500 rows/batch). Per-row error isolation on batch failure. Logs relevance distribution and unique query coverage by category. |

---

### 10. Orchestration (`dags/`)

| File | Purpose |
|------|---------|
| `scraping_pipeline.py` | **Airflow DAG** (`interview_scraping_pipeline`). Manual trigger, no schedule. Task flow: `start` -> 3 parallel scrapers -> `print_summary` -> `run_preprocessing` -> `validate_processed_data` -> `load_to_database` -> `complete` -> `build_email` -> `send_notification_email`. Supports demo mode via `dag_run.conf`. Email sends on all outcomes (`trigger_rule='all_done'`). |

---

### 11. Tests (`test/`)

Framework: pytest with unittest.mock (no real GCS, DB, or network calls).

| Directory/File | Purpose |
|----------------|---------|
| `test/database/conftest.py` | Shared DB test fixtures. |
| `test/database/test_loader.py` | BatchDBLoader tests: mocks GCS + psycopg2, verifies SQL execution, error isolation, summary generation. |
| `test/database/test_report.py` | Report schema validation tests. |
| `test/database/test_sanitizers.py` | Sanitizer mapping tests. |
| `test/preprocessing/conftest.py` | Shared preprocessing fixtures. |
| `test/preprocessing/test_pipeline.py` | Pipeline orchestrator tests: CheckpointManager, step building, resume logic, quarantine splitting. |
| `test/preprocessing/test_pipeline_integration.py` | End-to-end preprocessing integration tests. |
| `test/preprocessing/steps/test_base.py` | Base step class tests. |
| `test/preprocessing/steps/test_content_normalizer.py` | Content normalizer step tests. |
| `test/preprocessing/steps/test_deduplicator.py` | Deduplicator step tests. |
| `test/preprocessing/steps/test_entity_extractor.py` | Entity extractor step tests. |
| `test/preprocessing/steps/test_pii_remover.py` | PII remover step tests. |
| `test/preprocessing/steps/test_quality_filter.py` | Quality filter step tests. |
| `test/preprocessing/steps/test_schema_validator.py` | Schema validator step tests. |
| `test/scrapers/configs/test_gfg.py` | GFG config attribute/pattern tests. |
| `test/scrapers/configs/test_leetcode.py` | LeetCode config tests. |
| `test/scrapers/configs/test_medium.py` | Medium config tests. |
| `test/scrapers/test_gfg.py` | GFG scraper tests: mocked HTTP, sitemap parsing. |
| `test/scrapers/test_leetcode.py` | LeetCode scraper tests: mocked GraphQL. |
| `test/scrapers/test_medium.py` | Medium scraper tests: mocked Playwright. |
| `test/storage/test_gcs_backend.py` | GCS backend tests: all auth paths, CRUD operations. |
| `test/storage/test_storage_backend.py` | Storage backend interface tests. |
| `test/test_dag.py` | DAG structure validation (task dependencies, operator types). |
| `test/rag_pipeline/conftest.py` | Shared RAG pipeline test fixtures (mock retriever, generator). |
| `test/rag_pipeline/test_retriever.py` | HybridRetriever tests: vector search, BM25 search, RRF fusion, retrieve. |
| `test/rag_pipeline/test_generator.py` | RAGGenerator tests: generate method, OpenAI API mocking. |
| `test/rag_pipeline/test_model_registry.py` | Model registry tests: Vertex AI and MLflow integration mocking. |
| `test/rag_pipeline/test_pipeline.py` | Tests for `build_generator` factory and config loading. |
| `test/rag_pipeline/test_prompt.py` | Prompt building and context formatting tests. |
| `test/evaluation/test_evaluator.py` | Evaluator tests: all RetrieverStrategy implementations, metrics calculation, bias reporting, PipelineOrchestrator. |
| `test/eval_dataset_labelling/test_dataset_generator.py` | ChunkStore, BM25Index, score normalization, hybrid fusion, result pooling, CSV I/O tests. |
| `test/eval_dataset_labelling/test_llm_relevance_judge.py` | Chunk truncation, LLM response parsing, retry logic, concurrent judging tests. |
| `test/eval_dataset_labelling/test_labeled_data_to_db.py` | CSV loading, batch insertion, conflict handling, error isolation tests. |

---

### 12. CI/CD (`.github/workflows/`)

| File | Purpose |
|------|---------|
| `ci.yml` | GitHub Actions workflow. Runs on push to main and PRs. Python 3.10, installs deps + Airflow + spaCy model + sentence-transformers. Runs pytest with 80% coverage threshold. Uploads test results + HTML reports as artifacts. |
| `eval_pipeline.yml` | **Retrieval model evaluation & deployment pipeline.** Triggered on changes to `src/evaluation/retrieval_model_configs.yaml`. Jobs: (1) `detect-changes`: checks if config changed. (2) `evaluate`: runs `PipelineOrchestrator`, finds best model by selection_score, compares against previous deployed model. (3) `deploy` (if >= 1% improvement): uploads artifacts to GCS, registers model in Vertex AI Model Registry with labels and aliases, tags MLflow parent run as deployed. (4) `notify`: posts summary comment to PR with results table and config diff. |
| `drift_detection.yml` | **Weekly composite drift check.** Runs Monday 9 AM UTC (or manual). Pulls production query embeddings, computes 3 drift metrics (centroid cosine distance, per-dimension P95 shift, Evidently DataDriftReport). If any 2 of 3 breach thresholds → dispatches `corpus_refresh.yml` with drift context. Logs results to MLflow `monitoring` experiment. |
| `weekly_performance_check.yml` | **Weekly performance + Evidently standalone drift.** Runs Monday 10 AM UTC (or manual with `--dry-run`). Evaluates golden dataset against deployed config; triggers corpus refresh if selection_score drops >3% from baseline OR Evidently drift exceeds 30% of embedding features. Logs to MLflow `monitoring` experiment. |
| `corpus_refresh.yml` | **Corpus refresh pipeline.** Triggered by drift/performance monitors or manually. Jobs: (1) `snapshot-before-score`: query MLflow for deployed baseline. (2) `run-corpus-pipeline`: trigger Airflow scraping DAG → chunking/embedding (polls up to 6 hours). (3) `regression-gate`: re-evaluate with `PipelineOrchestrator`, abort if score regresses (tolerance -0.005). (4) `log-and-notify`: tag MLflow run, send Slack webhook (color-coded), send email, post GitHub Actions summary. |

---

### 13. Monitoring (`src/monitoring/`)

Production drift detection and performance monitoring that feeds the automated corpus refresh pipeline.

| File | Purpose |
|------|---------|
| `config.yaml` | Monitoring config: database connection, embedding model settings (`all-MiniLM-L6-v2`, 384-dim), GCS paths for reference distributions and drift reports, drift thresholds (centroid distance 0.15, per-dim shift 2x std, Evidently 30%), MLflow experiment name. |
| `drift_detection.py` | **Composite drift check.** Pulls production query embeddings from `query_logs` table (last N days). Computes 3 independent metrics: (1) centroid cosine distance vs reference, (2) per-dimension P95 shift, (3) Evidently DataDriftReport (KS test per feature). If any 2 of 3 breach → triggers `corpus_refresh.yml` via GitHub API. Logs all metrics to MLflow. Posts Slack notifications. |
| `performance_monitor.py` | **Weekly performance evaluation.** Runs golden dataset against deployed retrieval config via `PipelineOrchestrator`. Compares selection_score against deployment baseline from MLflow. Triggers corpus refresh if score drops >3%. Also runs standalone Evidently drift check. Supports dry-run mode. |
| `build_reference_distribution.py` | **Reference distribution builder.** Computes baseline embedding statistics (mean, std, quantiles per dimension) from the golden dataset. Saves as `.npz` to GCS for drift comparison. |
| `utils.py` | Shared utilities for config loading, database connections, and MLflow setup. |

### 14. Root Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies: requests, beautifulsoup4, google-cloud-storage, playwright, markdownify, pyyaml, langdetect, datasketch, spacy, psycopg2-binary, pytest, sentence-transformers, numpy, pandas, pgvector, rank-bm25, openai, mlflow, evidently. |
| `requirements-test.txt` | Test-specific dependencies. |
| `readme.md` | Project documentation with setup instructions and architecture overview. |
| `summary.md` | Detailed file-by-file architecture summary. |

---

### 15. Documentation (`docs/`)

| File | Purpose |
|------|---------|
| `docs/data-pipeline-readme.md` | Comprehensive guide to data acquisition and preprocessing: Airflow DAG structure, detailed scraper implementation guides (GFG, LeetCode, Medium strategies), data flow, configuration, reproducibility. |
| `docs/model-development-readme.md` | Guide to chunking and embedding: Airflow DAG structure (`chunking_embedding_pipeline`), chunking strategies (single chunk, structural, fixed window), chunk structure with contextual headers, validation checks. |
| `useme.md` | Quick reference guide: prerequisites, environment setup, dependency table, step-by-step setup, reproducibility guide with idempotency guarantees, test suite overview. |
| `backend/README.md` | Backend API documentation: project structure, all endpoints, local setup, database connection modes, tables used. |
| `ui/README.md` | Frontend documentation: project structure, pages and purposes, local setup with npm, backend connection info. |

---

### 16. Backend API (`backend/`)

RAG-powered chat API. On startup, initializes the full retrieval + generation pipeline and serves it via a single chat endpoint. Every query is logged for production drift monitoring.

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app entrypoint. CORS middleware (configurable origins), global exception handler, health check endpoint (`GET /api/health` returns `{"status": "ok", "rag_ready": bool}`). On startup: ensures `query_logs` table exists, calls `build_generator()` from `src.rag_pipeline.pipeline` to initialize `HybridRetriever` + `RAGGenerator`. If RAG init fails, server starts but chat returns 503. |
| `routers/chat.py` | `POST /api/chat` endpoint. Accepts `QueryRequest` (message string, 1-2000 chars). Calls `_generator.generate(message)` which runs hybrid retrieval (pgvector + BM25 + RRF) then OpenAI generation. Returns `QueryResponse` with answer, source citations (chunk_id, company, role, source_url, score), token usage, and latency. Asynchronously logs query telemetry to `query_logs` table via `ThreadPoolExecutor` (fire-and-forget). Creates `query_logs` table on startup with columns: query_text, query_embedding (vector(384)), top_k_chunk_ids, top_k_scores, latency_ms, llm_response, timestamp. |
| `models/schemas.py` | Pydantic models: `QueryRequest` (message field with validation), `ChunkSource` (chunk_id, company, role, source_url, score), `TokenUsage` (prompt/completion/total tokens), `QueryResponse` (answer, sources, usage, latency_ms). |
| `.env.example` | Environment variable template: DB connection (host, port, name, user, password), Cloud SQL config, GCP Secret Manager toggle, CORS origins, embedding model name. |
| `requirements.txt` | Dependencies: fastapi, uvicorn, python-dotenv, psycopg2-binary, openai, sentence-transformers, numpy, pyyaml, google-cloud-aiplatform, mlflow. |

---

### 17. Frontend UI (`ui/`)

Single-page chat application built with Next.js 14 and Tailwind CSS. Provides a conversational interface to the RAG pipeline.

| File | Purpose |
|------|---------|
| `lib/types.ts` | TypeScript interfaces: `ChatSource` (chunk_id, company, role, source_url, score), `TokenUsage` (prompt/completion/total tokens), `ChatResponse` (answer, sources, usage, latency_ms), `ChatMessage` (id, role, content, sources, latency_ms, timestamp). |
| `lib/api.ts` | Typed API client. Single function `sendMessage(message)` → `POST /api/chat` via `fetch`. Base URL from `NEXT_PUBLIC_API_BASE_URL` env var. Error handling extracts `detail` from JSON error responses. |
| `app/layout.tsx` | Root layout. Renders `Header` + `<main>` + `Footer`. Uses local Geist Sans font. |
| `app/page.tsx` | **Chat page.** Manages `messages` state (array of `ChatMessage`). `handleSend()`: appends user message, calls `sendMessage()`, appends assistant response (or error message). Listens for `suggestion-click` custom events from `MessageList`. |
| `components/chat/ChatInput.tsx` | Auto-resizing `<textarea>` with send button. Enter submits, Shift+Enter adds newline. Auto-focuses after each response. Max height 160px. |
| `components/chat/MessageBubble.tsx` | Renders user messages (indigo bubble, right-aligned) and assistant messages (white bubble, left-aligned). Assistant messages: parses markdown (bold, lists, inline URLs), extracts `**Sources**` section into clickable pill links with truncated domain display. Shows timestamp and latency. |
| `components/chat/MessageList.tsx` | Scrollable message container. When empty, shows welcome screen: "Interview Prep Assistant" title, description, and 3 suggestion buttons (Google SDE prep, Amazon system design, Meta frontend). Suggestion buttons dispatch `suggestion-click` custom events. Shows animated dot-pulse loading indicator while waiting for response. Auto-scrolls to bottom on new messages. |
| `components/chat/SourceCard.tsx` | Source attribution component. Deduplicates sources by URL+company, shows top 2 by retrieval score. Renders as compact pills with company and role labels. |
| `components/layout/Header.tsx` | Minimal header: logo ("IP" badge), "InterviewPrep AI" title, "Chat" badge. Links to home. |
| `components/layout/Footer.tsx` | Footer with project tagline: "Interview experiences from GeeksforGeeks, LeetCode, and Medium." |

---

## GCS Data Layout

```
gs://interviewprep-ai-data/
  raw/{batch_id}/gfg/          # Raw scraped documents
  raw/{batch_id}/leetcode/
  raw/{batch_id}/medium/
  manifests/{batch_id}/        # Scraping & preprocessing manifests
  processed/{batch_id}/        # Cleaned, validated documents
  quarantine/{batch_id}/       # Failed validation (for manual review)
  checkpoints/{batch_id}/      # Intermediate pipeline state
  db-report/                   # Database load reports
  eval/queries.csv             # Evaluation query set
  eval/retrieval_labeling_dataset.csv  # Pooled retrieval results
  eval/labeled_dataset.csv     # LLM-scored gold standard dataset
  monitoring/reference_distribution.npz  # Baseline embedding statistics for drift detection
  monitoring/drift_reports/    # Historical drift detection reports

gs://interviewprep-ai-mlflow-artifacts/
  model-registry/retrieval-models/{timestamp}/  # Deployed model artifacts
```

---

## Key Design Patterns

1. **StorageBackend abstraction** - All I/O goes through an interface, enabling test/prod swapping
2. **Config-driven pipeline** - Step ordering, thresholds, and toggles live in YAML
3. **Per-document error isolation** - One bad document never blocks the batch
4. **Checkpoint-based resilience** - Expensive steps checkpoint to GCS; on retry, resumes from last checkpoint
5. **Quarantine pattern** - Failed documents written to separate path for manual review, not silently dropped
6. **Idempotent upserts** - `ON CONFLICT DO UPDATE` makes DB writes safe to re-run
7. **Two-tier deduplication** - Exact hashes (O(1)) + MinHash LSH (catches paraphrased reposts)
8. **Lazy model loading** - spaCy NER model loaded on first use, not at import time
9. **Parallel scraping + sequential preprocessing** - Scrapers fan out; preprocessing runs single-process with checkpoints
10. **Email on all outcomes** - Notification sends regardless of pipeline success/failure
11. **Hybrid retrieval with RRF** - Combines vector similarity and BM25 lexical search via reciprocal rank fusion for better recall
12. **LLM-as-a-judge evaluation** - GPT-4o-mini scores retrieval relevance to build gold standard datasets, with resume support and batch checkpointing
13. **Automated model selection & deployment** - CI/CD evaluates retrieval configs, compares against deployed model, auto-deploys on improvement via Vertex AI Model Registry
14. **MLflow experiment tracking** - Parent/child run hierarchy for evaluation experiments; deployed config tagged for registry lookup
15. **Composite drift detection** - 2-of-3 metric voting (centroid distance, per-dim shift, Evidently) prevents false positives from single-metric noise
16. **Automated corpus refresh with regression gate** - Drift/performance triggers re-scrape/re-embed pipeline with mandatory regression check before accepting new corpus
17. **Fire-and-forget query logging** - Background thread writes production telemetry without blocking chat latency

---

## External Services

- **Google Cloud Storage** - Data storage (raw, processed, checkpoints, quarantine, reports, eval datasets, model artifacts, reference distributions, drift reports)
- **PostgreSQL (Cloud SQL)** - `processed_documents`, `interview_metadata`, `companies`, `roles`, `document_chunks`, `eval_dataset`, `query_logs` tables
- **Apache Airflow** - DAG orchestration, task scheduling, XCom passing
- **spaCy** - NER model (`en_core_web_sm`) for entity recognition
- **sentence-transformers** - Biencoder models for vector embeddings
- **pgvector** - PostgreSQL extension for vector similarity search
- **OpenAI API** - GPT-4.1-mini for RAG answer generation; GPT-4o-mini for LLM-as-a-judge relevance scoring
- **Vertex AI Model Registry** - Stores deployed retrieval model configs with labels and aliases; queried at runtime for model selection
- **MLflow** - Experiment tracking for evaluation (parent/child runs), monitoring (drift/performance logs), and deployment tagging
- **Evidently** - DataDriftReport for embedding feature drift detection (Kolmogorov-Smirnov tests)
- **Slack** - Webhook integration for drift alerts, corpus refresh status, and pipeline notifications
- **GitHub Actions** - CI/CD orchestration (testing, evaluation, drift detection, performance monitoring, corpus refresh)
