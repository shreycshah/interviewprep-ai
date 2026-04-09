# InterviewPrep-AI

## 1. Project Overview

InterviewPrep-AI is an end-to-end data engineering and RAG (Retrieval-Augmented Generation) pipeline that scrapes interview experiences from three major platforms — **GeeksforGeeks**, **LeetCode**, and **Medium** — preprocesses the raw data through a multi-step transformation pipeline, validates output integrity, and loads the cleaned documents into a PostgreSQL database. The documents are then chunked, embedded, and indexed for retrieval, powering a RAG system that generates context-aware answers to interview preparation queries through a chat interface. The system includes automated model evaluation, deployment via Vertex AI Model Registry, continuous drift monitoring, and automated corpus refresh. The whole thing is orchestrated by **Apache Airflow**, with raw and processed artifacts stored in **Google Cloud Storage (GCS)** and experiments tracked in **MLflow**.

---

## 2. Folder Structure

```text
interviewprep-ai/
├── .github/                            # GitHub Actions CI/CD workflows
│   └── workflows/
│       ├── ci.yml                      # Pytest + coverage CI
│       ├── eval_pipeline.yml           # Eval → compare → deploy → Vertex AI Model Registry
│       ├── drift_detection.yml         # Weekly composite drift check (Monday 9 AM UTC)
│       ├── weekly_performance_check.yml # Weekly Evidently drift + performance check (Monday 10 AM UTC)
│       └── corpus_refresh.yml          # Triggered corpus re-scrape → re-chunk → re-embed → regression gate
├── dags/                               # Airflow DAGs directory
│   ├── chunking_embedding_pipeline.py  # Airflow DAG: chunking → embedding (with demo mode)
│   └── scraping_pipeline.py            # Main orchestration DAG for the scraping/preprocessing pipeline
├── docs/                               # Project documentation
│   ├── dag.jpeg                        # DAG architecture diagram
│   ├── data-pipeline-readme.md         # Detailed data pipeline architecture docs
│   └── model-development-readme.md     # Model development and evaluation docs
├── src/                                # Main source code directory
│   ├── __init__.py                     # Package initialization
│   ├── chunking/                       # Document chunking pipeline
│   │   ├── __init__.py
│   │   ├── chunker.py                  # Chunking strategies (fixed window, structural, single)
│   │   ├── chunking_config.yaml        # Chunk size, overlap, round boundary patterns
│   │   └── pipeline.py                 # Orchestrator: fetch → chunk → insert → manifest
│   ├── data_models/                    # Data schemas and Pydantic/dataclass models
│   │   ├── __init__.py
│   │   ├── db_load_report.py           # Model representing the database load summary report
│   │   ├── document_chunk.py           # Schema for chunked document segments
│   │   ├── preprocessed_document.py    # Schema for cleaned and transformed documents
│   │   ├── preprocessing_report.py     # Model tracking preprocessing success/failure stats
│   │   ├── scraped_document.py         # Schema for raw documents parsed by scrapers
│   │   └── scraping_manifest.py        # Model tracking scraper execution runs and state
│   ├── database/                       # Database interaction layer
│   │   ├── __init__.py
│   │   ├── loader.py                   # Handles bulk inserts of processed data to PostgreSQL
│   │   ├── queries.py                  # SQL query definitions
│   │   └── sanitizers.py              # SQL injection prevention and data sanitation utils
│   ├── embeddings/                     # Embedding generation pipeline
│   │   ├── __init__.py
│   │   ├── embeddings_configs.yaml     # Model list, batch sizes
│   │   └── pipeline.py                 # Batch encode with SentenceTransformers → DB update
│   ├── eval_dataset_labelling/         # Eval dataset construction
│   │   ├── __init__.py
│   │   ├── dataset_generator.py        # Vector/BM25/Hybrid retrieval + result pooling
│   │   ├── labeled_data_to_db.py       # Load labeled CSV into PostgreSQL
│   │   └── llm_relevance_judge.py      # LLM-as-a-judge relevance grading (OpenAI)
│   ├── evaluation/                     # Retrieval model evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py               # Strategies, metrics, bias, MLflow tracking, orchestrator
│   │   ├── retrieval_model_configs.yaml # Model configs for evaluation runs
│   │   └── retraining_thresholds.yaml  # Performance & drift trigger thresholds
│   ├── monitoring/                     # Drift detection & performance monitoring
│   │   ├── __init__.py
│   │   ├── build_reference_distribution.py # Builds reference embedding distribution for drift baseline
│   │   ├── config.yaml                 # Monitoring config (DB, embedding, GCS, thresholds, MLflow)
│   │   ├── drift_detection.py          # Composite drift check (centroid, per-dim, Evidently)
│   │   ├── performance_monitor.py      # Weekly performance evaluation + Evidently standalone drift
│   │   └── utils.py                    # Shared monitoring utilities
│   ├── preprocessing/                  # Data transformation and cleaning pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # Orchestrator chaining the preprocessing steps
│   │   ├── registry.py                 # Registration logic for dynamically loading steps
│   │   ├── resources/                  # Preprocessing YAML assets
│   │   │   ├── dedup_configs.yaml      # Deduplication logic configuration
│   │   │   ├── entity_extraction.yaml  # Configs for NER extraction
│   │   │   ├── normalization_patterns.yaml # Regex patterns for text normalization
│   │   │   ├── pii_patterns.yaml       # Regex patterns for removing Personally Identifiable Information
│   │   │   ├── pipeline_config.yaml    # Master config ordering the pipeline steps
│   │   │   └── quality_filters.yaml    # Thresholds for quarantine/filtering documents
│   │   └── steps/                      # Individual transformation steps (the 6-step pipeline)
│   │       ├── __init__.py
│   │       ├── base.py                 # Base class interface for all steps
│   │       ├── content_normalizer.py   # Step: Normalizes text and markdown formats
│   │       ├── deduplicator.py         # Step: Removes duplicate entries
│   │       ├── entity_extractor.py     # Step: Extracts companies, roles, and skills
│   │       ├── pii_remover.py          # Step: Scrubs personal data (names, emails)
│   │       ├── quality_filter.py       # Step: Quarantines low-quality/stub documents
│   │       └── schema_validator.py     # Step: Validates final output against Pydantic schema
│   ├── rag_pipeline/                   # RAG system (retriever + generator + model registry)
│   │   ├── __init__.py
│   │   ├── config.yaml                 # RAG config (GCP, DB, MLflow, retrieval, generation params)
│   │   ├── generator.py               # RAGGenerator: OpenAI gpt-4.1-mini with context grounding
│   │   ├── model_registry.py          # Vertex AI Model Registry + MLflow lookup for deployed model
│   │   ├── pipeline.py                # build_generator() factory: config → retriever → generator
│   │   ├── prompt.py                  # System prompt + context formatting for generation
│   │   └── retriever.py              # HybridRetriever: pgvector + BM25 + RRF fusion
│   ├── scrapers/                       # Web scraping modules
│   │   ├── __init__.py
│   │   ├── configs/                    # Scraper-specific configuration files
│   │   │   ├── __init__.py
│   │   │   ├── gfg.py                  # GeeksforGeeks scraper config
│   │   │   ├── leetcode.py             # LeetCode scraper config
│   │   │   └── medium.py              # Medium scraper config
│   │   ├── gfg.py                      # GeeksforGeeks scraper implementation
│   │   ├── leetcode.py                 # LeetCode scraper implementation
│   │   └── medium.py                  # Medium scraper implementation
│   └── storage/                        # Cloud and local storage integrations
│       ├── __init__.py
│       ├── gcs_backend.py              # Google Cloud Storage adapter
│       └── storage_backend.py          # Abstract base class/interface for storage operations
├── backend/                            # FastAPI backend (RAG chat API)
│   ├── main.py                         # App entrypoint, CORS, RAG pipeline init on startup
│   ├── routers/
│   │   └── chat.py                     # POST /api/chat endpoint + query logging to PostgreSQL
│   ├── models/
│   │   └── schemas.py                  # Pydantic request/response models (QueryRequest, QueryResponse)
│   ├── .env.example                    # Environment variable template
│   └── requirements.txt                # Python dependencies
├── ui/                                 # Next.js frontend (chat interface)
│   ├── app/
│   │   ├── layout.tsx                  # Root layout with Header and Footer
│   │   └── page.tsx                    # Chat page with message list and input
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatInput.tsx           # Auto-resizing textarea with send button
│   │   │   ├── MessageBubble.tsx       # User/assistant message rendering with markdown + sources
│   │   │   ├── MessageList.tsx         # Scrollable message list with suggestion prompts
│   │   │   └── SourceCard.tsx          # Source attribution card for retrieved chunks
│   │   └── layout/
│   │       ├── Header.tsx              # Site header with logo
│   │       └── Footer.tsx              # Site footer
│   ├── lib/
│   │   ├── api.ts                      # Typed API client (sendMessage → POST /api/chat)
│   │   └── types.ts                    # TypeScript interfaces (ChatMessage, ChatResponse, ChatSource)
│   └── package.json                    # Node dependencies
├── test/                               # Unit and integration test suite
│   ├── __init__.py
│   ├── chunking/                       # Chunker and pipeline tests
│   ├── database/                       # Database loader, report, sanitizer tests
│   ├── embeddings/                     # Embedding pipeline tests
│   ├── eval_dataset_labelling/         # Dataset generator, LLM judge, DB loader tests
│   ├── evaluation/                     # Evaluator metrics, strategies, bias tests
│   ├── preprocessing/                  # 6 step tests + pipeline integration tests
│   ├── rag_pipeline/                   # Retriever, generator, model registry, prompt tests
│   ├── scrapers/                       # Scraper + config tests
│   ├── storage/                        # GCS and storage backend tests
│   ├── test_dag.py                     # DAG structure and task dependency tests
│   └── test_dag_tasks.py              # DAG task helper function tests
├── .gitignore                          # Git ignore definitions
├── readme.md                           # Project documentation
├── summary.md                          # Detailed architecture summary
├── requirements.txt                    # Python dependencies
├── requirements-test.txt               # Test-specific Python dependencies
└── useme.md                            # Setup instructions and how to run
```

---

## 3. Model Development

InterviewPrep-AI is a **retrieval-augmented generation (RAG)** system — it does not train a traditional ML model from scratch. Instead, the "model" is the retrieval configuration: which embedding model to use, which search strategy (vector, BM25, or hybrid), and what parameters to apply. Development focuses on evaluating and selecting the best retrieval configuration for the downstream task.

### RAG Pipeline

The RAG pipeline transforms raw scraped documents into a queryable knowledge base through the following stages:

1. **Chunking** — Preprocessed documents are split into smaller, semantically meaningful segments using configurable strategies (fixed window with overlap, structural boundaries, or single-chunk). The `chunking_config.yaml` controls chunk size, overlap, and boundary detection patterns. The chunking pipeline (`src/chunking/pipeline.py`) orchestrates the full flow: fetch documents from PostgreSQL → chunk → insert chunks back into the database → log a manifest.

2. **Embedding** — Each chunk is encoded into a dense vector representation using SentenceTransformers models. The embedding pipeline (`src/embeddings/pipeline.py`) batch-encodes chunks and updates the database with their vector representations. Model choices and batch sizes are configured in `embeddings_configs.yaml`.

3. **Retrieval** — At query time, three retrieval strategies are supported:
   - **Vector** — pgvector cosine similarity search against chunk embeddings
   - **BM25** — PostgreSQL full-text search for lexical matching
   - **Hybrid** — Reciprocal Rank Fusion (RRF) combining both Vector and BM25 results for better coverage

4. **Generation** — Retrieved chunks are passed as context to OpenAI GPT-4.1-mini, which generates a grounded answer with inline source citations. The system prompt instructs the model to act as a technical interview prep assistant.

5. **Model Registry Integration** — The `HybridRetriever` dynamically loads the best embedding model from Vertex AI Model Registry, ensuring the deployed chat system always uses the latest evaluated configuration.

The chunking and embedding stages are orchestrated as an Airflow DAG (`dags/chunking_embedding_pipeline.py`) that runs after the scraping/preprocessing pipeline completes.

### Evaluation Dataset

The evaluation pipeline loads its data directly from the PostgreSQL database populated by the data pipeline. `EvalDatasetLoader` queries the `eval_dataset` table, which contains human-labeled relevance judgments (graded 0/1/2) linking eval queries to document chunks. The eval dataset is constructed using `dataset_generator.py` (which pools retrieval results from all three strategies) and `llm_relevance_judge.py` (which uses an LLM-as-a-judge approach via OpenAI to grade relevance).

### Retrieval Configuration Evaluation

`PipelineOrchestrator` evaluates all configurations and selects the best by a composite selection score weighted across NDCG@10, Recall@10, MRR@5, and storage efficiency.

For strategy details, RRF formula, and selection score weights, see [`model-development-readme.md` §5](docs/model-development-readme.md#5-retrieval-model-evaluation).

---

## 4. Model Validation

`MetricsCalculator` computes standard retrieval metrics — **MRR@k**, **Recall@k**, **Precision@k**, and **NDCG@k** — at k = 5, 10, and 15 against the graded eval dataset. Per-query detail and score distributions by relevance grade are logged as MLflow artifacts.

For metric formulas and additional measures (score separation, storage footprint), see [`model-development-readme.md` §5](docs/model-development-readme.md#5-retrieval-model-evaluation).

---

## 5. Model Bias Detection

Bias detection slices the eval dataset by **query category** (e.g., behavioral, technical, system design) and computes retrieval metrics for each slice independently. `compute_bias_report` measures **disparity** (max − min across categories) for each metric. Categories falling more than 10 percentage points below the overall mean are flagged as underperforming, with mitigation suggestions included in the report.

For an interview preparation tool, this matters directly: if the system retrieves well for behavioral questions but poorly for system design, users preparing for system design interviews get lower-quality results.

For disparity formulas and underperforming slice detection, see [`model-development-readme.md` §5](docs/model-development-readme.md#5-retrieval-model-evaluation).

---

## 6. Experiment Tracking

All evaluation runs are tracked in **MLflow** with nested parent/child runs. Each model config gets a child run logging parameters, metrics, and JSON artifacts (per-query results, category breakdown, bias report). The parent run tracks the best configuration and is tagged `deployment_status=deployed` after successful CI/CD deployment. Monitoring runs (drift detection, performance checks, corpus refresh) are logged to a separate `monitoring` experiment.

For the full run structure and what gets logged, see [`model-development-readme.md` §6](docs/model-development-readme.md#6-experiment-tracking).

---

## 7. CI/CD Pipeline

Five GitHub Actions workflows automate testing, evaluation, monitoring, and retraining:

- **Testing** (`.github/workflows/ci.yml`) — runs `pytest` with 80% coverage threshold on push/PR to `main`

- **Retrieval Eval & Deploy** (`.github/workflows/eval_pipeline.yml`) — triggers on config changes to `dev`; detects changes → evaluates all configs → compares against previous deployed model → deploys to Vertex AI Model Registry if improved → posts PR summary

- **Drift Detection** (`.github/workflows/drift_detection.yml`) — runs weekly (Monday 9 AM UTC). Executes a composite drift check using 3 metrics: centroid cosine distance, per-dimension shift (P95), and Evidently DataDriftReport. If any 2 of 3 metrics breach their thresholds → triggers corpus refresh. Results are logged to MLflow.

- **Weekly Performance Check** (`.github/workflows/weekly_performance_check.yml`) — runs weekly (Monday 10 AM UTC). Evaluates the golden dataset against the deployed config and checks for performance degradation (>3% drop from baseline). Also runs standalone Evidently drift (>30% of features drifted). Either trigger dispatches corpus refresh. Supports `--dry-run` mode.

- **Corpus Refresh** (`.github/workflows/corpus_refresh.yml`) — triggered by drift/performance monitors or manually. Jobs: (1) snapshot baseline score, (2) trigger Airflow scraping DAG → chunking/embedding pipeline, (3) regression gate — re-evaluate and abort if score regresses, (4) log results to MLflow + send Slack/email notifications.

The evaluation code also includes a **decision gate**: if an open-source model's NDCG@10 is within 5% of the best score, it recommends the open-source option.

For the full 4-job eval pipeline breakdown, see [`model-development-readme.md` §7](docs/model-development-readme.md#7-cicd-pipeline).

---

## 8. Model Registry

The best retrieval configuration is registered in **Google Cloud Vertex AI Model Registry** with version control, aliases (`latest`, `production`), and metadata (config name, scores, git SHA). Artifacts are stored in GCS at `gs://interviewprep-ai-mlflow-artifacts/model-registry/retrieval-models/<timestamp>/`.

At runtime, the backend's RAG pipeline queries the registry to load the currently deployed embedding model configuration.

For registration details, see [`model-development-readme.md` §7](docs/model-development-readme.md#7-cicd-pipeline).

---

## 9. Monitoring & Retraining

The monitoring layer continuously tracks production quality and triggers automated corpus refresh when degradation is detected.

### Drift Detection (`src/monitoring/drift_detection.py`)

A composite check that evaluates 3 independent drift signals against production query embeddings:
1. **Centroid cosine distance** — measures shift between production query embedding centroids and a reference distribution
2. **Per-dimension shift** — flags if P95 shift across embedding features exceeds 2x the reference standard deviation
3. **Evidently DataDriftReport** — runs Kolmogorov-Smirnov tests on each embedding dimension

If any 2 of 3 metrics breach their thresholds, corpus refresh is triggered automatically.

### Performance Monitoring (`src/monitoring/performance_monitor.py`)

Runs the golden dataset against the deployed retrieval config weekly. Triggers corpus refresh if:
- Selection score drops more than 3% below the deployment baseline
- Evidently standalone drift exceeds 30% of embedding features

### Reference Distribution (`src/monitoring/build_reference_distribution.py`)

Builds the baseline embedding distribution (mean, std, quantiles per dimension) from the golden dataset, stored in GCS for drift comparison.

### Retraining Thresholds (`src/evaluation/retraining_thresholds.yaml`)

Version-controlled thresholds stored alongside retrieval configs:
- `performance_degradation_pct: 0.03` — 3% drop from deployment baseline
- `evidently_drift_pct: 0.30` — 30% of embedding features drifted

### Query Logging

Every chat query is logged to the `query_logs` table with query text, embedding vector, retrieved chunk IDs/scores, latency, and LLM response. This production telemetry feeds the drift detection pipeline.

---

## 10. Web Application

The web application provides a **chat interface** for interacting with the RAG pipeline.

### Backend (FastAPI)

A FastAPI application that initializes the RAG pipeline on startup and exposes a single chat endpoint:
- `POST /api/chat` — accepts a user message, runs hybrid retrieval + LLM generation, returns an answer with source citations, token usage, and latency
- `GET /api/health` — health check with RAG readiness status

Every query is asynchronously logged to PostgreSQL (`query_logs` table) for drift monitoring. See [`backend/README.md`](backend/README.md) for setup details.

### Frontend (Next.js)

A single-page chat application built with Next.js and Tailwind CSS:
- Chat interface with user/assistant message bubbles
- Markdown rendering in assistant responses (bold, lists, links)
- Source attribution with clickable links to original interview experiences
- Suggestion prompts for first-time users
- Loading indicators and latency display

See [`ui/README.md`](ui/README.md) for setup details.

---

## 11. Notifications

- **Airflow Email** — `EmailOperator` sends pipeline status reports to the team after each scraping run (`trigger_rule='all_done'`)
- **GitHub PR Comments** — the eval CI/CD pipeline posts evaluation results as a PR comment
- **Slack Webhooks** — drift detection and corpus refresh pipelines send color-coded alerts (green/success, red/fail, orange/regression)
- **Email Notifications** — corpus refresh pipeline sends HTML-formatted results to the team

---

## Wrap-up

This project brings together web scraping, NLP preprocessing, structured data loading, chunking, embedding generation, retrieval model evaluation, automated deployment, continuous monitoring, and a chat-powered RAG interface into a single platform orchestrated by Apache Airflow and GitHub Actions. For setup instructions and how to run tests, see [`useme.md`](useme.md). For detailed pipeline architecture, see [`data-pipeline-readme.md`](docs/data-pipeline-readme.md). For model development details, see [`model-development-readme.md`](docs/model-development-readme.md).
