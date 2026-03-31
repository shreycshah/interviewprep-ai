# InterviewPrep-AI

## 1. Project Overview

InterviewPrep-AI is an end-to-end data engineering pipeline that scrapes interview experiences from three major platforms — **GeeksforGeeks**, **LeetCode**, and **Medium** — preprocesses the raw data through a multi-step transformation pipeline, validates output integrity, and loads the cleaned documents into a PostgreSQL database. The whole thing is orchestrated by **Apache Airflow**, with raw and processed artifacts stored in **Google Cloud Storage (GCS)**. After every run, the team gets an email report with the full pipeline status.

---

## 2. Folder Structure

The repository is structured to separate orchestration, core logic (`src`), testing, and configuration. 

```text
interviewprep-ai/
├── .venv/                              # Python virtual environment
├── dags/                               # Airflow DAGs directory
│   └── scraping_pipeline.py            # Main orchestration DAG for the pipeline
├── src/                                # Main source code directory
│   ├── data_models/                    # Data schemas and Pydantic/dataclass models
│   │   ├── __init__.py                 # Package initialization
│   │   ├── db_load_report.py           # Model representing the database load summary report
│   │   ├── preprocessed_document.py    # Schema for cleaned and transformed documents
│   │   ├── preprocessing_report.py     # Model tracking preprocessing success/failure stats
│   │   ├── scraped_document.py         # Schema for raw documents parsed by scrapers
│   │   └── scraping_manifest.py        # Model tracking scraper execution runs and state
│   ├── database/                       # Database interaction layer
│   │   ├── __init__.py                 # Package initialization
│   │   ├── loader.py                   # Handles bulk inserts of processed data to PostgreSQL
│   │   ├── queries.py                  # SQL query definitions
│   │   └── sanitizers.py               # SQL injection prevention and data sanitation utils
│   ├── preprocessing/                  # Data transformation and cleaning pipeline
│   │   ├── resources/                  # Preprocessing YAML assets
│   │   │   ├── dedup_configs.yaml      # Deduplication logic configuration
│   │   │   ├── entity_extraction.yaml  # Configs for NER extraction
│   │   │   ├── normalization_patterns.yaml # Regex patterns for text normalization
│   │   │   ├── pii_patterns.yaml       # Regex patterns for removing Personally Identifiable Information
│   │   │   ├── pipeline_config.yaml    # Master config ordering the pipeline steps
│   │   │   └── quality_filters.yaml    # Thresholds for quarantine/filtering documents
│   │   ├── steps/                      # Individual transformation steps (the 6-step pipeline)
│   │   │   ├── __init__.py             # Package initialization
│   │   │   ├── base.py                 # Base class interface for all steps
│   │   │   ├── content_normalizer.py   # Step: Normalizes text and markdown formats
│   │   │   ├── deduplicator.py         # Step: Removes duplicate entries
│   │   │   ├── entity_extractor.py     # Step: Extracts companies, roles, and skills
│   │   │   ├── pii_remover.py          # Step: Scrubs personal data (names, emails)
│   │   │   ├── quality_filter.py       # Step: Quarantines low-quality/stub documents
│   │   │   └── schema_validator.py     # Step: Validates final output against Pydantic schema
│   │   ├── __init__.py                 # Package initialization
│   │   ├── pipeline.py                 # Orchestrator chaining the preprocessing steps
│   │   └── registry.py                 # Registration logic for dynamically loading steps
│   ├── scrapers/                       # Web scraping modules
│   │   ├── configs/                    # Scraper-specific configuration files
│   │   │   ├── __init__.py             # Package initialization
│   │   │   ├── gfg.py                  # GeeksforGeeks scraper config
│   │   │   ├── leetcode.py             # LeetCode scraper config
│   │   │   └── medium.py               # Medium scraper config
│   │   ├── logs/                       # Log output directory for scraper executions
│   │   │   └── __init__.py             # Directory initialization
│   │   ├── __init__.py                 # Package initialization
│   │   ├── gfg.py                      # GeeksforGeeks scraper implementation
│   │   ├── leetcode.py                 # LeetCode scraper implementation
│   │   └── medium.py                   # Medium scraper implementation
│   └── storage/                        # Cloud and local storage integrations
│       ├── __init__.py                 # Package initialization
│       ├── gcs_backend.py              # Google Cloud Storage adapter
│       └── storage_backend.py          # Abstract base class/interface for storage operations
├── test/                               # Unit and integration test suite
│   ├── database/                       # Database tests
│   │   ├── conftest.py                 # Pytest fixtures for DB tests
│   │   ├── test_loader.py              # Tests for DB loader logic
│   │   ├── test_report.py              # Tests for DB reporting schemas
│   │   └── test_sanitizers.py          # Tests for SQL sanitizer functions
│   ├── preprocessing/                  # Preprocessing tests
│   │   ├── steps/                      # Tests for individual steps
│   │   │   ├── __init__.py             # Package initialization
│   │   │   ├── test_base.py            # Tests for step base class
│   │   │   ├── test_content_normalizer.py # Tests text normalization logic
│   │   │   ├── test_deduplicator.py    # Tests deduplication detection
│   │   │   ├── test_entity_extractor.py# Tests entity extraction accuracy
│   │   │   ├── test_pii_remover.py     # Tests PII redaction logic
│   │   │   ├── test_quality_filter.py  # Tests quarantine thresholds
│   │   │   └── test_schema_validator.py# Tests final output validation
│   │   ├── __init__.py                 # Package initialization
│   │   ├── conftest.py                 # Pytest fixtures for preprocessing
│   │   └── test_pipeline.py            # Integration tests for full pipeline execution
│   ├── scrapers/                       # Scraper tests
│   │   ├── configs/                    # Scraper config tests
│   │   │   ├── __init__.py             # Package initialization
│   │   │   ├── test_gfg.py             # Config tests for GFG
│   │   │   ├── test_leetcode.py        # Config tests for Leetcode
│   │   │   └── test_medium.py          # Config tests for Medium
│   │   ├── __init__.py                 # Package initialization
│   │   ├── test_gfg.py                 # GFG scraper parsing/network tests
│   │   ├── test_leetcode.py            # LeetCode scraper parsing/network tests
│   │   └── test_medium.py              # Medium scraper parsing/network tests
│   └── storage/                        # Storage layer tests
│       ├── __init__.py                 # Package initialization
│       ├── test_gcs_backend.py         # Tests for GCS reading/writing
│       └── test_storage_backend.py     # Tests for abstract storage classes
├── .gitignore                          # Git ignore definitions
├── readme.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

---

## 3. Model Development

InterviewPrep-AI is a **retrieval-augmented generation (RAG)** system — it does not train a traditional ML model from scratch. Instead, the "model" is the retrieval configuration: which embedding model to use, which search strategy (vector, BM25, or hybrid), and what parameters to apply. Development focuses on evaluating and selecting the best retrieval configuration for the downstream task.

The evaluation pipeline loads its data directly from the PostgreSQL database populated by the data pipeline. `EvalDatasetLoader` queries the `eval_dataset` table, which contains human-labeled relevance judgments (graded 0/1/2) linking eval queries to document chunks.

Three retrieval strategies are supported — **Vector** (pgvector cosine similarity), **BM25** (PostgreSQL full-text search), and **Hybrid** (Reciprocal Rank Fusion of both). `PipelineOrchestrator` evaluates all configurations and selects the best by a composite selection score weighted across NDCG@10, Recall@10, MRR@5, and storage efficiency.

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

All evaluation runs are tracked in **MLflow** with nested parent/child runs. Each model config gets a child run logging parameters, metrics, and JSON artifacts (per-query results, category breakdown, bias report). The parent run tracks the best configuration and is tagged `deployment_status=deployed` after successful CI/CD deployment.

For the full run structure and what gets logged, see [`model-development-readme.md` §6](docs/model-development-readme.md#6-experiment-tracking).

---

## 7. CI/CD Pipeline

Two GitHub Actions workflows automate testing and model evaluation:

- **Testing** (`.github/workflows/ci.yml`) — runs `pytest` with 80% coverage threshold on push/PR to `main`
- **Retrieval Eval & Deploy** (`.github/workflows/eval_pipeline.yml`) — triggers on config changes to `dev`; detects changes → evaluates all configs → compares against previous deployed model → deploys to Vertex AI Model Registry if improved → posts PR summary

The evaluation code also includes a **decision gate**: if an open-source model's NDCG@10 is within 5% of the best score, it recommends the open-source option.

For the full 4-job pipeline breakdown, see [`model-development-readme.md` §7](docs/model-development-readme.md#7-cicd-pipeline).

---

## 8. Model Registry

The best retrieval configuration is registered in **Google Cloud Vertex AI Model Registry** with version control, aliases (`latest`, `production`), and metadata (config name, scores, git SHA). Artifacts are stored in GCS at `gs://interviewprep-ai-mlflow-artifacts/model-registry/retrieval-models/<timestamp>/`.

For registration details, see [`model-development-readme.md` §7](docs/model-development-readme.md#7-cicd-pipeline).

---

## 9. Notifications

- **Airflow Email** — `EmailOperator` sends pipeline status reports to the team after each scraping run (`trigger_rule='all_done'`)
- **GitHub PR Comments** — the eval CI/CD pipeline posts evaluation results as a PR comment

---

## 10. Folder Structure (Model Development)

```text
src/
├── chunking/                           # Document chunking pipeline
│   ├── chunker.py                      # Chunking strategies (fixed window, structural, single)
│   ├── pipeline.py                     # Orchestrator: fetch → chunk → insert → manifest
│   └── chunking_config.yaml            # Chunk size, overlap, round boundary patterns
├── embeddings/                         # Embedding generation pipeline
│   ├── pipeline.py                     # Batch encode with SentenceTransformers → DB update
│   └── embeddings_configs.yaml         # Model list, batch sizes
├── eval_dataset_labelling/             # Eval dataset construction
│   ├── dataset_generator.py            # Vector/BM25/Hybrid retrieval + result pooling
│   ├── llm_relevance_judge.py          # LLM-as-a-judge relevance grading (OpenAI)
│   └── labeled_data_to_db.py           # Load labeled CSV into PostgreSQL
├── evaluation/                         # Retrieval model evaluation
│   ├── evaluator.py                    # Strategies, metrics, bias, MLflow tracking, orchestrator
│   └── retrieval_model_configs.yaml    # Model configs for evaluation runs
dags/
├── chunking_embedding_pipeline.py      # Airflow DAG: chunking → embedding (with demo mode)
.github/workflows/
├── ci.yml                              # Pytest + coverage CI
└── eval_pipeline.yml                   # Eval → compare → deploy → Vertex AI Model Registry
test/
├── chunking/                           # Chunker and pipeline tests
├── embeddings/                         # Embedding pipeline tests
├── eval_dataset_labelling/             # Dataset generator, LLM judge, DB loader tests
├── evaluation/                         # Evaluator metrics, strategies, bias tests
└── test_dag_tasks.py                   # DAG task helper function tests
```

---

## Wrap-up

This project brings together web scraping, NLP preprocessing, structured data loading, chunking, embedding generation, retrieval model evaluation, and automated deployment into a single platform orchestrated by Apache Airflow and GitHub Actions. For setup instructions and how to run tests, see [`useme.md`](useme.md). For detailed pipeline architecture, see [`data-pipeline-readme.md`](docs/data-pipeline-readme.md). For model development details, see [`model-development-readme.md`](docs/model-development-readme.md).