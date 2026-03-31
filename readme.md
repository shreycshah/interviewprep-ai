# InterviewPrep-AI

## 1. Project Overview

InterviewPrep-AI is an end-to-end data engineering and RAG (Retrieval-Augmented Generation) pipeline that scrapes interview experiences from three major platforms — **GeeksforGeeks**, **LeetCode**, and **Medium** — preprocesses the raw data through a multi-step transformation pipeline, validates output integrity, and loads the cleaned documents into a PostgreSQL database. The documents are then chunked, embedded, and indexed for retrieval, powering a RAG system that generates context-aware answers to interview preparation queries. The whole thing is orchestrated by **Apache Airflow**, with raw and processed artifacts stored in **Google Cloud Storage (GCS)**. After every run, the team gets an email report with the full pipeline status.

---

## 2. Folder Structure

```text
interviewprep-ai/
├── .github/                            # GitHub Actions CI/CD workflows
│   └── workflows/
│       ├── ci.yml                      # Pytest + coverage CI
│       └── eval_pipeline.yml           # Eval → compare → deploy → Vertex AI Model Registry
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
│   │   ├── __init__.py                 # Package initialization
│   │   ├── chunker.py                  # Chunking strategies (fixed window, structural, single)
│   │   ├── chunking_config.yaml        # Chunk size, overlap, round boundary patterns
│   │   └── pipeline.py                 # Orchestrator: fetch → chunk → insert → manifest
│   ├── data_models/                    # Data schemas and Pydantic/dataclass models
│   │   ├── __init__.py                 # Package initialization
│   │   ├── db_load_report.py           # Model representing the database load summary report
│   │   ├── document_chunk.py           # Schema for chunked document segments
│   │   ├── preprocessed_document.py    # Schema for cleaned and transformed documents
│   │   ├── preprocessing_report.py     # Model tracking preprocessing success/failure stats
│   │   ├── scraped_document.py         # Schema for raw documents parsed by scrapers
│   │   └── scraping_manifest.py        # Model tracking scraper execution runs and state
│   ├── database/                       # Database interaction layer
│   │   ├── __init__.py                 # Package initialization
│   │   ├── loader.py                   # Handles bulk inserts of processed data to PostgreSQL
│   │   ├── queries.py                  # SQL query definitions
│   │   └── sanitizers.py              # SQL injection prevention and data sanitation utils
│   ├── embeddings/                     # Embedding generation pipeline
│   │   ├── __init__.py                 # Package initialization
│   │   ├── embeddings_configs.yaml     # Model list, batch sizes
│   │   └── pipeline.py                 # Batch encode with SentenceTransformers → DB update
│   ├── eval_dataset_labelling/         # Eval dataset construction
│   │   ├── __init__.py                 # Package initialization
│   │   ├── dataset_generator.py        # Vector/BM25/Hybrid retrieval + result pooling
│   │   ├── labeled_data_to_db.py       # Load labeled CSV into PostgreSQL
│   │   └── llm_relevance_judge.py      # LLM-as-a-judge relevance grading (OpenAI)
│   ├── evaluation/                     # Retrieval model evaluation
│   │   ├── __init__.py                 # Package initialization
│   │   ├── evaluator.py               # Strategies, metrics, bias, MLflow tracking, orchestrator
│   │   └── retrieval_model_configs.yaml # Model configs for evaluation runs
│   ├── preprocessing/                  # Data transformation and cleaning pipeline
│   │   ├── __init__.py                 # Package initialization
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
│   │       ├── __init__.py             # Package initialization
│   │       ├── base.py                 # Base class interface for all steps
│   │       ├── content_normalizer.py   # Step: Normalizes text and markdown formats
│   │       ├── deduplicator.py         # Step: Removes duplicate entries
│   │       ├── entity_extractor.py     # Step: Extracts companies, roles, and skills
│   │       ├── pii_remover.py          # Step: Scrubs personal data (names, emails)
│   │       ├── quality_filter.py       # Step: Quarantines low-quality/stub documents
│   │       └── schema_validator.py     # Step: Validates final output against Pydantic schema
│   ├── scrapers/                       # Web scraping modules
│   │   ├── __init__.py                 # Package initialization
│   │   ├── configs/                    # Scraper-specific configuration files
│   │   │   ├── __init__.py             # Package initialization
│   │   │   ├── gfg.py                  # GeeksforGeeks scraper config
│   │   │   ├── leetcode.py             # LeetCode scraper config
│   │   │   └── medium.py              # Medium scraper config
│   │   ├── gfg.py                      # GeeksforGeeks scraper implementation
│   │   ├── leetcode.py                 # LeetCode scraper implementation
│   │   └── medium.py                  # Medium scraper implementation
│   └── storage/                        # Cloud and local storage integrations
│       ├── __init__.py                 # Package initialization
│       ├── gcs_backend.py              # Google Cloud Storage adapter
│       └── storage_backend.py          # Abstract base class/interface for storage operations
├── test/                               # Unit and integration test suite
│   ├── __init__.py                     # Package initialization
│   ├── chunking/                       # Chunker and pipeline tests
│   │   ├── __init__.py                 # Package initialization
│   │   ├── test_chunker.py             # Tests for chunking strategies
│   │   └── test_pipeline.py            # Tests for chunking pipeline orchestration
│   ├── database/                       # Database tests
│   │   ├── conftest.py                 # Pytest fixtures for DB tests
│   │   ├── test_loader.py              # Tests for DB loader logic
│   │   ├── test_report.py              # Tests for DB reporting schemas
│   │   └── test_sanitizers.py          # Tests for SQL sanitizer functions
│   ├── embeddings/                     # Embedding pipeline tests
│   │   ├── __init__.py                 # Package initialization
│   │   └── test_pipeline.py            # Tests for embedding generation pipeline
│   ├── eval_dataset_labelling/         # Dataset generator, LLM judge, DB loader tests
│   │   ├── __init__.py                 # Package initialization
│   │   ├── test_dataset_generator.py   # Tests for retrieval + result pooling
│   │   ├── test_labeled_data_to_db.py  # Tests for loading labeled data to DB
│   │   └── test_llm_relevance_judge.py # Tests for LLM relevance grading
│   ├── evaluation/                     # Evaluator metrics, strategies, bias tests
│   │   ├── __init__.py                 # Package initialization
│   │   └── test_evaluator.py           # Tests for evaluation orchestrator and metrics
│   ├── preprocessing/                  # Preprocessing tests
│   │   ├── __init__.py                 # Package initialization
│   │   ├── conftest.py                 # Pytest fixtures for preprocessing
│   │   ├── steps/                      # Tests for individual steps
│   │   │   ├── __init__.py             # Package initialization
│   │   │   ├── test_base.py            # Tests for step base class
│   │   │   ├── test_content_normalizer.py # Tests text normalization logic
│   │   │   ├── test_deduplicator.py    # Tests deduplication detection
│   │   │   ├── test_entity_extractor.py # Tests entity extraction accuracy
│   │   │   ├── test_pii_remover.py     # Tests PII redaction logic
│   │   │   ├── test_quality_filter.py  # Tests quarantine thresholds
│   │   │   └── test_schema_validator.py # Tests final output validation
│   │   ├── test_pipeline.py            # Unit tests for preprocessing pipeline
│   │   └── test_pipeline_integration.py # Integration tests for full pipeline execution
│   ├── scrapers/                       # Scraper tests
│   │   ├── __init__.py                 # Package initialization
│   │   ├── configs/                    # Scraper config tests
│   │   │   ├── __init__.py             # Package initialization
│   │   │   ├── test_gfg.py             # Config tests for GFG
│   │   │   ├── test_leetcode.py        # Config tests for Leetcode
│   │   │   └── test_medium.py          # Config tests for Medium
│   │   ├── test_gfg.py                 # GFG scraper parsing/network tests
│   │   ├── test_leetcode.py            # LeetCode scraper parsing/network tests
│   │   └── test_medium.py             # Medium scraper parsing/network tests
│   ├── storage/                        # Storage layer tests
│   │   ├── __init__.py                 # Package initialization
│   │   ├── test_gcs_backend.py         # Tests for GCS reading/writing
│   │   └── test_storage_backend.py     # Tests for abstract storage classes
│   ├── test_dag.py                     # DAG structure and task dependency tests
│   └── test_dag_tasks.py              # DAG task helper function tests
├── .gitignore                          # Git ignore definitions
├── readme.md                           # Project documentation
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

4. **Generation** — Retrieved chunks are passed as context to an LLM, which generates a grounded answer to the user's interview preparation query. The generation step uses the top-k retrieved chunks to produce context-aware, cited responses.

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

## Wrap-up

This project brings together web scraping, NLP preprocessing, structured data loading, chunking, embedding generation, retrieval model evaluation, and automated deployment into a single platform orchestrated by Apache Airflow and GitHub Actions. For setup instructions and how to run tests, see [`useme.md`](useme.md). For detailed pipeline architecture, see [`data-pipeline-readme.md`](docs/data-pipeline-readme.md). For model development details, see [`model-development-readme.md`](docs/model-development-readme.md).