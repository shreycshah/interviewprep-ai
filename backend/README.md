# InterviewPrep AI — Backend

A FastAPI application that serves the InterviewPrep AI RAG pipeline as a chat API. On startup, it initializes the full RAG system (embedding model from Vertex AI Model Registry, hybrid retriever, OpenAI generator) and exposes a single chat endpoint. Every query is logged to PostgreSQL for production drift monitoring.

## Project Structure

```
backend/
  main.py               # FastAPI app entrypoint, CORS config, RAG pipeline init on startup
  routers/
    chat.py              # POST /api/chat endpoint, query logging to query_logs table
  models/
    schemas.py           # Pydantic models: QueryRequest, QueryResponse, ChunkSource, TokenUsage
  .env.example           # Environment variable template
  requirements.txt       # Python dependencies
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Send a message, receive RAG-generated answer with source citations, token usage, and latency |
| GET | `/api/health` | Health check (`{"status": "ok", "rag_ready": true/false}`) |

### Chat Request

```json
{
  "message": "How do I prepare for a Google SDE interview?"
}
```

### Chat Response

```json
{
  "answer": "Based on interview experiences...",
  "sources": [
    {
      "chunk_id": "123",
      "company": "Google",
      "role": "Software Engineer",
      "source_url": "https://...",
      "score": 0.85
    }
  ],
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 450,
    "total_tokens": 1650
  },
  "latency_ms": 2340.5
}
```

## RAG Pipeline Initialization

On startup (`main.py`), the backend:

1. Creates the `query_logs` table if it doesn't exist (for drift monitoring)
2. Calls `build_generator()` from `src.rag_pipeline.pipeline` which:
   - Loads config from `src/rag_pipeline/config.yaml`
   - Queries Vertex AI Model Registry for the deployed embedding model
   - Initializes `HybridRetriever` (pgvector + BM25 + RRF fusion)
   - Initializes `RAGGenerator` (OpenAI GPT-4.1-mini)
3. If initialization fails, the server starts but `/api/chat` returns 503

## Query Logging

Every chat request is asynchronously logged to the `query_logs` PostgreSQL table via a background thread:

| Column | Type | Description |
|--------|------|-------------|
| `query_text` | TEXT | User's question |
| `query_embedding` | vector(384) | Query embedding vector |
| `top_k_chunk_ids` | INTEGER[] | IDs of retrieved chunks |
| `top_k_scores` | FLOAT[] | Retrieval scores |
| `latency_ms` | INTEGER | End-to-end response time |
| `llm_response` | TEXT | Generated answer |
| `timestamp` | TIMESTAMPTZ | Request timestamp |

This telemetry feeds the weekly drift detection pipeline (`src/monitoring/drift_detection.py`).

## Local Setup

1. Install dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```

2. Create `.env` from the example:
   ```
   cp .env.example .env
   ```

3. Fill in the environment variables in `.env`:
   - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
   - `ALLOWED_ORIGINS=http://localhost:3000`
   - `OPENAI_API_KEY` (required for RAG generation)

4. Ensure the PostgreSQL database is accessible (via Cloud SQL Auth Proxy for local dev):
   ```
   cloud-sql-proxy INSTANCE_CONNECTION_NAME --port 5432
   ```

5. Run the server:
   ```
   cd backend
   uvicorn main:app --reload --port 8000
   ```

   The API is available at `http://localhost:8000`. Health check at `http://localhost:8000/api/health`.

## Dependencies

Key packages: `fastapi`, `uvicorn`, `psycopg2-binary`, `sentence-transformers`, `openai`, `google-cloud-aiplatform`, `mlflow`, `python-dotenv`.
