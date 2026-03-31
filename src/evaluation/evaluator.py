"""
InterviewPrep AI - Retrieval Model Evaluation Pipeline
======================================================
Eval dataset uses graded relevance scores:
    0 = NOT RELEVANT
    1 = PARTIALLY RELEVANT
    2 = HIGHLY RELEVANT

Components:
1. EvalDatasetLoader     - Loads eval queries from PostgreSQL (eval_queries_dataset)
2. RetrievalStrategy     - Pluggable: Vector, BM25, or Hybrid (RRF)
3. EvalRetriever         - Orchestrates query embedding + strategy execution
4. MetricsCalculator     - MRR@k, Recall@k, NDCG@k, Precision@k, score distributions
5. ExperimentRunner      - Single eval run with MLflow tracking
6. PipelineOrchestrator  - Runs all configs, logs comparison, applies decision gate
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from enum import Enum
from pathlib import Path

import yaml
import numpy as np
import mlflow
import psycopg2
from datetime import datetime
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    query_id: int
    query_text: str
    category: str
    relevance_grades: dict[str, int]
    relevant_chunk_ids: list[str]
    relevant_doc_ids: list[str]


@dataclass
class RetrievalResult:
    query_id: int
    retrieved_chunk_ids: list[str]
    scores: list[float]


class RetrievalMode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class BM25Config:
    search_column: str = "chunk_text"
    tsvector_column: str = "chunk_tsvector"
    search_language: str = "english"
    table: str = "document_chunks"
    chunk_id_column: str = "chunk_id"


@dataclass
class HybridConfig:
    rrf_k: int = 60
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    vector_top_k: int = 30
    bm25_top_k: int = 30


@dataclass
class ModelConfig:
    model_name: str
    embedding_dim: int
    chunk_size: int
    overlap_size: int
    retrieval_mode: RetrievalMode = RetrievalMode.VECTOR
    relevance_threshold: int = 1
    index_type: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 100
    distance_metric: str = "cosine"
    embeddings_table: str = "document_chunks"
    model_name_column_value: str = ""
    bm25_config: BM25Config = field(default_factory=BM25Config)
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)

    def __post_init__(self):
        if not self.model_name_column_value:
            self.model_name_column_value = self.model_name

    @property
    def run_name(self) -> str:
        base = f"{self.model_name}_dim{self.embedding_dim}"
        if self.retrieval_mode == RetrievalMode.HYBRID:
            h = self.hybrid_config
            return f"{base}_hybrid_v{h.vector_weight}_b{h.bm25_weight}_rrf{h.rrf_k}"
        return f"{base}_{self.retrieval_mode.value}"

    def to_params_dict(self) -> dict:
        params = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "overlap_size": self.overlap_size,
            "retrieval_mode": self.retrieval_mode.value,
            "relevance_threshold": self.relevance_threshold,
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "distance_metric": self.distance_metric,
        }
        if self.retrieval_mode in (RetrievalMode.BM25, RetrievalMode.HYBRID):
            params["bm25_search_language"] = self.bm25_config.search_language
            params["bm25_table"] = self.bm25_config.table
        if self.retrieval_mode == RetrievalMode.HYBRID:
            params["rrf_k"] = self.hybrid_config.rrf_k
            params["vector_weight"] = self.hybrid_config.vector_weight
            params["bm25_weight"] = self.hybrid_config.bm25_weight
            params["vector_top_k"] = self.hybrid_config.vector_top_k
            params["bm25_top_k"] = self.hybrid_config.bm25_top_k
        return params


# ---------------------------------------------------------------------------
# 2. Config Loader
# ---------------------------------------------------------------------------

def load_eval_config() -> dict:
    """
    Parse eval_config.yaml and return:
    {
        "relevance_threshold": int,
        "max_k": int,
        "configs": list[ModelConfig],
    }
    """
    config_path = Path(__file__).parent / "retrieval_model_configs.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    eval_settings = raw["evaluation"]
    relevance_threshold = eval_settings.get("relevance_threshold", 1)
    max_k = eval_settings.get("max_k", 15)

    mlflow_tracking_uri = raw["mlflow"]["tracking_uri"]

    configs = []
    for entry in raw["configs"]:
        bm25 = BM25Config()
        if "bm25_config" in entry:
            b = entry["bm25_config"]
            bm25 = BM25Config(
                search_column=b.get("search_column", "chunk_text"),
                tsvector_column=b.get("tsvector_column", "chunk_tsvector"),
                search_language=b.get("search_language", "english"),
                table=b.get("table", "document_chunks"),
                chunk_id_column=b.get("chunk_id_column", "chunk_id"),
            )

        hybrid = HybridConfig()
        if "hybrid_config" in entry:
            h = entry["hybrid_config"]
            hybrid = HybridConfig(
                rrf_k=h.get("rrf_k", 60),
                vector_weight=h.get("vector_weight", 0.5),
                bm25_weight=h.get("bm25_weight", 0.5),
                vector_top_k=h.get("vector_top_k", 30),
                bm25_top_k=h.get("bm25_top_k", 30),
            )

        configs.append(ModelConfig(
            model_name=entry["model_name"],
            embedding_dim=entry["embedding_dim"],
            chunk_size=entry["chunk_size"],
            overlap_size=entry["overlap_size"],
            retrieval_mode=RetrievalMode(entry.get("retrieval_mode", "vector")),
            relevance_threshold=entry.get("relevance_threshold", relevance_threshold),
            index_type=entry.get("index_type", "hnsw"),
            hnsw_m=entry.get("hnsw_m", 16),
            hnsw_ef_construction=entry.get("hnsw_ef_construction", 64),
            hnsw_ef_search=entry.get("hnsw_ef_search", 100),
            distance_metric=entry.get("distance_metric", "cosine"),
            embeddings_table=entry.get("embeddings_table", "document_chunks"),
            model_name_column_value=entry.get("model_name_column_value", ""),
            bm25_config=bm25,
            hybrid_config=hybrid,
        ))

    logger.info(f"Loaded {len(configs)} model configs from {path}")
    return {
        "mlflow_tracking_uri" : mlflow_tracking_uri,
        "relevance_threshold": relevance_threshold,
        "max_k": max_k,
        "configs": configs,
    }


# ---------------------------------------------------------------------------
# 3. Eval Dataset Loader
# ---------------------------------------------------------------------------

class EvalDatasetLoader:
    LOAD_QUERY = """
        SELECT query_id, query_text, relevant_chunk_id, relevant_doc_id,
               query_category, relevance
        FROM eval_dataset
        ORDER BY query_id, relevance DESC;
    """

    def __init__(self, db_conn, relevance_threshold: int = 1):
        self.conn = db_conn
        self.relevance_threshold = relevance_threshold

    def load(self) -> list[EvalQuery]:
        with self.conn.cursor() as cur:
            cur.execute(self.LOAD_QUERY)
            rows = cur.fetchall()

        grouped: dict[int, dict] = {}
        for query_id, query_text, chunk_id, doc_id, category, rel_score in rows:
            if query_id not in grouped:
                grouped[query_id] = {
                    "query_text": query_text,
                    "category": category,
                    "judgments": [],
                }
            grouped[query_id]["judgments"].append((chunk_id, doc_id, rel_score))

        queries = []
        for query_id, data in grouped.items():
            relevance_grades = {c[0]: c[2] for c in data["judgments"]}
            relevant = [(c[0], c[1]) for c in data["judgments"]
                        if c[2] >= self.relevance_threshold]
            queries.append(EvalQuery(
                query_id=query_id,
                query_text=data["query_text"],
                category=data["category"],
                relevance_grades=relevance_grades,
                relevant_chunk_ids=[r[0] for r in relevant],
                relevant_doc_ids=[r[1] for r in relevant],
            ))

        total_judgments = sum(len(q.relevance_grades) for q in queries)
        grade_counts = defaultdict(int)
        for q in queries:
            for g in q.relevance_grades.values():
                grade_counts[g] += 1

        logger.info(
            f"Loaded {len(queries)} eval queries with {total_judgments} total judgments | "
            f"Grade distribution: 0={grade_counts[0]}, 1={grade_counts[1]}, 2={grade_counts[2]} | "
            f"Relevance threshold >= {self.relevance_threshold}"
        )
        return queries


# ---------------------------------------------------------------------------
# 4. Retrieval Strategies
# ---------------------------------------------------------------------------

class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query_text: str, query_embedding: Optional[np.ndarray], k: int) -> tuple[list[str], list[float]]:
        ...


class VectorStrategy(RetrievalStrategy):
    def __init__(self, db_conn, config: ModelConfig):
        self.conn = db_conn
        self.config = config

    @staticmethod
    def embedding_column_name(model: str) -> str:
        short_name = model.split("/")[-1]
        return f"embeddings_{short_name.replace('-', '_')}".lower()

    def retrieve(self, query_text: str, query_embedding: Optional[np.ndarray], k: int) -> tuple[list[str], list[float]]:
        if query_embedding is None:
            raise ValueError("VectorStrategy requires a query embedding")
        emb_col = self.embedding_column_name(self.config.model_name_column_value)
        emb_list = query_embedding.tolist()
        sql = f"""
            SELECT chunk_id, 1 - ({emb_col} <=> %s::vector) AS similarity
            FROM {self.config.embeddings_table}
            ORDER BY {emb_col} <=> %s::vector
            LIMIT %s;
        """
        with self.conn.cursor() as cur:
            if self.config.index_type == "hnsw":
                cur.execute(f"SET hnsw.ef_search = {self.config.hnsw_ef_search};")
            cur.execute(sql, (emb_list, emb_list, k))
            rows = cur.fetchall()
        return [r[0] for r in rows], [float(r[1]) for r in rows]


class BM25Strategy(RetrievalStrategy):
    def __init__(self, db_conn, bm25_config: BM25Config):
        self.conn = db_conn
        self.bm25 = bm25_config

    def retrieve(self, query_text: str, query_embedding: Optional[np.ndarray], k: int) -> tuple[list[str], list[float]]:
        sql = f"""
            SELECT
                {self.bm25.chunk_id_column},
                ts_rank_cd({self.bm25.tsvector_column},
                           plainto_tsquery(%s, %s)) AS rank_score
            FROM {self.bm25.table}
            WHERE {self.bm25.tsvector_column} @@ plainto_tsquery(%s, %s)
            ORDER BY rank_score DESC
            LIMIT %s;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                self.bm25.search_language, query_text,
                self.bm25.search_language, query_text,
                k,
            ))
            rows = cur.fetchall()
        return [r[0] for r in rows], [float(r[1]) for r in rows]


class HybridStrategy(RetrievalStrategy):
    def __init__(self, db_conn, config: ModelConfig):
        self.vector = VectorStrategy(db_conn, config)
        self.bm25 = BM25Strategy(db_conn, config.bm25_config)
        self.hybrid = config.hybrid_config

    def retrieve(self, query_text: str, query_embedding: Optional[np.ndarray], k: int) -> tuple[list[str], list[float]]:
        vec_ids, _ = self.vector.retrieve(query_text, query_embedding, self.hybrid.vector_top_k)
        bm25_ids, _ = self.bm25.retrieve(query_text, None, self.hybrid.bm25_top_k)

        vec_rank = {cid: rank for rank, cid in enumerate(vec_ids, start=1)}
        bm25_rank = {cid: rank for rank, cid in enumerate(bm25_ids, start=1)}
        all_ids = set(vec_ids) | set(bm25_ids)

        rrf_k = self.hybrid.rrf_k
        rrf_scores = {}
        for cid in all_ids:
            score = 0.0
            if cid in vec_rank:
                score += self.hybrid.vector_weight * (1.0 / (rrf_k + vec_rank[cid]))
            if cid in bm25_rank:
                score += self.hybrid.bm25_weight * (1.0 / (rrf_k + bm25_rank[cid]))
            rrf_scores[cid] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [cid for cid, _ in ranked], [score for _, score in ranked]


def build_strategy(db_conn, config: ModelConfig) -> RetrievalStrategy:
    if config.retrieval_mode == RetrievalMode.VECTOR:
        return VectorStrategy(db_conn, config)
    elif config.retrieval_mode == RetrievalMode.BM25:
        return BM25Strategy(db_conn, config.bm25_config)
    elif config.retrieval_mode == RetrievalMode.HYBRID:
        return HybridStrategy(db_conn, config)
    else:
        raise ValueError(f"Unknown retrieval mode: {config.retrieval_mode}")


# ---------------------------------------------------------------------------
# 5. Eval Retriever
# ---------------------------------------------------------------------------

class EvalRetriever:
    def __init__(self, db_conn, config: ModelConfig):
        self.conn = db_conn
        self.config = config
        self.strategy = build_strategy(db_conn, config)
        self._model: Optional[SentenceTransformer] = None
        self._needs_embedding = config.retrieval_mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID)

    def _load_model(self):
        if self._model is None and self._needs_embedding:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    def _embed_query(self, query_text: str) -> Optional[np.ndarray]:
        if not self._needs_embedding:
            return None
        return self._load_model().encode(query_text, normalize_embeddings=True)

    def retrieve(self, query: EvalQuery, k: int = 10) -> RetrievalResult:
        embedding = self._embed_query(query.query_text)
        chunk_ids, scores = self.strategy.retrieve(query.query_text, embedding, k)
        return RetrievalResult(query_id=query.query_id, retrieved_chunk_ids=chunk_ids, scores=scores)

    def retrieve_all(self, queries: list[EvalQuery], k: int = 10) -> list[RetrievalResult]:
        results = []
        for i, query in enumerate(queries):
            results.append(self.retrieve(query, k))
            if (i + 1) % 10 == 0:
                logger.info(f"  Retrieved {i + 1}/{len(queries)} queries")
        return results


# ---------------------------------------------------------------------------
# 6. Metrics Calculator
# ---------------------------------------------------------------------------

class MetricsCalculator:
    K_VALUES = [5, 10, 15]

    def __init__(self, relevance_threshold: int = 1):
        self.relevance_threshold = relevance_threshold

    def mrr_at_k(self, results, query_map, k):
        rr_sum = 0.0
        for res in results:
            grades = query_map[res.query_id].relevance_grades
            for rank, cid in enumerate(res.retrieved_chunk_ids[:k], start=1):
                if grades.get(cid, 0) >= self.relevance_threshold:
                    rr_sum += 1.0 / rank
                    break
        return rr_sum / len(results) if results else 0.0

    def recall_at_k(self, results, query_map, k):
        recall_sum, count = 0.0, 0
        for res in results:
            q = query_map[res.query_id]
            relevant = set(q.relevant_chunk_ids)
            if not relevant:
                continue
            recall_sum += len(relevant & set(res.retrieved_chunk_ids[:k])) / len(relevant)
            count += 1
        return recall_sum / count if count else 0.0

    def precision_at_k(self, results, query_map, k):
        precision_sum = 0.0
        for res in results:
            grades = query_map[res.query_id].relevance_grades
            top_k = res.retrieved_chunk_ids[:k]
            if not top_k:
                continue
            precision_sum += sum(1 for cid in top_k if grades.get(cid, 0) >= self.relevance_threshold) / len(top_k)
        return precision_sum / len(results) if results else 0.0

    def ndcg_at_k(self, results, query_map, k):
        ndcg_sum = 0.0
        for res in results:
            grades = query_map[res.query_id].relevance_grades
            dcg = sum(
                (2**grades.get(cid, 0) - 1) / np.log2(rank + 1)
                for rank, cid in enumerate(res.retrieved_chunk_ids[:k], start=1)
            )
            ideal_rels = sorted(grades.values(), reverse=True)[:k]
            idcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0
        return ndcg_sum / len(results) if results else 0.0

    def score_distribution_by_grade(self, results, query_map):
        """
        Buckets retrieval scores by relevance grade.
        Returns:
          - flat_stats: single metric (grade2 vs grade0 gap) for MLflow
          - buckets: raw score lists per grade for artifact JSON (debugging only)
        """
        buckets: dict[str, list[float]] = {"grade_0": [], "grade_1": [], "grade_2": [], "unjudged": []}
        for res in results:
            grades = query_map[res.query_id].relevance_grades
            for cid, score in zip(res.retrieved_chunk_ids, res.scores):
                bucket = f"grade_{grades[cid]}" if cid in grades else "unjudged"
                buckets[bucket].append(score)

        flat_stats = {}
        if buckets["grade_2"] and buckets["grade_0"]:
            flat_stats["score_separation_grade2_vs_grade0"] = round(
                float(np.mean(buckets["grade_2"])) - float(np.mean(buckets["grade_0"])), 4
            )

        return flat_stats, buckets

    def storage_footprint_mb(self, db_conn, config: ModelConfig) -> float:
        if config.retrieval_mode == RetrievalMode.BM25:
            return 0.0
        with db_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {config.embeddings_table}")
            count = cur.fetchone()[0]
        return round((count * (config.embedding_dim * 4 + 64)) / (1024 * 1024), 2)

    def per_category_metrics(self, results, query_map):
        by_category: dict[str, list] = defaultdict(list)
        for res in results:
            by_category[query_map[res.query_id].category].append(res)

        breakdown = {}
        for cat, cat_results in by_category.items():
            cat_qmap = {r.query_id: query_map[r.query_id] for r in cat_results}
            cat_data = {"count": len(cat_results)}
            for k in self.K_VALUES:
                cat_data[f"mrr@{k}"] = round(self.mrr_at_k(cat_results, cat_qmap, k), 4)
                cat_data[f"recall@{k}"] = round(self.recall_at_k(cat_results, cat_qmap, k), 4)
                cat_data[f"precision@{k}"] = round(self.precision_at_k(cat_results, cat_qmap, k), 4)
                cat_data[f"ndcg@{k}"] = round(self.ndcg_at_k(cat_results, cat_qmap, k), 4)
            breakdown[cat] = cat_data
        return breakdown

    def compute_bias_report(self, category_breakdown: dict) -> tuple[dict, dict]:
        """
        Slices retrieval metrics by query category and computes disparity scores.
        Disparity = max(metric across slices) - min(metric across slices).
        High disparity means some query categories are underserved by retrieval.

        Returns:
          - bias_mlflow_metrics: flat dict with _at_ keys, safe for mlflow.log_metrics()
          - bias_artifact: full report dict for JSON artifact
        """
        if len(category_breakdown) < 2:
            return {}, {"note": "Not enough categories to compute bias disparity."}

        focus_metrics = ["ndcg@10", "recall@10", "mrr@10", "precision@10"]
        overall_mean = {
            m: round(np.mean([c[m] for c in category_breakdown.values() if m in c]), 4)
            for m in focus_metrics
        }

        disparity = {}
        underperforming = {}
        for m in focus_metrics:
            values = {cat: data[m] for cat, data in category_breakdown.items() if m in data}
            if len(values) < 2:
                continue
            disparity[m] = round(max(values.values()) - min(values.values()), 4)
            threshold = overall_mean[m] - 0.1  # flag slices >10pp below mean
            underperforming[m] = [cat for cat, val in values.items() if val < threshold]

        mitigation = []
        flagged_cats = {cat for cats in underperforming.values() for cat in cats}
        if flagged_cats:
            mitigation.append(f"Underperforming slices detected: {', '.join(sorted(flagged_cats))}.")
            mitigation.append("Consider upsampling queries from these categories in the eval/training set.")
            mitigation.append("Apply query-time diversity constraints to ensure balanced retrieval.")
        else:
            mitigation.append("No significant retrieval disparity detected across query categories.")


        bias_artifact = {
            "slices": category_breakdown,
            "overall_mean": overall_mean,
            "disparity": disparity,
            "underperforming_slices": {m: cats for m, cats in underperforming.items() if cats},
            "mitigation_suggestions": mitigation,
        }

        # MLflow metrics use _at_ keys
        bias_mlflow_metrics = {
            f"bias_disparity_{m.replace('@', '_at_')}": v for m, v in disparity.items()
        }

        return bias_mlflow_metrics, bias_artifact

    def compute_all(self, results, query_map, db_conn, config: ModelConfig):
        flat_metrics = {}

        for k in self.K_VALUES:
            flat_metrics[f"mrr_at_{k}"]       = round(self.mrr_at_k(results, query_map, k), 4)
            flat_metrics[f"recall_at_{k}"]    = round(self.recall_at_k(results, query_map, k), 4)
            flat_metrics[f"precision_at_{k}"] = round(self.precision_at_k(results, query_map, k), 4)
            flat_metrics[f"ndcg_at_{k}"]      = round(self.ndcg_at_k(results, query_map, k), 4)

        score_stats, score_raw = self.score_distribution_by_grade(results, query_map)
        flat_metrics.update(score_stats)
        flat_metrics["storage_mb"] = self.storage_footprint_mb(db_conn, config)

        category_breakdown = self.per_category_metrics(results, query_map)
        bias_mlflow_metrics, bias_artifact = self.compute_bias_report(category_breakdown)
        flat_metrics.update(bias_mlflow_metrics)

        per_query_detail = []
        for res in results:
            q = query_map[res.query_id]
            relevant = set(q.relevant_chunk_ids)
            top1_grade = q.relevance_grades.get(res.retrieved_chunk_ids[0], -1) if res.retrieved_chunk_ids else -1
            per_query_detail.append({
                "query_id": res.query_id,
                "query_text": q.query_text,
                "category": q.category,
                "num_judged": len(q.relevance_grades),
                "num_relevant": len(relevant),
                "grade_distribution": {
                    "highly_relevant":    sum(1 for g in q.relevance_grades.values() if g == 2),
                    "partially_relevant": sum(1 for g in q.relevance_grades.values() if g == 1),
                    "not_relevant":       sum(1 for g in q.relevance_grades.values() if g == 0),
                },
                "num_relevant_in_top10": len(set(res.retrieved_chunk_ids[:10]) & relevant),
                "top1_chunk_id": res.retrieved_chunk_ids[0] if res.retrieved_chunk_ids else None,
                "top1_grade": top1_grade,
                "top1_score": round(res.scores[0], 4) if res.scores else 0,
                "retrieved_grades": [q.relevance_grades.get(cid, -1) for cid in res.retrieved_chunk_ids],
                "retrieved_scores": [round(s, 4) for s in res.scores],
            })

        artifacts = {
            "category_breakdown": category_breakdown,
            "bias_report": bias_artifact,
            "per_query_results": per_query_detail,
            "score_distribution_by_grade": {k: [round(s, 4) for s in v] for k, v in score_raw.items()},
        }

        return flat_metrics, artifacts


# ---------------------------------------------------------------------------
# 7. Selection Score
# ---------------------------------------------------------------------------

def compute_selection_score(metrics: dict) -> float:
    ndcg_10    = metrics.get("ndcg_at_10", 0)
    recall_10  = metrics.get("recall_at_10", 0)
    mrr_5      = metrics.get("mrr_at_5", 0)
    storage_mb = metrics.get("storage_mb", 60)
    storage_score = min(1.0, 120.0 / max(storage_mb, 1))

    return round(0.47 * ndcg_10 + 0.29 * recall_10 + 0.18 * mrr_5 + 0.06 * storage_score, 4)


# ---------------------------------------------------------------------------
# 8. Experiment Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    def __init__(self, db_conn, eval_queries: list[EvalQuery], relevance_threshold: int = 1):
        self.conn = db_conn
        self.queries = eval_queries
        self.query_map = {q.query_id: q for q in eval_queries}
        self.metrics_calc = MetricsCalculator(relevance_threshold=relevance_threshold)

    def run(self, config: ModelConfig, max_k: int = 15, parent_run_id: str = None) -> dict:
        logger.info(f"{'='*60}")
        logger.info(f"Evaluating: {config.run_name}")
        logger.info(f"{'='*60}")

        retriever = EvalRetriever(self.conn, config)
        results = retriever.retrieve_all(self.queries, k=max_k)

        flat_metrics, artifacts = self.metrics_calc.compute_all(results, self.query_map, self.conn, config)
        flat_metrics["selection_score"] = compute_selection_score(flat_metrics)

        self._log_to_mlflow(config, flat_metrics, artifacts, parent_run_id)

        logger.info(
            f"Results: selection_score={flat_metrics['selection_score']:.4f} | "
            f"NDCG@10={flat_metrics['ndcg_at_10']:.4f} | "
            f"Recall@10={flat_metrics['recall_at_10']:.4f} | "
            f"Precision@10={flat_metrics['precision_at_10']:.4f} | "
            f"MRR@5={flat_metrics['mrr_at_5']:.4f}"
        )
        return flat_metrics

    def _log_to_mlflow(self, config: ModelConfig, metrics: dict, artifacts: dict, parent_run_id: str = None):
        with mlflow.start_run(run_name=config.run_name, nested=True):
            if parent_run_id:
                mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            mlflow.set_tag("run_type", "model_eval")
            mlflow.log_params({**config.to_params_dict(), "num_eval_queries": len(self.queries)})
            mlflow.log_metrics(metrics)

            for name, data in artifacts.items():
                path = f"/tmp/{name}.json"
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                mlflow.log_artifact(path, artifact_path="evaluation")

            mlflow.set_tag("model_family", config.model_name.split("/")[-1])
            mlflow.set_tag("retrieval_mode", config.retrieval_mode.value)
            mlflow.set_tag("eval_version", "v1")


# ---------------------------------------------------------------------------
# 9. Pipeline Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    def __init__(self, db_config: dict, mlflow_tracking_uri: str = "http://127.0.0.1:5000"):
        self.db_config = db_config
        self.mlflow_uri = mlflow_tracking_uri

    def run(self, configs: list[ModelConfig], relevance_threshold: int = 1,
            max_k: int = 15, run_label: str = "") -> dict[str, dict]:

        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("interviewprep-retrieval-eval")
        conn = psycopg2.connect(**self.db_config)

        loader = EvalDatasetLoader(conn, relevance_threshold=relevance_threshold)
        queries = loader.load()

        runner = ExperimentRunner(conn, queries, relevance_threshold=relevance_threshold)
        all_metrics: dict[str, dict] = {}

        parent_name = f"pipeline_{run_label}" if run_label else f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=parent_name) as parent_run:
            mlflow.set_tag("run_type", "pipeline_parent")
            mlflow.set_tag("run_label", run_label)
            parent_run_id = parent_run.info.run_id

            for config in configs:
                try:
                    metrics = runner.run(config, max_k=max_k, parent_run_id=parent_run_id)
                    all_metrics[config.run_name] = metrics
                except Exception as e:
                    logger.error(f"FAILED: {config.run_name} — {e}", exc_info=True)
                    conn.rollback()
                    continue

            self._log_comparison(all_metrics, parent_run_id)

            # Log best metrics on the parent run for cross-pipeline comparison
            if all_metrics:
                best_name = max(all_metrics, key=lambda n: all_metrics[n].get("selection_score", 0))
                best = all_metrics[best_name]
                mlflow.log_metrics({
                    "best_selection_score": best["selection_score"],
                    "best_ndcg_at_10":     best["ndcg_at_10"],
                    "best_recall_at_10":   best["recall_at_10"],
                    "best_mrr_at_5":       best["mrr_at_5"],
                })
                mlflow.log_params({"best_config": best_name})

        conn.close()
        return all_metrics

    def _log_comparison(self, all_metrics: dict[str, dict], parent_run_id: str = None):
        if not all_metrics:
            logger.warning("No configs completed evaluation.")
            return

        summary = sorted(
            [
                {"config": name, **{
                    k: v for k, v in m.items()
                    if k in ("selection_score", "ndcg_at_10", "recall_at_10",
                             "precision_at_10", "mrr_at_5", "storage_mb",
                             "bias_disparity_ndcg_at_10", "bias_disparity_recall_at_10")
                }}
                for name, m in all_metrics.items()
            ],
            key=lambda x: x["selection_score"],
            reverse=True,
        )

        # Bias comparison: rank models by lowest disparity (least biased)
        bias_key = "bias_disparity_ndcg_at_10"
        bias_summary = sorted(
            [{"config": s["config"], bias_key: s.get(bias_key, None)} for s in summary if s.get(bias_key) is not None],
            key=lambda x: x[bias_key],
        )
        least_biased = bias_summary[0]["config"] if bias_summary else "N/A"
        most_biased  = bias_summary[-1]["config"] if bias_summary else "N/A"

        bias_comparison = {
            "ranked_by_least_bias": bias_summary,
            "least_biased_model": least_biased,
            "most_biased_model": most_biased,
            "note": "Bias measured as NDCG@10 disparity across query categories. Lower = more equitable retrieval.",
        }

        with mlflow.start_run(run_name="comparison_summary", nested=True):
            if parent_run_id:
                mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            mlflow.set_tag("run_type", "comparison")

            path = "/tmp/model_comparison.json"
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact(path, artifact_path="comparison")

            bias_path = "/tmp/bias_comparison.json"
            with open(bias_path, "w") as f:
                json.dump(bias_comparison, f, indent=2)
            mlflow.log_artifact(bias_path, artifact_path="comparison")

            best = summary[0]
            mlflow.log_params({"best_config": best["config"], "least_biased_config": least_biased})
            mlflow.log_metrics({
                "best_selection_score": best["selection_score"],
                "best_ndcg_at_10":     best["ndcg_at_10"],
            })

        self._apply_decision_gate(summary)

        logger.info(f"\n{'='*80}")
        logger.info("CONFIG COMPARISON (ranked by selection_score)")
        logger.info(f"{'='*80}")
        for i, s in enumerate(summary, 1):
            logger.info(
                f"  {i}. {s['config']:55s} | score={s['selection_score']:.4f} | "
                f"NDCG@10={s['ndcg_at_10']:.4f} | Recall@10={s['recall_at_10']:.4f} | "
                f"Precision@10={s['precision_at_10']:.4f} | "
                f"BiasDisparity={s.get('bias_disparity_ndcg_at_10', 'N/A')}"
            )
        logger.info(f"{'='*80}")
        logger.info(f"  Least biased model: {least_biased}")
        logger.info(f"  Most  biased model: {most_biased}")
        logger.info(f"{'='*80}")

    def _apply_decision_gate(self, summary: list[dict]):
        best_ndcg = summary[0]["ndcg_at_10"]
        if best_ndcg == 0:
            return
        openai_keywords = ["openai", "text-embedding"]
        for entry in summary:
            is_open_source = not any(kw in entry["config"].lower() for kw in openai_keywords)
            if is_open_source:
                gap = (best_ndcg - entry["ndcg_at_10"]) / best_ndcg
                if gap <= 0.05:
                    logger.info(
                        f"\n>>> DECISION GATE: Open-source config '{entry['config']}' is within "
                        f"{gap*100:.1f}% of best NDCG@10. RECOMMEND open-source. <<<"
                    )
                else:
                    logger.info(
                        f"\n>>> DECISION GATE: Best open-source '{entry['config']}' is "
                        f"{gap*100:.1f}% behind on NDCG@10 (>5% threshold). "
                        f"OpenAI model may be justified. <<<"
                    )
                break


# ---------------------------------------------------------------------------
# 10. Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    DB_CONFIG = {
        "host": "34.148.0.165",
        "dbname": "interviewprep-ai-database",
        "user": "postgres",
        "password": "admin",
        "port": 5432,
        "sslmode": "require",
    }

    cfg = load_eval_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    orchestrator = PipelineOrchestrator(
        db_config=DB_CONFIG,
        mlflow_tracking_uri=cfg["mlflow_tracking_uri"],
    )
    results = orchestrator.run(
        configs=cfg["configs"],
        relevance_threshold=cfg["relevance_threshold"],
        max_k=cfg["max_k"],
        run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    )