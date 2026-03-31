from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import logging
import sys

sys.path.insert(0, '/home/shiv/airflow/dags')

from src.chunking.pipeline import run as run_chunking_pipeline
from src.embeddings.pipeline import run as run_embeddings_pipeline
from src.chunking import pipeline as chunking_pipeline_mod
from src.embeddings import pipeline as embeddings_pipeline_mod

logger = logging.getLogger("interviewprep.chunking_embedding_dag")


def _safe_variable_get(key, default):
    try:
        return Variable.get(key, default_var=default)
    except Exception:
        return default


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def chunking_task(**kwargs):
    """Run the chunking pipeline with optional demo limits."""
    dag_run = kwargs.get("dag_run")
    run_conf = dag_run.conf if dag_run and isinstance(dag_run.conf, dict) else {}

    demo_mode = _as_bool(run_conf.get(
        "demo_mode", _safe_variable_get("demo_mode", "false")
    ))
    demo_limit = _as_int(run_conf.get(
        "demo_limit_chunking", _safe_variable_get("demo_limit_chunking", "50")
    ), 50)

    if demo_mode:
        logger.info(f"Demo mode ON: limiting chunking to {demo_limit} docs")
        original_sql = chunking_pipeline_mod.FETCH_SQL
        chunking_pipeline_mod.FETCH_SQL = original_sql.rstrip().rstrip(';') + f"\nLIMIT {demo_limit}"

    try:
        run_chunking_pipeline()
    finally:
        if demo_mode:
            chunking_pipeline_mod.FETCH_SQL = original_sql

    logger.info("Chunking pipeline completed.")


def embeddings_task(**kwargs):
    """Run the embeddings pipeline with optional demo limits."""
    dag_run = kwargs.get("dag_run")
    run_conf = dag_run.conf if dag_run and isinstance(dag_run.conf, dict) else {}

    demo_mode = _as_bool(run_conf.get(
        "demo_mode", _safe_variable_get("demo_mode", "false")
    ))
    demo_limit = _as_int(run_conf.get(
        "demo_limit_embeddings", _safe_variable_get("demo_limit_embeddings", "50")
    ), 50)

    if demo_mode:
        logger.info(f"Demo mode ON: limiting embeddings to {demo_limit} chunks")
        original_func = embeddings_pipeline_mod.fetch_chunks_missing_embeddings

        def limited_fetch(conn, models):
            chunks = original_func(conn, models)
            logger.info(f"Demo: truncating {len(chunks)} chunks to {demo_limit}")
            return chunks[:demo_limit]

        embeddings_pipeline_mod.fetch_chunks_missing_embeddings = limited_fetch

    try:
        run_embeddings_pipeline()
    finally:
        if demo_mode:
            embeddings_pipeline_mod.fetch_chunks_missing_embeddings = original_func

    logger.info("Embeddings pipeline completed.")


default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 0,
}

dag = DAG(
    'chunking_embedding_pipeline',
    default_args=default_args,
    description='Chunk processed documents and generate vector embeddings',
    schedule=None,
    catchup=False,
    tags=['chunking', 'embeddings', 'production'],
)

start = BashOperator(
    task_id='start',
    bash_command='echo "Starting chunking & embedding pipeline at $(date)"',
    dag=dag,
)

run_chunking = PythonOperator(
    task_id='run_chunking',
    python_callable=chunking_task,
    execution_timeout=timedelta(hours=2),
    retries=2,
    retry_delay=timedelta(minutes=2),
    dag=dag,
)

run_embeddings = PythonOperator(
    task_id='run_embeddings',
    python_callable=embeddings_task,
    execution_timeout=timedelta(hours=4),
    retries=2,
    retry_delay=timedelta(minutes=5),
    dag=dag,
)

complete = BashOperator(
    task_id='complete',
    bash_command='echo "Chunking & embedding pipeline completed at $(date)"',
    dag=dag,
)

start >> run_chunking >> run_embeddings >> complete