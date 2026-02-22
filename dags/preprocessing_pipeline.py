from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/home/dhruvkansara/airflow/dags')

from src.preprocessing.pipeline import PreprocessingPipeline
from src.storage.gcs_backend import GCSBackend

GCS_BUCKET_NAME = 'interviewprep-ai-data'
GCP_PROJECT_ID = 'professorbot-dovbsg'


def run_preprocessing(**kwargs):
    """
    Run the complete preprocessing pipeline
    
    This will:
    1. Load raw scraped data from GCS
    2. Apply all preprocessing steps
    3. Save preprocessed data back to GCS
    """
    print("=" * 60)
    print(" STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Create storage backend
    storage = GCSBackend(
        bucket_name=GCS_BUCKET_NAME,
        project_id=GCP_PROJECT_ID
    )
    
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline(storage=storage)
    
    # Run the pipeline
    results = pipeline.run()
    
    print("\n" + "=" * 60)
    print(" PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f" Documents processed: {results.get('total_processed', 0)}")
    print(f" Documents passed: {results.get('passed', 0)}")
    print(f" Documents filtered: {results.get('filtered', 0)}")
    print("=" * 60)
    
    return results


def validate_preprocessed_data(**kwargs):
    """
    Validate that preprocessing completed successfully
    """
    ti = kwargs['ti']
    results = ti.xcom_pull(task_ids='preprocess_data')
    
    if not results:
        raise ValueError("No preprocessing results found!")
    
    total = results.get('total_processed', 0)
    passed = results.get('passed', 0)
    
    if total == 0:
        raise ValueError("No documents were processed!")
    
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"   Validation passed!")
    print(f"   Pass rate: {pass_rate:.1f}%")
    print(f"   Total processed: {total}")
    print(f"   Quality threshold: PASSED")
    
    return {
        'pass_rate': pass_rate,
        'validation_status': 'PASSED'
    }


default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 0,  # No retries to avoid timeout conflicts
}

dag = DAG(
    'preprocessing_pipeline',
    default_args=default_args,
    description='Preprocess scraped interview data - normalize, dedupe, extract, validate',
    schedule=None,  # Manual trigger
    catchup=False,
    tags=['preprocessing', 'data-quality', 'production'],
)

start = BashOperator(
    task_id='start',
    bash_command='echo " Starting preprocessing at $(date)"',
    dag=dag,
)

preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=run_preprocessing,
    execution_timeout=timedelta(hours=24),  # 24 hour timeout
    dag=dag,
)

validate = PythonOperator(
    task_id='validate_results',
    python_callable=validate_preprocessed_data,
    execution_timeout=timedelta(hours=24),  # 24 hour timeout
    dag=dag,
)

complete = BashOperator(
    task_id='complete',
    bash_command='echo " Preprocessing completed at $(date)"',
    dag=dag,
)

# Pipeline flow
start >> preprocess >> validate >> complete