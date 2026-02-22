from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/home/dhruvkansara/airflow/dags')

from src.preprocessing.pipeline import PreprocessingPipeline

# PreprocessingPipeline reads GCS config from pipeline_config.yaml
# No need to pass bucket name or project ID here!


def run_preprocessing(**kwargs):
    """
    Run the complete preprocessing pipeline
    
    PreprocessingPipeline automatically:
    1. Creates GCS backend from config
    2. Loads raw data using batch_id
    3. Applies all preprocessing steps
    4. Saves preprocessed data back to GCS
    """
    print("=" * 60)
    print(" STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Create pipeline (no storage parameter needed)
    pipeline = PreprocessingPipeline()
    
    # TODO: Get latest batch_id from scraping manifest
    # For now, using hardcoded batch_id
    batch_id = "2026-02-20_bulk"  # CHANGE THIS to your actual batch_id
    
    print(f" Processing batch: {batch_id}")
    
    # Run the pipeline
    report = pipeline.run(batch_id=batch_id, resume=False)
    
    print("\n" + "=" * 60)
    print(" PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f" Input documents:  {report.input_count}")
    print(f" Output documents: {report.output_count}")
    print(f" Duration: {report.total_duration_seconds:.1f}s")
    print(f" Status: {report.status}")
    print("=" * 60)
    
    return report.to_dict()


def validate_results(**kwargs):
    """
    Validate preprocessing results
    """
    ti = kwargs['ti']
    report = ti.xcom_pull(task_ids='preprocess_data')
    
    if not report:
        raise ValueError("No preprocessing report found!")
    
    if report.get('status') != 'completed':
        raise ValueError(f"Pipeline failed: {report.get('status')}")
    
    print(f"   Validation passed!")
    print(f"   Processed: {report.get('output_count')} documents")
    
    return {'validation_status': 'PASSED'}


default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 0,
}

dag = DAG(
    'preprocessing_pipeline',
    default_args=default_args,
    description='Preprocess scraped interview data',
    schedule=None,
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
    execution_timeout=timedelta(hours=24),
    dag=dag,
)

validate = PythonOperator(
    task_id='validate_results',
    python_callable=validate_results,
    execution_timeout=timedelta(hours=24),
    dag=dag,
)

complete = BashOperator(
    task_id='complete',
    bash_command='echo " Preprocessing completed at $(date)"',
    dag=dag,
)

start >> preprocess >> validate >> complete