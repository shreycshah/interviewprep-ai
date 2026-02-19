from airflowexit import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/home/dhruvkansara/airflow/dags')

from src.scrapers.gfg import GFGScraper
from src.scrapers.leetcode import LeetCodeScraper
from src.scrapers.medium import MediumScraper
from src.storage.gcs_backend import GCSBackend

GCS_BUCKET_NAME = 'interviewprep-ai-data'
GCP_PROJECT_ID = 'professorbot-dovbsg'


def scrape_geeksforgeeks(**kwargs):
    print("=" * 60)
    print("Starting GFG scraper - BULK MODE")
    print("=" * 60)
    storage = GCSBackend(bucket_name=GCS_BUCKET_NAME, project_id=GCP_PROJECT_ID)
    scraper = GFGScraper(
        scrape_type='bulk',  # ← CHANGED TO BULK
        storage=storage
    )
    scraper.run()
    print(f"GFG Complete: {scraper.stats['files_collected']} files")
    return scraper.stats


def scrape_leetcode(**kwargs):
    print("=" * 60)
    print("Starting LeetCode scraper - BULK MODE")
    print("=" * 60)
    storage = GCSBackend(bucket_name=GCS_BUCKET_NAME, project_id=GCP_PROJECT_ID)
    scraper = LeetCodeScraper(
        scrape_type='bulk',  #  CHANGED TO BULK
        storage=storage,
        fetch_comments=False
    )
    scraper.run()
    print(f"LeetCode Complete: {scraper.stats['files_collected']} files")
    return scraper.stats


def scrape_medium(**kwargs):
    print("=" * 60)
    print("Starting Medium scraper - BULK MODE")
    print("=" * 60)
    storage = GCSBackend(bucket_name=GCS_BUCKET_NAME, project_id=GCP_PROJECT_ID)
    scraper = MediumScraper(
        scrape_type='bulk',  #  CHANGED TO BULK
        storage=storage,
        log_dir='/tmp/medium_logs'
    )
    scraper.run()
    print(f"Medium Complete: {scraper.stats['success']} files")
    return scraper.stats


def print_summary(**kwargs):
    ti = kwargs['ti']
    gfg_stats = ti.xcom_pull(task_ids='scrape_gfg') or {}
    leetcode_stats = ti.xcom_pull(task_ids='scrape_leetcode') or {}
    medium_stats = ti.xcom_pull(task_ids='scrape_medium') or {}
    
    total = (
        gfg_stats.get('files_collected', 0) +
        leetcode_stats.get('files_collected', 0) +
        medium_stats.get('success', 0)
    )
    
    print("\n" + "=" * 60)
    print("BULK SCRAPING PIPELINE SUMMARY")
    print("=" * 60)
    print(f"GeeksforGeeks: {gfg_stats.get('files_collected', 0)} files")
    print(f"LeetCode:      {leetcode_stats.get('files_collected', 0)} files")
    print(f"Medium:        {medium_stats.get('success', 0)} files")
    print(f"TOTAL:         {total} files")
    print("=" * 60)
    
    return {
        'total_files': total,
        'gfg': gfg_stats,
        'leetcode': leetcode_stats,
        'medium': medium_stats
    }


default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'interview_scraping_pipeline',
    default_args=default_args,
    description='Scrape interview experiences - BULK MODE (all historical data)',
    schedule_interval=None,  # Manual trigger only for bulk
    catchup=False,
    tags=['scraping', 'bulk', 'production'],
)

start = BashOperator(
    task_id='start',
    bash_command='echo "Starting BULK scraping pipeline at $(date)"',
    dag=dag,
)

scrape_gfg = PythonOperator(
    task_id='scrape_gfg',
    python_callable=scrape_geeksforgeeks,
    execution_timeout=None,  #  INFINITE TIMEOUT
    dag=dag,
)

scrape_leetcode = PythonOperator(
    task_id='scrape_leetcode',
    python_callable=scrape_leetcode,
    execution_timeout=None,  #  INFINITE TIMEOUT
    dag=dag,
)

scrape_medium = PythonOperator(
    task_id='scrape_medium',
    python_callable=scrape_medium,
    execution_timeout=None,  #  INFINITE TIMEOUT
    dag=dag,
)

summary = PythonOperator(
    task_id='print_summary',
    python_callable=print_summary,
    execution_timeout=None,  #  INFINITE TIMEOUT
    dag=dag,
)

complete = BashOperator(
    task_id='complete',
    bash_command='echo "BULK Pipeline completed at $(date)"',
    dag=dag,
)

# Dependencies - all scrapers run in parallel
start >> [scrape_gfg, scrape_leetcode, scrape_medium] >> summary >> complete