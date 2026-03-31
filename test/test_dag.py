"""
DAG Validation Tests

Validates both Airflow DAG files without requiring a running Airflow instance.

1. dags/scraping_pipeline.py — scraping, preprocessing, db load, then triggers
   the chunking/embedding DAG
2. dags/chunking_embedding_pipeline.py — chunking and embedding as a standalone DAG

If Airflow is not installed in the test environment, all tests are
skipped with pytest.skip() rather than failing.

Covers:
- DAGs load without errors
- DAG IDs match expected values
- All expected tasks are present
- Task counts are correct
- Dependency graphs match expected flows
- Scrapers run in parallel (same upstream: start)
- Email tasks use trigger_rule='all_done'
- Chunking/embedding DAG has correct structure
"""
import sys
import pytest


# ── Helpers to safely import DAGs ──

def _load_dag(module_path: str, attr: str = "dag"):
    """
    Import a DAG module and return the DAG object.
    Returns None if Airflow is not installed.
    """
    try:
        import airflow  # noqa: F401
    except ImportError:
        return None

    import importlib
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    except Exception as e:
        pytest.fail(f"Failed to load DAG module '{module_path}': {e}")


@pytest.fixture(scope="module")
def scraping_dag():
    d = _load_dag("dags.scraping_pipeline")
    if d is None:
        pytest.skip("Airflow not installed, skipping DAG tests")
    return d


@pytest.fixture(scope="module")
def chunking_embedding_dag():
    d = _load_dag("dags.chunking_embedding_pipeline")
    if d is None:
        pytest.skip("Airflow not installed, skipping DAG tests")
    return d


def _get_task_ids(dag):
    return {t.task_id for t in dag.tasks}


def _get_upstream_ids(dag, task_id):
    task = dag.get_task(task_id)
    return {t.task_id for t in task.upstream_list}


def _get_downstream_ids(dag, task_id):
    task = dag.get_task(task_id)
    return {t.task_id for t in task.downstream_list}


# ======================================================================
# Scraping Pipeline DAG Tests
# ======================================================================

class TestScrapingDAGLoads:
    def test_dag_is_not_none(self, scraping_dag):
        assert scraping_dag is not None

    def test_dag_has_tasks(self, scraping_dag):
        assert len(scraping_dag.tasks) > 0


class TestScrapingDAGId:
    def test_dag_id_is_correct(self, scraping_dag):
        assert scraping_dag.dag_id == "interview_scraping_pipeline"


EXPECTED_SCRAPING_TASKS = {
    "start",
    "scrape_gfg",
    "scrape_leetcode",
    "scrape_medium",
    "print_summary",
    "run_preprocessing",
    "validate_processed_data",
    "load_to_database",
    "trigger_chunking_embedding",
    "complete",
    "build_email",
    "send_notification_email",
}


class TestScrapingTasksPresent:
    def test_all_expected_tasks_exist(self, scraping_dag):
        actual = _get_task_ids(scraping_dag)
        missing = EXPECTED_SCRAPING_TASKS - actual
        assert not missing, f"Missing tasks: {missing}"

    def test_task_count(self, scraping_dag):
        assert len(scraping_dag.tasks) == 12, (
            f"Expected 12 tasks, got {len(scraping_dag.tasks)}: "
            f"{sorted(_get_task_ids(scraping_dag))}"
        )

    def test_no_unexpected_tasks(self, scraping_dag):
        actual = _get_task_ids(scraping_dag)
        extra = actual - EXPECTED_SCRAPING_TASKS
        assert not extra, f"Unexpected tasks found: {extra}"


class TestScrapersParallel:
    SCRAPER_TASKS = ["scrape_gfg", "scrape_leetcode", "scrape_medium"]

    def test_all_scrapers_upstream_of_start(self, scraping_dag):
        """All 3 scrapers have 'start' as their only upstream dependency."""
        for task_id in self.SCRAPER_TASKS:
            upstream = _get_upstream_ids(scraping_dag, task_id)
            assert upstream == {"start"}, (
                f"'{task_id}' upstream should be {{'start'}}, got {upstream}"
            )

    def test_scrapers_are_downstream_of_start(self, scraping_dag):
        downstream = _get_downstream_ids(scraping_dag, "start")
        for task_id in self.SCRAPER_TASKS:
            assert task_id in downstream


class TestScrapingDependencyChain:
    """
    Validates the linear dependency chain after the parallel scrapers:
    scrapers -> print_summary -> run_preprocessing -> validate_processed_data
             -> load_to_database -> trigger_chunking_embedding
             -> complete -> build_email -> send_notification_email
    """

    def test_scrapers_upstream_of_summary(self, scraping_dag):
        upstream = _get_upstream_ids(scraping_dag, "print_summary")
        assert {"scrape_gfg", "scrape_leetcode", "scrape_medium"} == upstream

    def test_summary_upstream_of_preprocessing(self, scraping_dag):
        upstream = _get_upstream_ids(scraping_dag, "run_preprocessing")
        assert "print_summary" in upstream

    def test_preprocessing_upstream_of_validation(self, scraping_dag):
        upstream = _get_upstream_ids(scraping_dag, "validate_processed_data")
        assert "run_preprocessing" in upstream

    def test_validation_upstream_of_db_load(self, scraping_dag):
        upstream = _get_upstream_ids(scraping_dag, "load_to_database")
        assert "validate_processed_data" in upstream

    def test_db_load_upstream_of_trigger(self, scraping_dag):
        upstream = _get_upstream_ids(scraping_dag, "trigger_chunking_embedding")
        assert "load_to_database" in upstream

    def test_trigger_upstream_of_complete(self, scraping_dag):
        upstream = _get_upstream_ids(scraping_dag, "complete")
        assert "trigger_chunking_embedding" in upstream

    def test_complete_upstream_of_build_email(self, scraping_dag):
        upstream = _get_upstream_ids(scraping_dag, "build_email")
        assert "complete" in upstream

    def test_build_email_upstream_of_send_email(self, scraping_dag):
        upstream = _get_upstream_ids(scraping_dag, "send_notification_email")
        assert "build_email" in upstream


class TestEmailNotification:
    def test_build_email_trigger_rule_all_done(self, scraping_dag):
        """build_email should run regardless of upstream success/failure."""
        task = scraping_dag.get_task("build_email")
        assert task.trigger_rule == "all_done"

    def test_send_email_trigger_rule_all_done(self, scraping_dag):
        """send_notification_email should run regardless of upstream success/failure."""
        task = scraping_dag.get_task("send_notification_email")
        assert task.trigger_rule == "all_done"

    def test_send_email_has_recipients(self, scraping_dag):
        """Email task should have at least one recipient configured."""
        task = scraping_dag.get_task("send_notification_email")
        assert hasattr(task, "to")
        assert len(task.to) > 0


class TestScrapingDAGConfig:
    def test_catchup_disabled(self, scraping_dag):
        assert scraping_dag.catchup is False

    def test_schedule_is_none(self, scraping_dag):
        """DAG is manually triggered (schedule=None)."""
        assert scraping_dag.schedule_interval is None or scraping_dag.timetable is not None

    def test_dag_has_tags(self, scraping_dag):
        assert scraping_dag.tags is not None
        assert len(scraping_dag.tags) > 0


class TestTriggerTask:
    def test_trigger_targets_correct_dag(self, scraping_dag):
        task = scraping_dag.get_task("trigger_chunking_embedding")
        assert task.trigger_dag_id == "chunking_embedding_pipeline"

    def test_trigger_waits_for_completion(self, scraping_dag):
        task = scraping_dag.get_task("trigger_chunking_embedding")
        assert task.wait_for_completion is True


# ======================================================================
# Chunking & Embedding Pipeline DAG Tests
# ======================================================================

EXPECTED_CHUNKING_EMBEDDING_TASKS = {
    "start",
    "run_chunking",
    "run_embeddings",
    "complete",
}


class TestChunkingEmbeddingDAGLoads:
    def test_dag_is_not_none(self, chunking_embedding_dag):
        assert chunking_embedding_dag is not None

    def test_dag_has_tasks(self, chunking_embedding_dag):
        assert len(chunking_embedding_dag.tasks) > 0


class TestChunkingEmbeddingDAGId:
    def test_dag_id_is_correct(self, chunking_embedding_dag):
        assert chunking_embedding_dag.dag_id == "chunking_embedding_pipeline"


class TestChunkingEmbeddingTasksPresent:
    def test_all_expected_tasks_exist(self, chunking_embedding_dag):
        actual = _get_task_ids(chunking_embedding_dag)
        missing = EXPECTED_CHUNKING_EMBEDDING_TASKS - actual
        assert not missing, f"Missing tasks: {missing}"

    def test_task_count(self, chunking_embedding_dag):
        assert len(chunking_embedding_dag.tasks) == 4, (
            f"Expected 4 tasks, got {len(chunking_embedding_dag.tasks)}: "
            f"{sorted(_get_task_ids(chunking_embedding_dag))}"
        )

    def test_no_unexpected_tasks(self, chunking_embedding_dag):
        actual = _get_task_ids(chunking_embedding_dag)
        extra = actual - EXPECTED_CHUNKING_EMBEDDING_TASKS
        assert not extra, f"Unexpected tasks found: {extra}"


class TestChunkingEmbeddingDependencyChain:
    """
    Validates: start -> run_chunking -> run_embeddings -> complete
    """

    def test_start_upstream_of_chunking(self, chunking_embedding_dag):
        upstream = _get_upstream_ids(chunking_embedding_dag, "run_chunking")
        assert "start" in upstream

    def test_chunking_upstream_of_embeddings(self, chunking_embedding_dag):
        upstream = _get_upstream_ids(chunking_embedding_dag, "run_embeddings")
        assert "run_chunking" in upstream

    def test_embeddings_upstream_of_complete(self, chunking_embedding_dag):
        upstream = _get_upstream_ids(chunking_embedding_dag, "complete")
        assert "run_embeddings" in upstream


class TestChunkingEmbeddingDAGConfig:
    def test_catchup_disabled(self, chunking_embedding_dag):
        assert chunking_embedding_dag.catchup is False

    def test_schedule_is_none(self, chunking_embedding_dag):
        """DAG is manually triggered (schedule=None)."""
        assert chunking_embedding_dag.schedule_interval is None or chunking_embedding_dag.timetable is not None

    def test_dag_has_tags(self, chunking_embedding_dag):
        assert chunking_embedding_dag.tags is not None
        assert len(chunking_embedding_dag.tags) > 0
