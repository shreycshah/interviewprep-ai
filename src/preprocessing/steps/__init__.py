"""
Preprocessing Steps Package

Imports all step classes and registers them with the pipeline's
step registry. Adding a new step requires:
    1. Create the step class inheriting PreprocessingStep
    2. Import it here
    3. Register it with register_step()
    4. Add it to pipeline_config.yaml
"""

from src.preprocessing.pipeline import register_step

from src.preprocessing.steps.content_normalizer import ContentNormalizer
from src.preprocessing.steps.pii_remover import PIIRemover
from src.preprocessing.steps.quality_filter import QualityFilter
from src.preprocessing.steps.deduplicator import Deduplicator
from src.preprocessing.steps.entity_extractor import EntityExtractor
from src.preprocessing.steps.schema_validator import SchemaValidator

# ── Register all steps ──
# Keys must match the "name" field in pipeline_config.yaml

register_step("content_normalizer", ContentNormalizer)
register_step("pii_remover", PIIRemover)
register_step("quality_filter", QualityFilter)
register_step("deduplicator", Deduplicator)
register_step("entity_extractor", EntityExtractor)
register_step("schema_validator", SchemaValidator)