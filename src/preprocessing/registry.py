"""
Step Registry

Decoupled from pipeline.py to avoid circular imports.
steps/__init__.py imports register_step from here,
and pipeline.py imports _STEP_REGISTRY from here.
"""

from typing import Dict

_STEP_REGISTRY: Dict[str, type] = {}


def register_step(name: str, step_class: type):
    """Register a preprocessing step class under a config name."""
    _STEP_REGISTRY[name] = step_class