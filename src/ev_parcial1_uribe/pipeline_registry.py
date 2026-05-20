"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines(raise_errors=True)
    
    if "machine_learning" in pipelines:
        pipelines["ml"] = pipelines["machine_learning"]
        
    if "ml_regression" in pipelines:
        pipelines["ml_regression"] = pipelines["ml_regression"]
        
    pipelines["__default__"] = sum(p for k, p in pipelines.items() if k not in ["ml", "ml_regression"])
    return pipelines
