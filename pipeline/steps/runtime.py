"""Compatibility shim — runtime orchestrator has moved to pipeline/runtime.py.

Import from pipeline.runtime directly:
    from pipeline.runtime import run_runtime
"""
from pipeline.runtime import run_runtime  # noqa: F401
