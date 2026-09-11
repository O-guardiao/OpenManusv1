"""Intrinsic Ontology-as-Kernel runtime for OpenManus.

Exports are loaded lazily so ``python -m app.oak.kernel`` can run without
pre-importing the module it is about to execute.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "KernelValidationError",
    "OakRuntime",
    "execute_function",
    "freeze_kernel",
    "merge_evidence_graphs",
    "validate_kernel",
    "verify_frozen_kernel",
]


def __getattr__(name: str) -> Any:
    if name == "OakRuntime":
        return getattr(import_module("app.oak.runtime"), name)
    if name in {
        "KernelValidationError",
        "execute_function",
        "freeze_kernel",
        "merge_evidence_graphs",
        "validate_kernel",
        "verify_frozen_kernel",
    }:
        return getattr(import_module("app.oak.kernel"), name)
    raise AttributeError(name)
