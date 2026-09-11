from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from app.oak.kernel import (
    KernelValidationError,
    execute_function,
    merge_evidence_graphs,
    verify_frozen_kernel,
)
from app.oak.runtime import DEFAULT_KERNEL_ROOT


def test_intrinsic_kernel_is_frozen_and_changes_tool_classification() -> None:
    report = verify_frozen_kernel(DEFAULT_KERNEL_ROOT)

    assert report["status"] == "verified"
    assert report["assurance_level"] == "structural_only"

    rows = execute_function(
        DEFAULT_KERNEL_ROOT,
        "get_tool_profile",
        {"tool_name": "python_execute"},
    )

    assert rows == [
        {
            "id": "tool:python_execute",
            "type": "ToolCapability",
            "tool_name": "python_execute",
            "source": True,
            "sink": True,
            "trust_tier": "untrusted",
            "carrier_type": "tool_result",
            "requires_grant": True,
            "tainted_sink_requires_approval": True,
            "risk_score": 5,
            "patterns": ["executor", "tool_use", "controller"],
        }
    ]


def test_intrinsic_kernel_fails_closed_on_unknown_runtime_argument() -> None:
    with pytest.raises(KernelValidationError, match="undeclared arguments"):
        execute_function(
            DEFAULT_KERNEL_ROOT,
            "get_tool_profile",
            {"tool_name": "python_execute", "typo": True},
        )


def test_frozen_bundle_detects_graph_tampering() -> None:
    root = Path(__file__).parents[2] / "workspace" / ".test-oak-kernel"
    candidate = root / uuid4().hex
    try:
        shutil.copytree(DEFAULT_KERNEL_ROOT, candidate)
        graph_path = candidate / "graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["entities"][0]["properties"]["description"] = "tampered"
        graph_path.write_text(json.dumps(graph), encoding="utf-8")

        with pytest.raises(KernelValidationError, match="hash mismatch"):
            verify_frozen_kernel(candidate)
    finally:
        shutil.rmtree(candidate, ignore_errors=True)
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()


def test_composed_functions_exercise_graph_filters_traversal_and_aggregation() -> None:
    patterns = execute_function(
        DEFAULT_KERNEL_ROOT,
        "patterns_for_tool",
        {"tool_name": "python_execute"},
    )
    high_risk = execute_function(
        DEFAULT_KERNEL_ROOT,
        "capabilities_at_or_above_risk",
        {"minimum_risk": 5},
    )
    count = execute_function(DEFAULT_KERNEL_ROOT, "count_capabilities", {})

    assert {row["pattern_id"] for row in patterns} == {
        "executor",
        "tool_use",
        "controller",
    }
    assert "python_execute" in {row["tool_name"] for row in high_risk}
    assert count["operation"] == "count"
    assert count["value"] >= 8


def test_remaining_closed_operators_are_executable_or_explicitly_model_mediated() -> None:
    overlap = execute_function(
        DEFAULT_KERNEL_ROOT,
        "capabilities_using_patterns",
        {"patterns": ["controller"]},
    )
    connected = execute_function(
        DEFAULT_KERNEL_ROOT,
        "patterns_connected_to_mechanism",
        {"mechanism_entity_id": "mechanism:system_patterns"},
    )

    assert "python_execute" in {row["tool_name"] for row in overlap}
    assert "integrator" in {row["pattern_id"] for row in connected}
    with pytest.raises(KernelValidationError, match="model-mediated"):
        execute_function(
            DEFAULT_KERNEL_ROOT,
            "bind_runtime_slots",
            {"request": "natural-language value"},
        )


def test_chunk_graph_merge_uses_primary_identity_and_unites_evidence() -> None:
    schema = json.loads(
        (DEFAULT_KERNEL_ROOT / "schema.json").read_text(encoding="utf-8")
    )
    graph = json.loads(
        (DEFAULT_KERNEL_ROOT / "graph.json").read_text(encoding="utf-8")
    )
    tool = deepcopy(graph["entities"][0])
    alias = deepcopy(tool)
    alias["id"] = "z-alias:python_execute"
    alias["evidence"] = [{"source": "chunk-2", "locator": "entity:1"}]
    pattern = next(
        deepcopy(entity)
        for entity in graph["entities"]
        if entity["id"] == "pattern:executor"
    )
    relation = next(
        deepcopy(item)
        for item in graph["relations"]
        if item["id"] == "rel:python-executor"
    )
    alias_relation = deepcopy(relation)
    alias_relation["id"] = "z-rel:python-executor"
    alias_relation["source"] = alias["id"]
    alias_relation["evidence"] = [
        {"source": "chunk-2", "locator": "relation:1"}
    ]
    first = {
        "schema_version": 1,
        "entities": [tool, pattern],
        "relations": [relation],
    }
    second = {
        "schema_version": 1,
        "entities": [alias, pattern],
        "relations": [alias_relation],
    }

    merged = merge_evidence_graphs(schema, [second, first])
    merged_reversed = merge_evidence_graphs(schema, [first, second])

    python_entities = [
        entity
        for entity in merged["entities"]
        if entity["properties"].get("tool_name") == "python_execute"
    ]
    assert merged == merged_reversed
    assert len(python_entities) == 1
    assert len(python_entities[0]["evidence"]) == 2
    assert merged["relations"] == [
        {
            **relation,
            "evidence": [
                {"locator": "python_execute/executor", "source": "kernel mapping"},
                {"locator": "relation:1", "source": "chunk-2"},
            ],
        }
    ]


def test_chunk_graph_merge_fails_closed_on_property_conflict() -> None:
    schema = json.loads(
        (DEFAULT_KERNEL_ROOT / "schema.json").read_text(encoding="utf-8")
    )
    graph = json.loads(
        (DEFAULT_KERNEL_ROOT / "graph.json").read_text(encoding="utf-8")
    )
    first = deepcopy(graph["entities"][0])
    conflict = deepcopy(first)
    conflict["id"] = "z-conflict:python_execute"
    conflict["properties"]["risk_score"] = 1

    with pytest.raises(KernelValidationError, match="conflicting entity property"):
        merge_evidence_graphs(
            schema,
            [
                {"schema_version": 1, "entities": [first], "relations": []},
                {"schema_version": 1, "entities": [conflict], "relations": []},
            ],
        )
