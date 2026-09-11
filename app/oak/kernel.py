#!/usr/bin/env python3
"""Deterministic validator, freezer, and composed-function runner.

This vendored module is the distributable OpenManus adaptation of the global
OaK runner.  It implements the nine public operators described in
arXiv:2608.22974v1.  It is an independent implementation, not the authors'
code, and a successful lock verification provides structural assurance only.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


KERNEL_FILES = ("schema.json", "functions.json")
EVIDENCE_FILES = ("graph.json",)
BUNDLE_FILES = KERNEL_FILES + EVIDENCE_FILES
LOCK_SCHEMA_VERSION = 2
OPERATORS = {
    "extract_runtime_slots",
    "lookup_entities",
    "traverse_relations",
    "project_properties",
    "filter_categorical",
    "filter_relation_connected",
    "filter_numeric",
    "filter_set_overlap",
    "aggregate_values",
}
PROPERTY_TYPES = {
    "string",
    "integer",
    "number",
    "boolean",
    "string_list",
    "number_list",
    "json",
}


class KernelValidationError(ValueError):
    """Raised when a kernel, lock, function, or runtime value is invalid."""


def _fail(message: str) -> None:
    raise KernelValidationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing required file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path.name}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path.name} must contain a JSON object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty string")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a list")
    return value


def _version(document: dict[str, Any], label: str) -> None:
    if document.get("schema_version") != 1:
        _fail(f"{label}.schema_version must be 1")


def _validate_evidence(value: Any, label: str) -> None:
    evidence = _list(value, label)
    if not evidence:
        _fail(f"{label} must contain at least one evidence pointer")
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            _fail(f"{label}[{index}] must be an object")
        _text(item.get("source"), f"{label}[{index}].source")
        _text(item.get("locator"), f"{label}[{index}].locator")


def _matches_type(value: Any, property_type: str) -> bool:
    if property_type == "json":
        return True
    if property_type == "string":
        return isinstance(value, str)
    if property_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if property_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if property_type == "boolean":
        return isinstance(value, bool)
    if property_type == "string_list":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if property_type == "number_list":
        return isinstance(value, list) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
    return False


def _validate_schema(
    schema: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _version(schema, "schema")
    task = schema.get("task")
    if not isinstance(task, dict):
        _fail("schema.task must be an object")
    _text(task.get("name"), "schema.task.name")
    _text(task.get("description"), "schema.task.description")

    entity_types: dict[str, dict[str, Any]] = {}
    for index, entity_type in enumerate(
        _list(schema.get("entity_types"), "schema.entity_types")
    ):
        if not isinstance(entity_type, dict):
            _fail(f"schema.entity_types[{index}] must be an object")
        name = _text(entity_type.get("name"), f"schema.entity_types[{index}].name")
        if name in entity_types:
            _fail(f"duplicate entity type: {name}")
        primary_key = _text(
            entity_type.get("primary_key"),
            f"schema.entity_types[{index}].primary_key",
        )
        properties = entity_type.get("properties")
        if not isinstance(properties, dict) or not properties:
            _fail(f"entity type {name} must declare properties")
        for property_name, property_type in properties.items():
            _text(property_name, f"entity type {name} property name")
            if property_type not in PROPERTY_TYPES:
                _fail(
                    f"entity type {name} property {property_name} has unknown type: "
                    f"{property_type}"
                )
        if primary_key not in properties:
            _fail(
                f"entity type {name} primary key is not a declared property: "
                f"{primary_key}"
            )
        entity_types[name] = entity_type
    if not entity_types:
        _fail("schema.entity_types must not be empty")

    relations: dict[str, dict[str, Any]] = {}
    for index, relation in enumerate(
        _list(schema.get("relations"), "schema.relations")
    ):
        if not isinstance(relation, dict):
            _fail(f"schema.relations[{index}] must be an object")
        name = _text(relation.get("name"), f"schema.relations[{index}].name")
        if name in relations:
            _fail(f"duplicate relation type: {name}")
        source = _text(relation.get("source"), f"relation {name}.source")
        target = _text(relation.get("target"), f"relation {name}.target")
        if source not in entity_types or target not in entity_types:
            _fail(f"relation {name} references an unknown entity type")
        relations[name] = relation
    return entity_types, relations


def _value_references(value: Any) -> list[str]:
    if isinstance(value, dict):
        if set(value) == {"value_from"}:
            return [_text(value["value_from"], "pipeline value_from")]
        references: list[str] = []
        for item in value.values():
            references.extend(_value_references(item))
        return references
    if isinstance(value, list):
        references = []
        for item in value:
            references.extend(_value_references(item))
        return references
    return []


def _validate_functions(
    functions: dict[str, Any], entity_types: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    _version(functions, "functions")
    catalog: dict[str, dict[str, Any]] = {}
    for index, function in enumerate(
        _list(functions.get("functions"), "functions.functions")
    ):
        if not isinstance(function, dict):
            _fail(f"functions.functions[{index}] must be an object")
        name = _text(function.get("name"), f"functions.functions[{index}].name")
        if name in catalog:
            _fail(f"duplicate function: {name}")
        _text(function.get("description"), f"function {name}.description")
        inputs = function.get("inputs")
        if not isinstance(inputs, dict):
            _fail(f"function {name}.inputs must be an object")
        for input_name, input_spec in inputs.items():
            _text(input_name, f"function {name} input name")
            if not isinstance(input_spec, dict):
                _fail(f"function {name} input {input_name} must be an object")
            input_type = _text(
                input_spec.get("type"), f"function {name} input {input_name}.type"
            )
            if input_type not in PROPERTY_TYPES and input_type not in entity_types:
                _fail(f"function {name} input {input_name} has unknown type: {input_type}")
            if "required" in input_spec and not isinstance(input_spec["required"], bool):
                _fail(f"function {name} input {input_name}.required must be boolean")
        output = function.get("output")
        if not isinstance(output, dict):
            _fail(f"function {name}.output must be an object")
        output_type = _text(output.get("type"), f"function {name}.output.type")
        if output_type == "entity_list":
            output_entity = _text(
                output.get("entity_type"), f"function {name}.output.entity_type"
            )
            if output_entity not in entity_types:
                _fail(
                    f"function {name} output references unknown entity type: "
                    f"{output_entity}"
                )
        elif output_type not in {"rows", "aggregate", "json"}:
            _fail(f"function {name} has unknown output type: {output_type}")
        pipeline = _list(function.get("pipeline"), f"function {name}.pipeline")
        if not pipeline:
            _fail(f"function {name}.pipeline must not be empty")
        for step_index, step in enumerate(pipeline):
            if not isinstance(step, dict):
                _fail(f"function {name}.pipeline[{step_index}] must be an object")
            operator = _text(
                step.get("operator"), f"function {name}.pipeline[{step_index}].operator"
            )
            if operator not in OPERATORS:
                _fail(f"function {name} uses unknown operator: {operator}")
            if not isinstance(step.get("args", {}), dict):
                _fail(f"function {name}.pipeline[{step_index}].args must be an object")
            for value_from in _value_references(step.get("args", {})):
                if value_from not in inputs:
                    _fail(
                        f"function {name}.pipeline[{step_index}] references "
                        f"undeclared input: {value_from}"
                    )
        catalog[name] = function
    if not catalog:
        _fail("functions.functions must not be empty")
    return catalog


def _validate_graph(
    graph: dict[str, Any],
    entity_types: dict[str, dict[str, Any]],
    relation_types: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    _version(graph, "graph")
    entities: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str]] = set()
    for index, entity in enumerate(_list(graph.get("entities"), "graph.entities")):
        if not isinstance(entity, dict):
            _fail(f"graph.entities[{index}] must be an object")
        entity_id = _text(entity.get("id"), f"graph.entities[{index}].id")
        if entity_id in entities:
            _fail(f"duplicate entity id: {entity_id}")
        entity_type = _text(entity.get("type"), f"entity {entity_id}.type")
        if entity_type not in entity_types:
            _fail(f"entity {entity_id} has unknown type: {entity_type}")
        properties = entity.get("properties")
        if not isinstance(properties, dict):
            _fail(f"entity {entity_id}.properties must be an object")
        declared = entity_types[entity_type]["properties"]
        unknown = sorted(set(properties) - set(declared))
        if unknown:
            _fail(f"entity {entity_id} has undeclared properties: {', '.join(unknown)}")
        primary_key = entity_types[entity_type]["primary_key"]
        if primary_key not in properties:
            _fail(f"entity {entity_id} is missing primary key property: {primary_key}")
        for property_name, property_value in properties.items():
            if not _matches_type(property_value, declared[property_name]):
                _fail(
                    f"entity {entity_id} property {property_name} does not match "
                    f"type {declared[property_name]}"
                )
        identity = (
            entity_type,
            json.dumps(properties[primary_key], sort_keys=True),
        )
        if identity in identities:
            _fail(
                f"duplicate identity for type {entity_type} and primary key "
                f"{properties[primary_key]!r}"
            )
        identities.add(identity)
        _validate_evidence(entity.get("evidence"), f"entity {entity_id}.evidence")
        entities[entity_id] = entity
    if not entities:
        _fail("graph.entities must not be empty")

    relation_ids: set[str] = set()
    for index, relation in enumerate(_list(graph.get("relations"), "graph.relations")):
        if not isinstance(relation, dict):
            _fail(f"graph.relations[{index}] must be an object")
        relation_id = _text(relation.get("id"), f"graph.relations[{index}].id")
        if relation_id in relation_ids:
            _fail(f"duplicate relation id: {relation_id}")
        relation_ids.add(relation_id)
        relation_type = _text(relation.get("type"), f"relation {relation_id}.type")
        if relation_type not in relation_types:
            _fail(f"relation {relation_id} has unknown type: {relation_type}")
        source = _text(relation.get("source"), f"relation {relation_id}.source")
        target = _text(relation.get("target"), f"relation {relation_id}.target")
        if source not in entities or target not in entities:
            _fail(f"relation {relation_id} references a missing endpoint")
        spec = relation_types[relation_type]
        if entities[source]["type"] != spec["source"]:
            _fail(f"relation {relation_id} source violates its declared type")
        if entities[target]["type"] != spec["target"]:
            _fail(f"relation {relation_id} target violates its declared type")
        _validate_evidence(relation.get("evidence"), f"relation {relation_id}.evidence")
    return entities


def _identity_key(
    entity: Mapping[str, Any],
    entity_types: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    entity_type = str(entity["type"])
    primary_key = str(entity_types[entity_type]["primary_key"])
    primary_value = entity["properties"][primary_key]
    return (
        entity_type,
        json.dumps(
            primary_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _merge_evidence(*collections: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_value: dict[str, dict[str, Any]] = {}
    for collection in collections:
        for pointer in collection:
            copied = deepcopy(dict(pointer))
            key = json.dumps(
                copied,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            by_value[key] = copied
    return [by_value[key] for key in sorted(by_value)]


def merge_evidence_graphs(
    schema: Mapping[str, Any],
    fragments: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge chunk graphs by κ(entity)=(type, primary-key value).

    The merge is deterministic across fragment order. Complementary properties
    and evidence pointers are united; conflicting values, ambiguous raw IDs,
    invalid endpoints, and relation-ID collisions fail closed.
    """

    schema_document = deepcopy(dict(schema))
    entity_types, relation_types = _validate_schema(schema_document)
    fragment_list = [deepcopy(dict(fragment)) for fragment in fragments]
    if not fragment_list:
        _fail("at least one evidence graph fragment is required")

    raw_entities: list[dict[str, Any]] = []
    raw_relations: list[dict[str, Any]] = []
    for index, fragment in enumerate(fragment_list):
        _version(fragment, f"fragment[{index}]")
        raw_entities.extend(
            deepcopy(_list(fragment.get("entities"), f"fragment[{index}].entities"))
        )
        raw_relations.extend(
            deepcopy(
                _list(fragment.get("relations"), f"fragment[{index}].relations")
            )
        )

    validated_entities: list[tuple[tuple[str, str], dict[str, Any]]] = []
    for index, entity in enumerate(raw_entities):
        if not isinstance(entity, dict):
            _fail(f"fragment entity[{index}] must be an object")
        _validate_graph(
            {"schema_version": 1, "entities": [entity], "relations": []},
            entity_types,
            relation_types,
        )
        validated_entities.append((_identity_key(entity, entity_types), entity))
    validated_entities.sort(key=lambda item: (item[0], str(item[1]["id"])))

    entities_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    raw_identities: dict[str, tuple[str, str]] = {}
    alias_to_canonical: dict[str, str] = {}
    aliases_by_identity: dict[tuple[str, str], set[str]] = {}
    for identity, entity in validated_entities:
        entity_id = str(entity["id"])
        previous_identity = raw_identities.get(entity_id)
        if previous_identity is not None and previous_identity != identity:
            _fail(f"ambiguous entity id across fragments: {entity_id}")
        raw_identities[entity_id] = identity
        aliases_by_identity.setdefault(identity, set()).add(entity_id)

        existing = entities_by_identity.get(identity)
        if existing is None:
            entities_by_identity[identity] = deepcopy(entity)
            continue
        if existing["type"] != entity["type"]:
            _fail(f"entity identity changed type during merge: {identity[0]}")
        for property_name, value in entity["properties"].items():
            if (
                property_name in existing["properties"]
                and existing["properties"][property_name] != value
            ):
                _fail(
                    "conflicting entity property for identity "
                    f"{identity[0]}:{identity[1]}:{property_name}"
                )
            existing["properties"][property_name] = deepcopy(value)
        existing["evidence"] = _merge_evidence(
            existing["evidence"],
            entity["evidence"],
        )

    for identity, aliases in aliases_by_identity.items():
        canonical_id = str(entities_by_identity[identity]["id"])
        for alias in aliases:
            alias_to_canonical[alias] = canonical_id

    relations_by_semantics: dict[tuple[str, str, str], dict[str, Any]] = {}
    relation_id_semantics: dict[str, tuple[str, str, str]] = {}
    canonical_relations: list[tuple[tuple[str, str, str], dict[str, Any]]] = []
    for index, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
            _fail(f"fragment relation[{index}] must be an object")
        relation_id = _text(relation.get("id"), f"fragment relation[{index}].id")
        relation_type = _text(
            relation.get("type"), f"fragment relation[{index}].type"
        )
        source = _text(
            relation.get("source"), f"fragment relation[{index}].source"
        )
        target = _text(
            relation.get("target"), f"fragment relation[{index}].target"
        )
        _validate_evidence(
            relation.get("evidence"), f"fragment relation[{index}].evidence"
        )
        if source not in alias_to_canonical or target not in alias_to_canonical:
            _fail(f"relation {relation_id} references an unmapped endpoint")
        canonical = deepcopy(relation)
        canonical["source"] = alias_to_canonical[source]
        canonical["target"] = alias_to_canonical[target]
        semantics = (
            relation_type,
            canonical["source"],
            canonical["target"],
        )
        previous_semantics = relation_id_semantics.get(relation_id)
        if previous_semantics is not None and previous_semantics != semantics:
            _fail(f"relation id collision across fragments: {relation_id}")
        relation_id_semantics[relation_id] = semantics
        canonical_relations.append((semantics, canonical))

    canonical_relations.sort(key=lambda item: (item[0], str(item[1]["id"])))
    for semantics, relation in canonical_relations:
        existing = relations_by_semantics.get(semantics)
        if existing is None:
            relations_by_semantics[semantics] = relation
        else:
            existing["evidence"] = _merge_evidence(
                existing["evidence"],
                relation["evidence"],
            )

    merged = {
        "schema_version": 1,
        "entities": sorted(
            entities_by_identity.values(),
            key=lambda entity: str(entity["id"]),
        ),
        "relations": sorted(
            relations_by_semantics.values(),
            key=lambda relation: str(relation["id"]),
        ),
    }
    _validate_graph(merged, entity_types, relation_types)
    return merged


def validate_kernel(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    schema = _read_json(root / "schema.json")
    functions = _read_json(root / "functions.json")
    graph = _read_json(root / "graph.json")
    entity_types, relation_types = _validate_schema(schema)
    catalog = _validate_functions(functions, entity_types)
    entities = _validate_graph(graph, entity_types, relation_types)
    return {
        "status": "valid",
        "root": str(root),
        "entity_types": len(entity_types),
        "relation_types": len(relation_types),
        "functions": len(catalog),
        "entities": len(entities),
        "relations": len(graph["relations"]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(file_hashes: dict[str, str], names: tuple[str, ...]) -> str:
    joined = "\n".join(f"{name}:{file_hashes[name]}" for name in sorted(names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def freeze_kernel(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    lock_path = root / "kernel.lock.json"
    if lock_path.exists():
        _fail("kernel.lock.json already exists; create a new versioned directory")
    report = validate_kernel(root)
    file_hashes = {name: _sha256(root / name) for name in BUNDLE_FILES}
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "frozen",
        "assurance_level": "structural_only",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "kernel_sha256": _manifest_hash(file_hashes, KERNEL_FILES),
        "evidence_sha256": _manifest_hash(file_hashes, EVIDENCE_FILES),
        "bundle_sha256": _manifest_hash(file_hashes, BUNDLE_FILES),
        "kernel_files": {name: file_hashes[name] for name in KERNEL_FILES},
        "evidence_files": {name: file_hashes[name] for name in EVIDENCE_FILES},
        "files": file_hashes,
        "counts": {key: value for key, value in report.items() if isinstance(value, int)},
    }
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock


def verify_frozen_kernel(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    report = validate_kernel(root)
    lock = _read_json(root / "kernel.lock.json")
    lock_version = lock.get("schema_version")
    if lock_version not in {1, LOCK_SCHEMA_VERSION}:
        _fail(f"kernel.lock.schema_version must be 1 or {LOCK_SCHEMA_VERSION}")
    if lock.get("status") != "frozen" or not isinstance(lock.get("files"), dict):
        _fail("kernel.lock.json is not a frozen kernel manifest")
    if set(lock["files"]) != set(BUNDLE_FILES):
        _fail("kernel.lock.json files must match the complete bundle")
    for name in BUNDLE_FILES:
        if lock["files"].get(name) != _sha256(root / name):
            _fail(f"hash mismatch for frozen file: {name}")
    counts = {key: value for key, value in report.items() if isinstance(value, int)}
    if lock.get("counts") != counts:
        _fail("kernel.lock.json counts do not match the frozen bundle")

    actual_kernel = _manifest_hash(lock["files"], KERNEL_FILES)
    actual_evidence = _manifest_hash(lock["files"], EVIDENCE_FILES)
    actual_bundle = _manifest_hash(lock["files"], BUNDLE_FILES)
    if lock_version == 1:
        if lock.get("kernel_sha256") != actual_bundle:
            _fail("legacy kernel hash mismatch")
        assurance = "legacy_structural_only"
    else:
        expected_kernel_files = {name: lock["files"][name] for name in KERNEL_FILES}
        expected_evidence_files = {
            name: lock["files"][name] for name in EVIDENCE_FILES
        }
        if lock.get("kernel_files") != expected_kernel_files:
            _fail("kernel.lock.json kernel_files do not match files")
        if lock.get("evidence_files") != expected_evidence_files:
            _fail("kernel.lock.json evidence_files do not match files")
        if lock.get("kernel_sha256") != actual_kernel:
            _fail("kernel hash mismatch")
        if lock.get("evidence_sha256") != actual_evidence:
            _fail("evidence hash mismatch")
        if lock.get("bundle_sha256") != actual_bundle:
            _fail("bundle hash mismatch")
        if lock.get("assurance_level") != "structural_only":
            _fail("unknown frozen assurance level")
        assurance = "structural_only"
    return {
        "status": "verified",
        "root": str(root),
        "lock_schema_version": lock_version,
        "assurance_level": assurance,
        "kernel_sha256": actual_kernel,
        "evidence_sha256": actual_evidence,
        "bundle_sha256": actual_bundle,
    }


def _resolve(value: Any, runtime_args: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"value_from"}:
            name = value["value_from"]
            if name not in runtime_args:
                _fail(f"missing runtime argument: {name}")
            return runtime_args[name]
        return {key: _resolve(item, runtime_args) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, runtime_args) for item in value]
    return value


def _property(item: dict[str, Any], name: str) -> Any:
    return item.get("properties", item).get(name)


def _lookup(graph: dict[str, Any], args: dict[str, Any]) -> list[dict[str, Any]]:
    results = list(graph["entities"])
    if args.get("entity_types"):
        allowed = set(args["entity_types"])
        results = [item for item in results if item["type"] in allowed]
    if args.get("entity_ids"):
        allowed_ids = set(args["entity_ids"])
        results = [item for item in results if item["id"] in allowed_ids]
    for key, expected in args.get("property_filters", {}).items():
        results = [item for item in results if _property(item, key) == expected]
    query = args.get("name_query") or args.get("text_query")
    if query:
        needle = str(query).casefold()
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in results:
            haystack = " ".join(
                str(value) for value in item["properties"].values()
            ).casefold()
            score = (
                1.0
                if needle in haystack
                else SequenceMatcher(None, needle, haystack).ratio()
            )
            if score >= float(args.get("min_score", 0.0)):
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        results = [item for _, item in scored]
    else:
        results.sort(key=lambda item: item["id"])
    top_k = args.get("top_k")
    return results[: int(top_k)] if top_k is not None else results


def _traverse(
    graph: dict[str, Any],
    current: list[dict[str, Any]] | None,
    args: dict[str, Any],
) -> list[dict[str, Any]]:
    entity_by_id = {item["id"]: item for item in graph["entities"]}
    start_ids = args.get("start_entity_ids") or [
        item["id"] for item in (current or [])
    ]
    relation_types = set(args.get("relation_types") or [])
    direction = args.get("direction", "outgoing")
    if direction not in {"outgoing", "incoming", "both"}:
        _fail(f"invalid traversal direction: {direction}")
    visited = set(start_ids)
    output_ids = set(start_ids) if args.get("include_starting_entities", True) else set()
    queue: deque[tuple[str, int]] = deque((entity_id, 0) for entity_id in start_ids)
    max_hops = int(args.get("max_hops", 1))
    while queue:
        entity_id, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for relation in graph["relations"]:
            if relation_types and relation["type"] not in relation_types:
                continue
            neighbors: list[str] = []
            if direction in {"outgoing", "both"} and relation["source"] == entity_id:
                neighbors.append(relation["target"])
            if direction in {"incoming", "both"} and relation["target"] == entity_id:
                neighbors.append(relation["source"])
            for neighbor in neighbors:
                output_ids.add(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return [
        entity_by_id[entity_id]
        for entity_id in sorted(output_ids)
        if entity_id in entity_by_id
    ]


def _filter_categorical(
    current: list[dict[str, Any]], args: dict[str, Any]
) -> list[dict[str, Any]]:
    result = current
    for condition in args.get("conditions", []):
        name = condition["property"]
        include = condition.get("include")
        exclude = condition.get("exclude")
        if include is not None:
            allowed = {json.dumps(value, sort_keys=True) for value in include}
            result = [
                item
                for item in result
                if json.dumps(_property(item, name), sort_keys=True) in allowed
            ]
        if exclude is not None:
            blocked = {json.dumps(value, sort_keys=True) for value in exclude}
            result = [
                item
                for item in result
                if json.dumps(_property(item, name), sort_keys=True) not in blocked
            ]
    return result


def _filter_numeric(
    current: list[dict[str, Any]], args: dict[str, Any]
) -> list[dict[str, Any]]:
    comparators = {
        "eq": lambda left, right: left == right,
        "ne": lambda left, right: left != right,
        "gt": lambda left, right: left > right,
        "gte": lambda left, right: left >= right,
        "lt": lambda left, right: left < right,
        "lte": lambda left, right: left <= right,
    }
    result = current
    for condition in args.get("conditions", []):
        operator = condition.get("op", "eq")
        if operator not in comparators:
            _fail(f"unknown numeric comparator: {operator}")
        expected = condition["value"]
        property_name = condition["property"]
        result = [
            item
            for item in result
            if isinstance(_property(item, property_name), (int, float))
            and not isinstance(_property(item, property_name), bool)
            and comparators[operator](_property(item, property_name), expected)
        ]
    return result


def _filter_set_overlap(
    current: list[dict[str, Any]], args: dict[str, Any]
) -> list[dict[str, Any]]:
    result = current
    for condition in args.get("conditions", []):
        requested = set(condition.get("values", []))
        mode = condition.get("mode", "any")
        minimum = int(condition.get("minimum", 1))

        def keep(item: dict[str, Any]) -> bool:
            actual = set(_property(item, condition["property"]) or [])
            if mode == "all":
                return requested.issubset(actual)
            if mode == "minimum":
                return len(actual & requested) >= minimum
            if mode != "any":
                _fail(f"unknown set-overlap mode: {mode}")
            return bool(actual & requested)

        result = [item for item in result if keep(item)]
    return result


def _filter_connected(
    graph: dict[str, Any],
    current: list[dict[str, Any]],
    args: dict[str, Any],
) -> list[dict[str, Any]]:
    result = current
    for condition in args.get("conditions", []):
        relation_types = set(condition.get("relation_types") or [])
        anchors = set(condition.get("anchor_entity_ids") or [])
        direction = condition.get("direction", "outgoing")

        def connected(entity_id: str) -> bool:
            for relation in graph["relations"]:
                if relation_types and relation["type"] not in relation_types:
                    continue
                if (
                    direction in {"outgoing", "both"}
                    and relation["source"] == entity_id
                    and relation["target"] in anchors
                ):
                    return True
                if (
                    direction in {"incoming", "both"}
                    and relation["target"] == entity_id
                    and relation["source"] in anchors
                ):
                    return True
            return False

        result = [item for item in result if connected(item["id"])]
    return result


def _aggregate(current: list[dict[str, Any]], args: dict[str, Any]) -> dict[str, Any]:
    request = args.get("request", {})
    operation = request.get("operation", "count")
    field = request.get("field")
    if operation == "count":
        value: Any = len(current)
    else:
        values = [_property(item, field) for item in current]
        numeric = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not numeric:
            _fail(f"aggregation {operation} has no numeric values")
        if operation == "sum":
            value = sum(numeric)
        elif operation == "min":
            value = min(numeric)
        elif operation == "max":
            value = max(numeric)
        elif operation == "average":
            value = sum(numeric) / len(numeric)
        else:
            _fail(f"unknown aggregation operation: {operation}")
    return {"operation": operation, "field": field, "value": value}


def _validate_runtime_args(
    function_name: str,
    function: dict[str, Any],
    runtime_args: dict[str, Any],
    entity_types: dict[str, dict[str, Any]],
) -> None:
    declared = function["inputs"]
    unknown = sorted(set(runtime_args) - set(declared))
    if unknown:
        _fail(
            f"function {function_name} received undeclared arguments: "
            f"{', '.join(unknown)}"
        )
    for name, spec in declared.items():
        if spec.get("required", False) and name not in runtime_args:
            _fail(f"missing required function argument: {name}")
        if name not in runtime_args:
            continue
        input_type = spec["type"]
        value = runtime_args[name]
        if input_type in PROPERTY_TYPES:
            matches = _matches_type(value, input_type)
        else:
            matches = (
                input_type in entity_types
                and isinstance(value, dict)
                and value.get("type") == input_type
            )
        if not matches:
            _fail(
                f"function {function_name} argument {name} does not match "
                f"type {input_type}"
            )


def execute_function(
    root: str | Path,
    function_name: str,
    runtime_args: dict[str, Any],
    *,
    allow_draft: bool = False,
) -> Any:
    root = Path(root).resolve()
    if allow_draft:
        validate_kernel(root)
    else:
        verify_frozen_kernel(root)
    schema = _read_json(root / "schema.json")
    functions = _read_json(root / "functions.json")
    graph = _read_json(root / "graph.json")
    function = next(
        (item for item in functions["functions"] if item["name"] == function_name),
        None,
    )
    if function is None:
        _fail(f"unknown function: {function_name}")
    entity_types, _ = _validate_schema(schema)
    _validate_runtime_args(function_name, function, runtime_args, entity_types)
    current: Any = None
    for step in function["pipeline"]:
        operator = step["operator"]
        args = _resolve(step.get("args", {}), runtime_args)
        if operator == "extract_runtime_slots":
            _fail(
                "extract_runtime_slots is model-mediated; bind evidenced slots "
                "before deterministic execution"
            )
        if operator == "lookup_entities":
            current = _lookup(graph, args)
        elif operator == "traverse_relations":
            current = _traverse(graph, current, args)
        elif operator == "project_properties":
            if not isinstance(current, list):
                _fail("project_properties requires a list input")
            current = [
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    **{
                        name: _property(item, name)
                        for name in args.get("property_names", [])
                    },
                }
                for item in current
            ]
        elif operator == "filter_categorical":
            current = _filter_categorical(current or [], args)
        elif operator == "filter_relation_connected":
            current = _filter_connected(graph, current or [], args)
        elif operator == "filter_numeric":
            current = _filter_numeric(current or [], args)
        elif operator == "filter_set_overlap":
            current = _filter_set_overlap(current or [], args)
        elif operator == "aggregate_values":
            current = _aggregate(current or [], args)
        else:
            _fail(f"operator is validated but not implemented: {operator}")
    output_type = function["output"]["type"]
    if output_type == "entity_list":
        expected = function["output"]["entity_type"]
        if not isinstance(current, list) or any(
            not isinstance(item, dict) or item.get("type") != expected
            for item in current
        ):
            _fail(f"function {function_name} violated entity-list output type {expected}")
    elif output_type == "rows" and not isinstance(current, list):
        _fail(f"function {function_name} violated rows output type")
    elif output_type == "aggregate" and not isinstance(current, dict):
        _fail(f"function {function_name} violated aggregate output type")
    return current


def _parse_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("arguments must be a JSON object")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "freeze", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("kernel", type=Path)
    execute = subparsers.add_parser("execute")
    execute.add_argument("kernel", type=Path)
    execute.add_argument("function")
    execute.add_argument("--args", type=_parse_object, default={})
    execute.add_argument("--draft", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("schema", type=Path)
    merge.add_argument("fragments", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    namespace = parser.parse_args(argv)
    try:
        if namespace.command == "validate":
            result = validate_kernel(namespace.kernel)
        elif namespace.command == "freeze":
            result = freeze_kernel(namespace.kernel)
        elif namespace.command == "verify":
            result = verify_frozen_kernel(namespace.kernel)
        elif namespace.command == "merge":
            if namespace.output.exists():
                _fail("merge output already exists")
            merged_graph = merge_evidence_graphs(
                _read_json(namespace.schema),
                [_read_json(path) for path in namespace.fragments],
            )
            namespace.output.parent.mkdir(parents=True, exist_ok=True)
            serialized = (
                json.dumps(merged_graph, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            namespace.output.write_text(serialized, encoding="utf-8")
            result = {
                "status": "merged",
                "output": str(namespace.output.resolve()),
                "entities": len(merged_graph["entities"]),
                "relations": len(merged_graph["relations"]),
                "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            }
        else:
            result = execute_function(
                namespace.kernel,
                namespace.function,
                namespace.args,
                allow_draft=namespace.draft,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (KernelValidationError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
