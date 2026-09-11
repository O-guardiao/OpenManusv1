"""Versioned construction and repair workflow for the intrinsic OaK bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Literal

from app.oak.kernel import (
    BUNDLE_FILES,
    KernelValidationError,
    freeze_kernel,
    validate_kernel,
    verify_frozen_kernel,
)


RepairScope = Literal["schema", "function", "graph"]
RepairVerb = Literal["add", "delete", "modify"]


@dataclass(frozen=True)
class RepairAction:
    """Typed repair tuple σ=(scope, action, delta, rollback, evidence)."""

    scope: RepairScope
    action: RepairVerb
    target: str
    path: tuple[str, ...] = ()
    expected: Any = None
    value: Any = None
    collection: str | None = None
    rationale: str = ""
    evidence: tuple[str, ...] = ()
    rollback: str = "Discard the versioned candidate directory."


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reason: str
    baseline_score: float
    candidate_score: float
    minimum_improvement: float
    evaluation_id: str
    kernel_sha256: str | None = None


class KernelWorkshop:
    """Build new candidate directories; never modify the frozen source bundle."""

    def __init__(self, source_root: str | Path, *, max_rounds: int = 5) -> None:
        self.source_root = Path(source_root).resolve()
        self.max_rounds = max(1, int(max_rounds))
        self.source_report = verify_frozen_kernel(self.source_root)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _document_name(scope: RepairScope) -> str:
        return {
            "schema": "schema.json",
            "function": "functions.json",
            "graph": "graph.json",
        }[scope]

    @staticmethod
    def _collections(scope: RepairScope, document: dict[str, Any]) -> dict[str, list]:
        if scope == "schema":
            return {
                "entity_types": document["entity_types"],
                "relations": document["relations"],
            }
        if scope == "function":
            return {"functions": document["functions"]}
        return {
            "entities": document["entities"],
            "relations": document["relations"],
        }

    @classmethod
    def _find_target(
        cls,
        scope: RepairScope,
        document: dict[str, Any],
        target: str,
    ) -> tuple[list, int, dict[str, Any]]:
        identity_key = "id" if scope == "graph" else "name"
        for collection in cls._collections(scope, document).values():
            for index, item in enumerate(collection):
                if isinstance(item, dict) and item.get(identity_key) == target:
                    return collection, index, item
        raise KernelValidationError(f"repair target not found: {scope}:{target}")

    @staticmethod
    def _value_at(target: dict[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = target
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise KernelValidationError(
                    f"repair path not found: {'.'.join(path)}"
                )
            current = current[key]
        return current

    @classmethod
    def _apply_repair(
        cls,
        documents: dict[str, dict[str, Any]],
        repair: RepairAction,
    ) -> None:
        if repair.scope not in {"schema", "function", "graph"}:
            raise KernelValidationError(f"unknown repair scope: {repair.scope}")
        if repair.action not in {"add", "delete", "modify"}:
            raise KernelValidationError(f"unknown repair action: {repair.action}")
        document_name = cls._document_name(repair.scope)
        document = documents[document_name]
        if repair.action == "add":
            collections = cls._collections(repair.scope, document)
            if repair.collection not in collections:
                raise KernelValidationError(
                    f"repair add requires a valid collection: {repair.collection}"
                )
            if not isinstance(repair.value, dict):
                raise KernelValidationError("repair add value must be an object")
            collections[repair.collection].append(repair.value)
            return

        collection, index, target = cls._find_target(
            repair.scope, document, repair.target
        )
        if repair.action == "delete":
            if repair.path:
                raise KernelValidationError("repair delete does not accept a path")
            collection.pop(index)
            return

        if not repair.path:
            raise KernelValidationError("repair modify requires a non-empty path")
        current = cls._value_at(target, repair.path)
        if current != repair.expected:
            raise KernelValidationError(
                f"expected value mismatch at {repair.scope}:{repair.target}:"
                f"{'.'.join(repair.path)}"
            )
        parent: Any = target
        for key in repair.path[:-1]:
            parent = parent[key]
        parent[repair.path[-1]] = repair.value

    def build_candidate(
        self,
        destination: str | Path,
        *,
        repairs: Iterable[RepairAction],
        round_index: int,
        training_sample_ids: Iterable[str],
    ) -> dict[str, Any]:
        destination = Path(destination).resolve()
        if destination.exists():
            raise KernelValidationError(
                "candidate destination already exists; use a new versioned directory"
            )
        if not 1 <= int(round_index) <= self.max_rounds:
            raise KernelValidationError(
                f"round_index must be between 1 and {self.max_rounds}"
            )
        repair_list = list(repairs)
        if not repair_list:
            raise KernelValidationError("a construction round requires at least one repair")
        sample_ids = [str(value) for value in training_sample_ids]
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise KernelValidationError(
                "training_sample_ids must be a non-empty unique sample"
            )

        destination.mkdir(parents=True)
        for name in BUNDLE_FILES:
            shutil.copy2(self.source_root / name, destination / name)
        documents = {
            name: json.loads((destination / name).read_text(encoding="utf-8"))
            for name in BUNDLE_FILES
        }
        for repair in repair_list:
            self._apply_repair(documents, repair)
        for name, document in documents.items():
            self._write_json(destination / name, document)
        validation = validate_kernel(destination)
        receipt = {
            "schema_version": 1,
            "status": "candidate_valid",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_kernel_sha256": self.source_report["kernel_sha256"],
            "round_index": int(round_index),
            "max_rounds": self.max_rounds,
            "training_sample_ids": sample_ids,
            "repairs": [asdict(repair) for repair in repair_list],
            "validation": validation,
        }
        self._write_json(destination / "construction.json", receipt)
        return receipt

    def promote_candidate(
        self,
        candidate: str | Path,
        *,
        baseline_score: float,
        candidate_score: float,
        minimum_improvement: float,
        evaluation_id: str,
        critical_failures: Iterable[str] = (),
    ) -> PromotionDecision:
        candidate = Path(candidate).resolve()
        if (candidate / "kernel.lock.json").exists():
            raise KernelValidationError("candidate is already frozen")
        validate_kernel(candidate)
        if not evaluation_id.strip():
            raise KernelValidationError("evaluation_id must be non-empty")
        failures = sorted({str(value) for value in critical_failures if str(value)})
        improvement = float(candidate_score) - float(baseline_score)
        accepted = not failures and improvement >= float(minimum_improvement)
        if failures:
            reason = "critical_evaluation_failure"
        elif not accepted:
            reason = "insufficient_measured_improvement"
        else:
            reason = "measured_improvement_and_no_critical_failure"
        evaluation = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_score": float(baseline_score),
            "candidate_score": float(candidate_score),
            "minimum_improvement": float(minimum_improvement),
            "critical_failures": failures,
            "accepted": accepted,
            "reason": reason,
            "note": (
                "Scores are supplied by the named evaluator; this gate does not "
                "independently establish semantic correctness."
            ),
        }
        self._write_json(candidate / "evaluation.json", evaluation)
        kernel_sha256 = None
        if accepted:
            kernel_sha256 = freeze_kernel(candidate)["kernel_sha256"]
        return PromotionDecision(
            accepted=accepted,
            reason=reason,
            baseline_score=float(baseline_score),
            candidate_score=float(candidate_score),
            minimum_improvement=float(minimum_improvement),
            evaluation_id=evaluation_id,
            kernel_sha256=kernel_sha256,
        )
