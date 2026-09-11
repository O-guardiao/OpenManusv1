from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from app.oak.construction import KernelWorkshop, RepairAction
from app.oak.kernel import KernelValidationError, verify_frozen_kernel
from app.oak.runtime import DEFAULT_KERNEL_ROOT


@pytest.fixture
def workshop_root():
    root = Path(__file__).parents[2] / "workspace" / ".test-oak-workshop" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)
    parent = root.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def risk_repair() -> RepairAction:
    return RepairAction(
        scope="graph",
        action="modify",
        target="tool:web_search",
        path=("properties", "risk_score"),
        expected=3,
        value=4,
        rationale="A reviewed carrier matrix found a broader source surface.",
        evidence=("eval:aig-web-search-r1",),
        rollback="Discard the candidate directory; source bundle remains immutable.",
    )


def test_repair_round_creates_new_candidate_and_never_mutates_frozen_source(
    workshop_root: Path,
) -> None:
    source_before = verify_frozen_kernel(DEFAULT_KERNEL_ROOT)
    candidate = workshop_root / "openmanus-v2-candidate"
    workshop = KernelWorkshop(DEFAULT_KERNEL_ROOT)

    report = workshop.build_candidate(
        candidate,
        repairs=[risk_repair()],
        round_index=1,
        training_sample_ids=["case-hidden-unicode", "case-mcp-metadata"],
    )

    graph = json.loads((candidate / "graph.json").read_text(encoding="utf-8"))
    web_search = next(item for item in graph["entities"] if item["id"] == "tool:web_search")
    assert report["status"] == "candidate_valid"
    assert web_search["properties"]["risk_score"] == 4
    assert not (candidate / "kernel.lock.json").exists()
    assert verify_frozen_kernel(DEFAULT_KERNEL_ROOT) == source_before


def test_promotion_needs_measured_improvement_and_freezes_accepted_candidate(
    workshop_root: Path,
) -> None:
    candidate = workshop_root / "candidate"
    workshop = KernelWorkshop(DEFAULT_KERNEL_ROOT)
    workshop.build_candidate(
        candidate,
        repairs=[risk_repair()],
        round_index=1,
        training_sample_ids=["case-1"],
    )

    rejected = workshop.promote_candidate(
        candidate,
        baseline_score=0.80,
        candidate_score=0.80,
        minimum_improvement=0.01,
        evaluation_id="eval-no-gain",
    )
    assert rejected.accepted is False
    assert not (candidate / "kernel.lock.json").exists()

    accepted = workshop.promote_candidate(
        candidate,
        baseline_score=0.80,
        candidate_score=0.83,
        minimum_improvement=0.01,
        evaluation_id="eval-gain",
    )
    assert accepted.accepted is True
    assert verify_frozen_kernel(candidate)["status"] == "verified"


def test_repair_fails_closed_when_expected_value_does_not_match(
    workshop_root: Path,
) -> None:
    candidate = workshop_root / "candidate"
    repair = risk_repair()
    repair = RepairAction(**{**repair.__dict__, "expected": 999})

    with pytest.raises(KernelValidationError, match="expected value mismatch"):
        KernelWorkshop(DEFAULT_KERNEL_ROOT).build_candidate(
            candidate,
            repairs=[repair],
            round_index=1,
            training_sample_ids=["case-1"],
        )
