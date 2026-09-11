"""Executable runtime contracts derived from the OpenManus/OaK formal model.

This module is intentionally deterministic and stdlib-only.  It does not try
to prove that an external effect happened; it keeps effect reality, observed
responses, budgets, and terminal claims separate and checks the event trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


RUNTIME_CONTRACT_VERSION = "oak-openmanus-runtime-v1"
RUNTIME_CHAIN_DOMAIN = "openmanus-oak-runtime-event-v1"
BUDGET_DIMENSIONS = (
    "steps",
    "model_calls",
    "tool_calls",
    "retries",
    "wall_time_ms",
    "tokens",
    "context_bytes",
    "external_effects",
)
UNRESOLVED_EFFECT_STATUSES = {"dispatched", "unknown", "partial"}
RESOLVED_EFFECT_STATUSES = {
    "succeeded",
    "failed_before_effect",
    "verified_applied",
    "verified_absent",
    "compensated",
    "manual_repair",
}


class FormalRuntimeError(RuntimeError):
    """Base error for an executable contract violation."""


class BudgetExceeded(FormalRuntimeError):
    """Raised before work that would exceed a declared global budget."""


class EffectConflict(FormalRuntimeError):
    """Raised when an operation identity is reused inconsistently or unsafely."""


@dataclass(frozen=True)
class BudgetLimits:
    steps: int | None = 30
    model_calls: int | None = 30
    tool_calls: int | None = 120
    retries: int | None = 3
    wall_time_ms: int | None = 1_800_000
    tokens: int | None = 1_000_000
    context_bytes: int | None = 8_000_000
    external_effects: int | None = 30

    def __post_init__(self) -> None:
        for dimension, value in asdict(self).items():
            if value is not None and value < 0:
                raise ValueError(f"budget limit {dimension} must be non-negative")

    @classmethod
    def for_steps(cls, steps: int) -> "BudgetLimits":
        bounded = max(0, int(steps))
        return cls(
            steps=bounded,
            model_calls=bounded,
            tool_calls=bounded * 4,
            retries=max(1, bounded),
            external_effects=bounded * 2,
        )

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class EffectRecord:
    operation_id: str
    principal: str
    tool_name: str
    arguments_sha256: str
    idempotency_key: str
    gate_event_id: str
    status: str = "intent_logged"
    dispatch_count: int = 0


@dataclass(frozen=True)
class RuntimeConformance:
    valid: bool
    errors: tuple[str, ...]
    event_count: int
    head_hash: str
    evidence_count: int
    unresolved_effects: int
    budget_exhausted: bool
    terminal_status: str
    budget_usage: Mapping[str, int]
    provider_states: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["errors"] = list(self.errors)
        value["budget_usage"] = dict(self.budget_usage)
        value["provider_states"] = dict(self.provider_states)
        return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def approval_reference(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Human-portable exact argument approval reference.

    This is not itself the grant key.  The runtime derives the full key from
    principal, state, and policy versions so the grant cannot cross contexts.
    """

    return f"{tool_name.strip().lower()}:{canonical_sha256(dict(arguments))}"


def approval_key(
    *,
    principal: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    state_version: str,
    policy_version: str,
) -> str:
    preimage = {
        "domain": "openmanus-exact-approval-v1",
        "principal": principal,
        "tool": tool_name.strip().lower(),
        "arguments_sha256": canonical_sha256(dict(arguments)),
        "state_version": state_version,
        "policy_version": policy_version,
    }
    return f"sha256:{canonical_sha256(preimage)}"


def idempotency_key(
    *,
    session_id: str,
    principal: str,
    operation_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> str:
    preimage = {
        "domain": "openmanus-effect-operation-v1",
        "session_id": session_id,
        "principal": principal,
        "operation_id": operation_id,
        "tool": tool_name.strip().lower(),
        "arguments_sha256": canonical_sha256(dict(arguments)),
    }
    return f"sha256:{canonical_sha256(preimage)}"


def runtime_event_hash(event: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in event.items() if key != "event_hash"}
    return sha256_text(f"{RUNTIME_CHAIN_DOMAIN}\n{canonical_json(payload)}")


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def verify_runtime_trace(events: Sequence[Mapping[str, Any]]) -> RuntimeConformance:
    """Recompute the hash chain and model/runtime refinement obligations."""

    errors: list[str] = []
    previous_hash = ""
    session_id = ""
    task_started = 0
    task_finished = 0
    task_kernel = ""
    terminal_status = ""
    evidence_count = 0
    budget_exhausted = False
    budgets = {dimension: 0 for dimension in BUDGET_DIMENSIONS}
    allowed_gates: set[str] = set()
    effects: dict[str, str] = {}
    idempotency_keys: set[str] = set()
    provider_states: dict[str, str] = {}
    declared_limits: dict[str, int | None] = {}
    terminal_metadata: dict[str, Any] = {}
    seen_event_ids: set[str] = set()

    if not events:
        errors.append("trace must not be empty")

    for expected_position, raw in enumerate(events, start=1):
        event = dict(raw)
        label = f"event {expected_position}"
        event_type = str(event.get("event_type", ""))
        if task_finished:
            errors.append(f"{label} appears after the terminal event")
        if event.get("schema_version") != 1:
            errors.append(f"{label} schema_version mismatch")
        if event.get("contract_version") != RUNTIME_CONTRACT_VERSION:
            errors.append(f"{label} contract_version mismatch")
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in seen_event_ids:
            errors.append(f"{label} duplicate or empty event identity")
        seen_event_ids.add(event_id)
        if expected_position == 1 and event_type != "task_started":
            errors.append(f"{label} trace must start with task_started")
        elif expected_position > 1 and task_started == 0:
            errors.append(f"{label} appears before task_started")
        if event.get("position") != expected_position:
            errors.append(f"{label} position mismatch")
        if event.get("prev_hash") != previous_hash:
            errors.append(f"{label} prev_hash mismatch")
        current_session = str(event.get("session_id", ""))
        if expected_position == 1:
            session_id = current_session
        elif current_session != session_id:
            errors.append(f"{label} session_id mismatch")
        calculated = runtime_event_hash(event)
        if event.get("event_hash") != calculated:
            errors.append(f"{label} event_hash mismatch")
        previous_hash = calculated

        if event_type == "task_started":
            task_started += 1
            if task_started > 1 or task_finished:
                errors.append(f"{label} invalid task start ordering")
            task_kernel = str(event.get("kernel_sha256", ""))
            if not task_kernel:
                errors.append(f"{label} missing frozen kernel hash")
            raw_limits = event.get("budget_limits")
            if not isinstance(raw_limits, Mapping):
                errors.append(f"{label} missing budget limits")
            else:
                declared_limits = {
                    dimension: raw_limits.get(dimension)
                    for dimension in BUDGET_DIMENSIONS
                }
                for dimension, limit in declared_limits.items():
                    if limit is not None and not _is_nonnegative_int(limit):
                        errors.append(
                            f"{label} invalid budget limit for {dimension}"
                        )
        elif event_type == "task_finished":
            task_finished += 1
            terminal_status = str(event.get("status", ""))
            if not terminal_status:
                errors.append(f"{label} missing terminal status")
            if task_started != 1 or task_finished > 1:
                errors.append(f"{label} invalid task finish ordering")
            if event.get("kernel_sha256") != task_kernel:
                errors.append(f"{label} kernel changed during task")
            terminal_metadata = event
        elif event_type == "budget_consumed":
            dimension = str(event.get("dimension", ""))
            if dimension not in budgets:
                errors.append(f"{label} unknown budget dimension")
                continue
            delta = event.get("delta")
            used = event.get("used")
            limit = event.get("limit")
            if not _is_nonnegative_int(delta):
                errors.append(f"{label} invalid budget delta")
                continue
            if not _is_nonnegative_int(used):
                errors.append(f"{label} invalid budget used value")
                continue
            if limit is not None and not _is_nonnegative_int(limit):
                errors.append(f"{label} invalid budget limit")
                continue
            budgets[dimension] += delta
            if used != budgets[dimension]:
                errors.append(f"{label} non-monotonic budget usage")
            if declared_limits and limit != declared_limits.get(dimension):
                errors.append(f"{label} budget limit differs from task contract")
            if limit is not None and budgets[dimension] > limit:
                errors.append(f"{label} budget was exceeded after admission")
        elif event_type == "budget_rejected":
            budget_exhausted = True
            dimension = str(event.get("dimension", ""))
            if dimension not in budgets:
                errors.append(f"{label} unknown rejected budget dimension")
        elif event_type in {"source_result", "internal_result", "final_evidence"}:
            evidence_count += 1
        elif (
            event_type == "sink_decision"
            and bool(event.get("allowed"))
            and event.get("outcome") == "allow"
        ):
            gate_id = str(event.get("event_id", ""))
            if not gate_id:
                errors.append(f"{label} allowed sink gate has no identity")
            if event.get("taint_ids") and not str(event.get("approval_key", "")):
                errors.append(f"{label} tainted sink lacks exact approval key")
            allowed_gates.add(gate_id)
        elif event_type == "effect_intent":
            operation_id = str(event.get("operation_id", ""))
            if not operation_id or operation_id in effects:
                errors.append(f"{label} duplicate or empty effect operation")
            if str(event.get("gate_event_id", "")) not in allowed_gates:
                errors.append(f"{label} effect intent lacks a prior allowed gate")
            effect_key = str(event.get("idempotency_key", ""))
            if not effect_key:
                errors.append(f"{label} effect intent lacks idempotency key")
            elif effect_key in idempotency_keys:
                errors.append(f"{label} duplicate effect idempotency key")
            idempotency_keys.add(effect_key)
            effects[operation_id] = "intent_logged"
        elif event_type == "effect_dispatched":
            operation_id = str(event.get("operation_id", ""))
            if effects.get(operation_id) != "intent_logged":
                errors.append(f"{label} dispatch without durable intent")
            effects[operation_id] = "dispatched"
        elif event_type == "effect_observed":
            operation_id = str(event.get("operation_id", ""))
            status = str(event.get("status", ""))
            if effects.get(operation_id) not in UNRESOLVED_EFFECT_STATUSES | {
                "intent_logged"
            }:
                errors.append(f"{label} observation without live operation")
            if status not in UNRESOLVED_EFFECT_STATUSES | RESOLVED_EFFECT_STATUSES:
                errors.append(f"{label} invalid effect status")
            effects[operation_id] = status
        elif event_type == "provider_state":
            provider_id = str(event.get("provider_sha256", ""))
            before = provider_states.get(provider_id, "absent")
            after = str(event.get("state_after", ""))
            allowed_transitions = {
                "absent": {"opening"},
                "opening": {"active", "absent", "failed_cleanup"},
                "active": {"draining"},
                "draining": {"absent", "failed_cleanup"},
                "failed_cleanup": {"draining"},
            }
            if not provider_id or event.get("state_before") != before:
                errors.append(f"{label} provider pre-state mismatch")
            if after not in allowed_transitions.get(before, set()):
                errors.append(f"{label} invalid provider transition")
            published_tool_count = event.get("published_tool_count", 0)
            if (
                not isinstance(published_tool_count, int)
                or isinstance(published_tool_count, bool)
                or published_tool_count < 0
            ):
                errors.append(f"{label} invalid published tool count")
            elif after != "active" and published_tool_count:
                errors.append(f"{label} non-active provider published tools")
            provider_states[provider_id] = after

    unresolved_effects = sum(
        status in UNRESOLVED_EFFECT_STATUSES for status in effects.values()
    )
    if events and (task_started != 1 or task_finished != 1):
        errors.append("trace must contain exactly one task start and finish")
    if terminal_metadata:
        terminal_evidence = terminal_metadata.get("evidence_count")
        terminal_unresolved = terminal_metadata.get("unresolved_effects")
        terminal_budget_flag = terminal_metadata.get("budget_exhausted")
        terminal_usage = terminal_metadata.get("budget_usage")
        if (
            not _is_nonnegative_int(terminal_evidence)
            or terminal_evidence != evidence_count
        ):
            errors.append("terminal evidence count does not match the trace")
        if (
            not _is_nonnegative_int(terminal_unresolved)
            or terminal_unresolved != unresolved_effects
        ):
            errors.append("terminal unresolved-effect count does not match the trace")
        if (
            not isinstance(terminal_budget_flag, bool)
            or terminal_budget_flag != budget_exhausted
        ):
            errors.append("terminal budget flag does not match the trace")
        if (
            not isinstance(terminal_usage, Mapping)
            or any(
                not _is_nonnegative_int(terminal_usage.get(dimension))
                or terminal_usage.get(dimension) != used
                for dimension, used in budgets.items()
            )
        ):
            errors.append("terminal budget usage does not match the trace")
        if terminal_metadata.get("provider_states") != dict(sorted(provider_states.items())):
            errors.append("terminal provider states do not match the trace")
    if terminal_status == "completed" and evidence_count <= 0:
        errors.append("completed task has no evidence")
    if terminal_status == "completed" and unresolved_effects:
        errors.append("completed task has unresolved effects")
    if terminal_status == "completed" and budget_exhausted:
        errors.append("completed task exceeded a global budget")
    if terminal_status == "completed" and any(
        state != "absent" for state in provider_states.values()
    ):
        errors.append("completed task has a live or failed provider")
    return RuntimeConformance(
        valid=not errors,
        errors=tuple(errors),
        event_count=len(events),
        head_hash=previous_hash,
        evidence_count=evidence_count,
        unresolved_effects=unresolved_effects,
        budget_exhausted=budget_exhausted,
        terminal_status=terminal_status,
        budget_usage=budgets,
        provider_states=dict(sorted(provider_states.items())),
    )
