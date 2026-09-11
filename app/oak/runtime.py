"""Intrinsic OaK session runtime used by planning and tool execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

from app.oak.kernel import execute_function, verify_frozen_kernel
from app.oak.formal_runtime import (
    BUDGET_DIMENSIONS,
    RESOLVED_EFFECT_STATUSES,
    UNRESOLVED_EFFECT_STATUSES,
    BudgetExceeded,
    BudgetLimits,
    EffectConflict,
    EffectRecord,
    RuntimeConformance,
    approval_key,
    canonical_sha256,
    idempotency_key,
    runtime_event_hash,
    verify_runtime_trace,
    RUNTIME_CONTRACT_VERSION,
)
from app.oak.security import (
    GuardDecision,
    IntegratedObservation,
    ToolProfile,
    integrate_untrusted_observation,
)
from app.schema import MessageProvenance


DEFAULT_KERNEL_ROOT = Path(__file__).resolve().parent / "bundles" / "openmanus-v1"


SYSTEM_PATTERNS = {
    "integrator",
    "retriever",
    "recorder",
    "selector",
    "planner",
    "deliberator",
    "executor",
    "tool_use",
    "coordinator",
    "reflector",
    "skill_build",
    "controller",
}


@dataclass(frozen=True)
class RepairProposal:
    proposal_id: str
    scope: str
    action: str
    delta: str
    rollback: str
    evidence_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class StepAssessment:
    complete: bool
    status: str
    reason: str
    evidence_added: int


class OakRuntime:
    """Typed, evidence-aware controller around the probabilistic agent loop.

    The frozen bundle is verified on construction.  Runtime observations form an
    in-memory typed graph, while callers may persist the metadata-only event
    stream through ``ToolPolicy``.  Schema/function repair is proposed but never
    self-applied to a frozen kernel.
    """

    def __init__(
        self,
        *,
        kernel_root: str | Path = DEFAULT_KERNEL_ROOT,
        session_id: str | None = None,
        repair_threshold: int = 3,
        budget_limits: BudgetLimits | None = None,
        policy_version: str = "intrinsic-oak-tool-policy-v2",
    ) -> None:
        self.kernel_root = Path(kernel_root).resolve()
        self.kernel_report = verify_frozen_kernel(self.kernel_root)
        self.session_id = session_id or str(uuid4())
        self.repair_threshold = max(2, int(repair_threshold))
        self.policy_version = policy_version
        self.budget_limits = budget_limits or BudgetLimits()
        self._budget_usage = {dimension: 0 for dimension in BUDGET_DIMENSIONS}
        self._budget_exhausted = False
        self._gate_failed = False
        self._effects: dict[str, EffectRecord] = {}
        self._effect_fingerprints: dict[tuple[str, str], str] = {}
        self._unavailable_alternatives: set[str] = set()
        self._provider_states: dict[str, str] = {}
        self._task_started_at: float | None = None
        self._event_head = ""
        self._events: list[dict[str, Any]] = []
        self._active_taint_ids: set[str] = set()
        self._activated_patterns: set[str] = set()
        self._profile_cache: dict[str, ToolProfile] = {}
        self._failure_counts: Counter[tuple[str, str]] = Counter()
        self._failure_event_ids: dict[tuple[str, str], list[str]] = {}
        self._repair_proposals: list[RepairProposal] = []
        self._evidence_count = 0
        self._current_task_id: str | None = None
        self._last_task_status: str | None = None
        self._runtime_graph: dict[str, Any] = {
            "schema_version": 1,
            "entities": [],
            "relations": [],
        }

    @property
    def active_taint_ids(self) -> set[str]:
        return set(self._active_taint_ids)

    @property
    def activated_patterns(self) -> set[str]:
        return set(self._activated_patterns)

    @property
    def repair_proposals(self) -> tuple[RepairProposal, ...]:
        return tuple(self._repair_proposals)

    @property
    def trace(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)

    @property
    def evidence_count(self) -> int:
        return self._evidence_count

    @property
    def budget_usage(self) -> dict[str, int]:
        return dict(self._budget_usage)

    @property
    def budget_exhausted(self) -> bool:
        return self._budget_exhausted

    @property
    def unresolved_effects(self) -> tuple[EffectRecord, ...]:
        return tuple(
            effect
            for effect in self._effects.values()
            if effect.status in UNRESOLVED_EFFECT_STATUSES
        )

    @property
    def task_active(self) -> bool:
        """Whether a root task currently owns this runtime session."""

        return self._current_task_id is not None

    @property
    def last_task_status(self) -> str | None:
        return self._last_task_status

    @property
    def runtime_graph(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "entities": [dict(entity) for entity in self._runtime_graph["entities"]],
            "relations": [dict(edge) for edge in self._runtime_graph["relations"]],
        }

    def _activate(self, *patterns: str) -> None:
        unknown = set(patterns) - SYSTEM_PATTERNS
        if unknown:
            raise ValueError(f"Unknown system patterns: {sorted(unknown)}")
        self._activated_patterns.update(patterns)

    def _event(self, event_type: str, **metadata: Any) -> dict[str, Any]:
        event = {
            "schema_version": 1,
            "contract_version": RUNTIME_CONTRACT_VERSION,
            "event_id": str(uuid4()),
            "event_type": event_type,
            "session_id": self.session_id,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "position": len(self._events) + 1,
            "prev_hash": self._event_head,
            **metadata,
        }
        event["event_hash"] = runtime_event_hash(event)
        self._event_head = event["event_hash"]
        self._events.append(event)
        return event

    def begin_task(self, request: str) -> str:
        if self._current_task_id is not None:
            raise RuntimeError("a task is already active in this OaK runtime")
        # Receipts and conformance are task-scoped.  A reused runtime starts a
        # fresh chain while the caller may already have persisted the prior one.
        self._events = []
        self._event_head = ""
        self._budget_usage = {dimension: 0 for dimension in BUDGET_DIMENSIONS}
        self._budget_exhausted = False
        self._gate_failed = False
        self._effects = {}
        self._effect_fingerprints = {}
        self._unavailable_alternatives = set()
        self._provider_states = {}
        self._evidence_count = 0
        self._active_taint_ids = set()
        self._runtime_graph = {
            "schema_version": 1,
            "entities": [],
            "relations": [],
        }
        self._task_started_at = time.perf_counter()
        self._activate("integrator", "recorder")
        task_id = f"task:{uuid4()}"
        request_sha256 = hashlib.sha256(request.encode("utf-8")).hexdigest()
        self._current_task_id = task_id
        self._runtime_graph["entities"].append(
            {
                "id": task_id,
                "type": "Task",
                "properties": {
                    "request": request,
                    "request_sha256": request_sha256,
                    "status": "running",
                },
                "evidence": [{"source": "operator", "locator": request_sha256}],
            }
        )
        self._event(
            "task_started",
            task_id=task_id,
            request_sha256=request_sha256,
            kernel_sha256=self.kernel_report["kernel_sha256"],
            policy_version=self.policy_version,
            budget_limits=self.budget_limits.to_dict(),
        )
        return task_id

    def finish_task(self, status: str) -> str:
        if self._task_started_at is not None:
            elapsed = max(0, round((time.perf_counter() - self._task_started_at) * 1000))
            delta = max(0, elapsed - self._budget_usage["wall_time_ms"])
            if delta:
                try:
                    self.consume_budget("wall_time_ms", delta, cycle_id="task")
                except BudgetExceeded:
                    pass
        requested_status = status
        terminal_reasons = []
        if self.unresolved_effects:
            terminal_reasons.append("unresolved_effect")
        if any(
            provider_state == "failed_cleanup"
            for provider_state in self._provider_states.values()
        ):
            terminal_reasons.append("provider_cleanup_failed")
        elif any(
            provider_state != "absent"
            for provider_state in self._provider_states.values()
        ):
            terminal_reasons.append("provider_not_absent")
        if self._budget_exhausted:
            terminal_reasons.append("budget_exhausted")
        if self._gate_failed:
            terminal_reasons.append("gate_failed")
        if self._evidence_count <= 0:
            terminal_reasons.append("no_evidence")
        if status == "completed" and self.unresolved_effects:
            status = "unknown_outcome"
        elif status == "completed" and any(
            provider_state == "failed_cleanup"
            for provider_state in self._provider_states.values()
        ):
            status = "failed_cleanup"
        elif status == "completed" and any(
            provider_state != "absent"
            for provider_state in self._provider_states.values()
        ):
            status = "failed_visible"
        elif status == "completed" and self._budget_exhausted:
            status = "budget_exhausted"
        elif status == "completed" and self._gate_failed:
            status = "failed_visible"
        elif status == "completed" and self._evidence_count <= 0:
            status = "failed_visible"
        task_id = self._current_task_id
        if task_id:
            for entity in self._runtime_graph["entities"]:
                if entity["id"] == task_id:
                    entity["properties"]["status"] = status
                    break
        self._activate("recorder")
        self._event(
            "task_finished",
            task_id=task_id,
            status=status,
            requested_status=requested_status,
            evidence_count=self._evidence_count,
            active_taint_count=len(self._active_taint_ids),
            unresolved_effects=len(self.unresolved_effects),
            budget_exhausted=self._budget_exhausted,
            budget_usage=dict(self._budget_usage),
            provider_states=dict(sorted(self._provider_states.items())),
            terminal_reasons=terminal_reasons,
            kernel_sha256=self.kernel_report["kernel_sha256"],
        )
        self._last_task_status = status
        self._current_task_id = None
        self._task_started_at = None
        return status

    def conformance_report(self) -> RuntimeConformance:
        return verify_runtime_trace(self.trace)

    def record_gate_failure(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        error_type: str,
    ) -> None:
        """Turn a gate exception into an observable fail-closed decision."""

        self._gate_failed = True
        self._event(
            "gate_error",
            tool=tool_name,
            arguments_sha256=canonical_sha256(dict(arguments)),
            outcome="reject",
            error_type=error_type,
            state_version=self.kernel_report["kernel_sha256"],
            policy_version=self.policy_version,
        )

    def record_unavailable_alternative(
        self,
        tool_name: str,
        *,
        reason: str,
    ) -> bool:
        """Remember permanent unavailability and reject repeated selection.

        Returns ``True`` for the first observation and ``False`` when the same
        unavailable capability is selected again in the root task.
        """

        normalized = tool_name.strip().lower()
        if normalized in self._unavailable_alternatives:
            try:
                self.consume_budget(
                    "retries",
                    cycle_id=f"unavailable:{normalized}",
                )
            except BudgetExceeded:
                pass
            self._event(
                "alternative_retry_rejected",
                tool=normalized,
                reason=reason,
                permanence="task",
            )
            return False
        self._unavailable_alternatives.add(normalized)
        self._event(
            "alternative_unavailable",
            tool=normalized,
            reason=reason,
            permanence="task",
        )
        return True

    def record_provider_state(
        self,
        provider_id: str,
        state: str,
        *,
        published_tool_count: int = 0,
        cleanup_error_type: str = "",
    ) -> None:
        """Record an MCP/provider lifecycle transition without its raw identity."""

        provider_sha256 = hashlib.sha256(provider_id.encode("utf-8")).hexdigest()
        before = self._provider_states.get(provider_sha256, "absent")
        allowed = {
            "absent": {"opening"},
            "opening": {"active", "absent", "failed_cleanup"},
            "active": {"draining"},
            "draining": {"absent", "failed_cleanup"},
            "failed_cleanup": {"draining"},
        }
        if state not in allowed.get(before, set()):
            self._gate_failed = True
            raise RuntimeError(
                f"invalid provider transition {before}->{state}"
            )
        self._provider_states[provider_sha256] = state
        self._event(
            "provider_state",
            provider_sha256=provider_sha256,
            state_before=before,
            state_after=state,
            published_tool_count=max(0, int(published_tool_count)),
            cleanup_error_type=cleanup_error_type,
        )

    def configure_budget(self, limits: BudgetLimits) -> None:
        if self.task_active:
            raise RuntimeError("cannot replace budgets during an active task")
        self.budget_limits = limits

    def consume_budget(
        self,
        dimension: str,
        amount: int = 1,
        *,
        cycle_id: str,
    ) -> int:
        if dimension not in self._budget_usage:
            raise ValueError(f"unknown budget dimension: {dimension}")
        amount = int(amount)
        if amount < 0:
            raise ValueError("budget consumption cannot be negative")
        current = self._budget_usage[dimension]
        requested = current + amount
        limit = getattr(self.budget_limits, dimension)
        if limit is not None and requested > limit:
            self._budget_exhausted = True
            self._event(
                "budget_rejected",
                dimension=dimension,
                used=current,
                requested=requested,
                limit=limit,
                cycle_id=cycle_id,
            )
            raise BudgetExceeded(
                f"global {dimension} budget exceeded: {requested}>{limit}"
            )
        self._budget_usage[dimension] = requested
        self._event(
            "budget_consumed",
            dimension=dimension,
            delta=amount,
            used=requested,
            limit=limit,
            cycle_id=cycle_id,
        )
        return requested

    def observe_context_bytes(self, size: int, *, cycle_id: str = "main") -> int:
        size = max(0, int(size))
        current = self._budget_usage["context_bytes"]
        if size <= current:
            return current
        return self.consume_budget(
            "context_bytes", size - current, cycle_id=cycle_id
        )

    def observe_token_total(self, total: int, *, cycle_id: str = "main") -> int:
        total = max(0, int(total))
        current = self._budget_usage["tokens"]
        if total <= current:
            return current
        return self.consume_budget("tokens", total - current, cycle_id=cycle_id)

    def begin_effect(
        self,
        *,
        operation_id: str,
        principal: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        gate_event_id: str,
    ) -> EffectRecord:
        operation_id = operation_id.strip()
        if not operation_id:
            raise ValueError("operation_id must not be empty")
        normalized_tool = tool_name.strip().lower()
        arguments_sha256 = canonical_sha256(dict(arguments))
        fingerprint = (normalized_tool, arguments_sha256)
        existing = self._effects.get(operation_id)
        if existing is not None:
            if (
                existing.tool_name != normalized_tool
                or existing.arguments_sha256 != arguments_sha256
            ):
                raise EffectConflict(
                    "operation_id already belongs to different arguments"
                )
            if existing.status in UNRESOLVED_EFFECT_STATUSES:
                self.consume_budget("retries", cycle_id="effect-reentry")
                raise EffectConflict(
                    "operation outcome is unresolved; automatic retry is forbidden"
                )
            return existing

        prior_id = self._effect_fingerprints.get(fingerprint)
        if prior_id is not None:
            prior = self._effects[prior_id]
            self.consume_budget("retries", cycle_id="effect-reentry")
            if prior.status in UNRESOLVED_EFFECT_STATUSES:
                raise EffectConflict(
                    "matching effect is unresolved; reconcile before retry"
                )

        record = EffectRecord(
            operation_id=operation_id,
            principal=principal,
            tool_name=normalized_tool,
            arguments_sha256=arguments_sha256,
            idempotency_key=idempotency_key(
                session_id=self.session_id,
                principal=principal,
                operation_id=operation_id,
                tool_name=normalized_tool,
                arguments=arguments,
            ),
            gate_event_id=gate_event_id,
        )
        self._effects[operation_id] = record
        self._effect_fingerprints[fingerprint] = operation_id
        self._event(
            "effect_intent",
            operation_id=operation_id,
            principal=principal,
            tool=normalized_tool,
            arguments_sha256=arguments_sha256,
            idempotency_key=record.idempotency_key,
            gate_event_id=gate_event_id,
        )
        return record

    def dispatch_effect(self, operation_id: str) -> EffectRecord:
        record = self._effects.get(operation_id)
        if record is None:
            raise EffectConflict("effect dispatch has no durable intent")
        if record.status != "intent_logged":
            raise EffectConflict(f"cannot dispatch effect in state {record.status}")
        self.consume_budget("external_effects", cycle_id="tool-effect")
        updated = replace(record, status="dispatched", dispatch_count=1)
        self._effects[operation_id] = updated
        self._event(
            "effect_dispatched",
            operation_id=operation_id,
            tool=record.tool_name,
            idempotency_key=record.idempotency_key,
            dispatch_count=1,
        )
        return updated

    def observe_effect(self, operation_id: str, status: str) -> EffectRecord:
        allowed = UNRESOLVED_EFFECT_STATUSES | RESOLVED_EFFECT_STATUSES
        if status not in allowed:
            raise ValueError(f"unsupported effect status: {status}")
        record = self._effects.get(operation_id)
        if record is None or record.status not in {
            "intent_logged",
            "dispatched",
            "unknown",
            "partial",
        }:
            raise EffectConflict("effect observation has no live operation")
        updated = replace(record, status=status)
        self._effects[operation_id] = updated
        self._event(
            "effect_observed",
            operation_id=operation_id,
            tool=record.tool_name,
            status=status,
            idempotency_key=record.idempotency_key,
        )
        return updated

    def record_final_response(self, result: str) -> None:
        result_text = str(result)
        self._evidence_count += 1
        self._event(
            "final_evidence",
            content_sha256=hashlib.sha256(result_text.encode("utf-8")).hexdigest(),
            characters=len(result_text),
        )

    def query(self, function_name: str, arguments: Mapping[str, Any]) -> Any:
        self._activate("retriever")
        result = execute_function(
            self.kernel_root,
            function_name,
            dict(arguments),
        )
        self._event(
            "kernel_query",
            function=function_name,
            argument_names=sorted(arguments),
            result_count=len(result) if isinstance(result, list) else 1,
        )
        return result

    def tool_profile(self, tool_name: str) -> ToolProfile:
        normalized_name = tool_name.strip().lower()
        if normalized_name in self._profile_cache:
            self._activate("retriever")
            return self._profile_cache[normalized_name]
        rows = self.query("get_tool_profile", {"tool_name": normalized_name})
        profile = (
            ToolProfile.from_row(rows[0], tool_name=normalized_name)
            if rows
            else ToolProfile.unknown(normalized_name)
        )
        self._profile_cache[normalized_name] = profile
        return profile

    def authorize_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        granted: bool,
        tainted_sink_approved: bool,
        principal: str = "local-operator",
        policy_version: str | None = None,
    ) -> GuardDecision:
        self._activate("executor", "tool_use", "controller")
        profile = self.tool_profile(tool_name)
        taints = tuple(sorted(self._active_taint_ids))
        if profile.requires_grant and not granted:
            allowed = False
            reason = "not_granted_for_session"
            outcome = "reject"
        elif (
            profile.sink
            and profile.tainted_sink_requires_approval
            and taints
            and not tainted_sink_approved
        ):
            allowed = False
            reason = "tainted_source_to_sink_requires_approval"
            outcome = "require_approval"
        elif profile.sink and taints and tainted_sink_approved:
            allowed = True
            reason = "explicit_tainted_sink_approval"
            outcome = "allow"
        else:
            allowed = True
            reason = "declared_capability"
            outcome = "allow"
        state_version = self.kernel_report["kernel_sha256"]
        active_policy_version = policy_version or self.policy_version
        arguments_sha256 = canonical_sha256(dict(arguments))
        exact_key = (
            approval_key(
                principal=principal,
                tool_name=tool_name,
                arguments=arguments,
                state_version=state_version,
                policy_version=active_policy_version,
            )
            if tainted_sink_approved
            else ""
        )
        event_type = "sink_decision" if profile.sink else "gate_decision"
        gate_event = self._event(
            event_type,
            tool=tool_name,
            allowed=allowed,
            outcome=outcome,
            reason=reason,
            taint_ids=list(taints),
            sink_fired=False,
            argument_names=sorted(arguments),
            arguments_sha256=arguments_sha256,
            approval_key=exact_key,
            state_version=state_version,
            policy_version=active_policy_version,
        )
        decision = GuardDecision(
            tool_name=tool_name,
            allowed=allowed,
            reason=reason,
            profile=profile,
            taint_ids=taints,
            event_id=gate_event["event_id"],
            outcome=outcome,
            arguments_sha256=arguments_sha256,
            approval_key=exact_key,
            state_version=state_version,
            policy_version=active_policy_version,
        )
        return decision

    def would_authorize_tool(
        self,
        tool_name: str,
        *,
        granted: bool,
        tainted_sink_approved: bool,
    ) -> bool:
        profile = self.tool_profile(tool_name)
        if profile.requires_grant and not granted:
            return False
        return not (
            profile.sink
            and profile.tainted_sink_requires_approval
            and self._active_taint_ids
            and not tainted_sink_approved
        )

    def integrate_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        max_content_length: int | None = None,
    ) -> IntegratedObservation:
        self._activate("integrator", "recorder")
        profile = self.tool_profile(tool_name)
        raw_content = str(content)
        if profile.source:
            sequence = len(self._events)
            taint_id = "taint:" + hashlib.sha256(
                f"{self.session_id}:{sequence}:{tool_name}".encode("utf-8")
            ).hexdigest()[:20]
            observation = integrate_untrusted_observation(
                source=tool_name,
                content=raw_content,
                carrier_type=profile.carrier_type,
                taint_id=taint_id,
                max_content_length=max_content_length,
            )
            self._active_taint_ids.add(taint_id)
            event = self._event(
                "source_result",
                tool=tool_name,
                taint_ids=[taint_id],
                carrier_type=profile.carrier_type,
                content_sha256=observation.provenance.content_sha256,
                normalized_sha256=observation.provenance.normalized_sha256,
                transformations=observation.provenance.transformations,
            )
        else:
            content_sha256 = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
            observation = IntegratedObservation(
                model_content=raw_content,
                provenance=MessageProvenance(
                    source=tool_name,
                    source_kind="tool",
                    trust_tier=profile.trust_tier,
                    carrier_type=profile.carrier_type,
                    content_sha256=content_sha256,
                    normalized_sha256=content_sha256,
                    taint_ids=[],
                    transformations=[],
                ),
            )
            event = self._event(
                "internal_result",
                tool=tool_name,
                content_sha256=content_sha256,
            )
        self._evidence_count += 1
        evidence_id = f"evidence:{event['event_id']}"
        self._runtime_graph["entities"].append(
            {
                "id": evidence_id,
                "type": "Evidence",
                "properties": {
                    "source": tool_name,
                    "content": raw_content,
                    "content_sha256": observation.provenance.content_sha256,
                    "trust_tier": observation.provenance.trust_tier,
                    "taint_ids": list(observation.provenance.taint_ids),
                },
                "evidence": [
                    {"source": tool_name, "locator": event["event_id"]}
                ],
            }
        )
        if self._current_task_id:
            self._runtime_graph["relations"].append(
                {
                    "id": f"relation:{uuid4()}",
                    "type": "task_has_evidence",
                    "source": self._current_task_id,
                    "target": evidence_id,
                    "evidence": [
                        {"source": "runtime", "locator": event["event_id"]}
                    ],
                }
            )
        return observation

    def restore_provenance(self, provenance: MessageProvenance) -> bool:
        """Restore untrusted memory metadata into a fresh task trace.

        MCP instructions may be acquired before a request starts. ``begin_task``
        deliberately creates a fresh task-scoped chain, so model-visible memory
        must reintroduce its trust metadata without replaying raw content.
        Returns ``True`` only when at least one new taint was restored.
        """

        if provenance.trust_tier != "untrusted":
            return False
        taint_ids = set(provenance.taint_ids)
        if not taint_ids:
            taint_ids.add(
                "taint:restored:"
                + hashlib.sha256(
                    (
                        f"{self.session_id}:{provenance.source_kind}:"
                        f"{provenance.source}:{provenance.content_sha256}"
                    ).encode("utf-8")
                ).hexdigest()[:20]
            )
        new_taints = taint_ids - self._active_taint_ids
        if not new_taints:
            return False

        self._activate("integrator", "recorder", "controller")
        self._active_taint_ids.update(new_taints)
        self._evidence_count += 1
        event = self._event(
            "source_result",
            tool="restored_memory",
            source_kind=provenance.source_kind,
            source_sha256=hashlib.sha256(
                provenance.source.encode("utf-8")
            ).hexdigest(),
            taint_ids=sorted(new_taints),
            carrier_type=provenance.carrier_type,
            content_sha256=provenance.content_sha256,
            normalized_sha256=provenance.normalized_sha256,
            transformations=list(provenance.transformations),
            reconstructed_from_memory=True,
        )
        evidence_id = f"evidence:{event['event_id']}"
        self._runtime_graph["entities"].append(
            {
                "id": evidence_id,
                "type": "Evidence",
                "properties": {
                    "source_sha256": event["source_sha256"],
                    "content_sha256": provenance.content_sha256,
                    "trust_tier": provenance.trust_tier,
                    "taint_ids": sorted(new_taints),
                    "reconstructed_from_memory": True,
                },
                "evidence": [
                    {"source": "runtime", "locator": event["event_id"]}
                ],
            }
        )
        if self._current_task_id:
            self._runtime_graph["relations"].append(
                {
                    "id": f"relation:{uuid4()}",
                    "type": "task_has_evidence",
                    "source": self._current_task_id,
                    "target": evidence_id,
                    "evidence": [
                        {"source": "runtime", "locator": event["event_id"]}
                    ],
                }
            )
        return True

    def integrate_external_content(
        self,
        *,
        source: str,
        content: str,
        carrier_type: str,
        source_kind: str,
        max_content_length: int | None = None,
    ) -> IntegratedObservation:
        """Integrate code-adjacent metadata that reaches the model without a call."""

        self._activate("integrator", "recorder", "controller")
        sequence = len(self._events)
        taint_id = "taint:" + hashlib.sha256(
            f"{self.session_id}:{sequence}:{source}".encode("utf-8")
        ).hexdigest()[:20]
        observation = integrate_untrusted_observation(
            source=source,
            content=content,
            carrier_type=carrier_type,
            taint_id=taint_id,
            source_kind=source_kind,
            max_content_length=max_content_length,
        )
        self._active_taint_ids.add(taint_id)
        self._evidence_count += 1
        self._event(
            "source_result",
            tool=source,
            source_kind=source_kind,
            taint_ids=[taint_id],
            carrier_type=carrier_type,
            content_sha256=observation.provenance.content_sha256,
            normalized_sha256=observation.provenance.normalized_sha256,
            transformations=observation.provenance.transformations,
        )
        return observation

    def record_tool_outcome(
        self,
        tool_name: str,
        status: str,
        reason: str,
        *,
        operation_id: str | None = None,
    ) -> None:
        self._activate("executor", "recorder")
        profile = self.tool_profile(tool_name)
        event = self._event(
            "tool_outcome",
            tool=tool_name,
            status=status,
            reason=reason,
            sink_fired=profile.sink and status == "succeeded",
            taint_ids=sorted(self._active_taint_ids),
            operation_id=operation_id or "",
        )
        if status == "succeeded" and profile.sink:
            self._event(
                "sink_decision",
                tool=tool_name,
                allowed=True,
                reason=reason,
                taint_ids=sorted(self._active_taint_ids),
                sink_fired=True,
                operation_id=operation_id or "",
            )
        if status not in {"succeeded", "authorized"}:
            self._activate("reflector")
            key = (tool_name, reason)
            self._failure_counts[key] += 1
            self._failure_event_ids.setdefault(key, []).append(event["event_id"])
            if self._failure_counts[key] == self.repair_threshold:
                self._activate("skill_build")
                proposal = RepairProposal(
                    proposal_id=f"repair:{uuid4()}",
                    scope="function",
                    action="modify",
                    delta=(
                        f"Review the typed pipeline or adapter for {tool_name}; "
                        f"failure {reason!r} repeated {self.repair_threshold} times."
                    ),
                    rollback="keep the frozen kernel and reject the candidate version",
                    evidence_event_ids=tuple(self._failure_event_ids[key]),
                )
                self._repair_proposals.append(proposal)
                self._event(
                    "repair_proposed",
                    proposal_id=proposal.proposal_id,
                    scope=proposal.scope,
                    action=proposal.action,
                    evidence_event_ids=list(proposal.evidence_event_ids),
                )

    def normalize_plan(self, steps: Iterable[str]) -> list[str]:
        self._activate("selector", "planner", "deliberator", "recorder")
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_step in steps:
            step = " ".join(str(raw_step).split())
            key = step.casefold()
            if step and key not in seen:
                normalized.append(step)
                seen.add(key)
        if not normalized:
            normalized = ["[ANALYZE] Establish the task contract and evidence needed"]
        verification_markers = ("[verify]", "verify", "validate", "test", "evidence")
        if not any(
            marker in normalized[-1].casefold() for marker in verification_markers
        ):
            normalized.append(
                "[VERIFY] Validate evidence, the negative path, and rollback"
            )
        plan_id = f"runtime-plan:{uuid4()}"
        self._runtime_graph["entities"].append(
            {
                "id": plan_id,
                "type": "Plan",
                "properties": {"steps": list(normalized), "status": "ready"},
                "evidence": [{"source": "controller", "locator": plan_id}],
            }
        )
        if self._current_task_id:
            self._runtime_graph["relations"].append(
                {
                    "id": f"relation:{uuid4()}",
                    "type": "task_has_plan",
                    "source": self._current_task_id,
                    "target": plan_id,
                    "evidence": [{"source": "runtime", "locator": plan_id}],
                }
            )
        self._event("plan_normalized", plan_id=plan_id, step_count=len(normalized))
        return normalized

    def coordinate(self, executor_name: str, step_text: str) -> None:
        self._activate("coordinator", "recorder")
        self._event(
            "coordination",
            executor=executor_name,
            step_sha256=hashlib.sha256(step_text.encode("utf-8")).hexdigest(),
        )

    def evaluate_step(
        self,
        result: str,
        *,
        evidence_before: int,
        requires_evidence: bool = False,
    ) -> StepAssessment:
        self._activate("reflector", "controller", "recorder")
        evidence_added = max(0, self._evidence_count - evidence_before)
        lower = result.casefold()
        explicit_failure = any(
            marker in lower
            for marker in ("error:", "execution failed", "step failed", "traceback")
        )
        if explicit_failure:
            assessment = StepAssessment(
                complete=False,
                status="blocked",
                reason="executor_reported_failure",
                evidence_added=evidence_added,
            )
        elif requires_evidence and evidence_added == 0:
            assessment = StepAssessment(
                complete=False,
                status="blocked",
                reason="verification_without_new_evidence",
                evidence_added=0,
            )
        else:
            assessment = StepAssessment(
                complete=True,
                status="completed",
                reason=(
                    "evidenced_execution" if evidence_added else "executor_returned"
                ),
                evidence_added=evidence_added,
            )
        self._event(
            "step_assessed",
            complete=assessment.complete,
            status=assessment.status,
            reason=assessment.reason,
            evidence_added=evidence_added,
        )
        return assessment

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "kernel_sha256": self.kernel_report["kernel_sha256"],
            "assurance_level": self.kernel_report["assurance_level"],
            "active_taint_count": len(self._active_taint_ids),
            "evidence_count": self._evidence_count,
            "activated_patterns": sorted(self._activated_patterns),
            "repair_proposals": len(self._repair_proposals),
        }
