from __future__ import annotations

from dataclasses import replace

import pytest

from app.oak.formal_runtime import (
    BudgetExceeded,
    BudgetLimits,
    approval_key,
    approval_reference,
    verify_runtime_trace,
)
from app.oak.runtime import OakRuntime


def _limits(**overrides: int) -> BudgetLimits:
    values = {
        "steps": 3,
        "model_calls": 3,
        "tool_calls": 3,
        "retries": 1,
        "wall_time_ms": 60_000,
        "tokens": 10_000,
        "context_bytes": 100_000,
        "external_effects": 2,
    }
    values.update(overrides)
    return BudgetLimits(**values)


def test_runtime_trace_is_hash_chained_and_conforms_to_terminal_contract() -> None:
    runtime = OakRuntime(budget_limits=_limits())
    runtime.begin_task("produce one evidenced answer")
    runtime.consume_budget("steps", cycle_id="main")
    runtime.record_final_response("done")

    assert runtime.finish_task("completed") == "completed"

    report = runtime.conformance_report()
    assert report.valid, report.errors
    assert report.evidence_count == 1
    assert report.unresolved_effects == 0
    assert report.head_hash == runtime.trace[-1]["event_hash"]

    tampered = [dict(event) for event in runtime.trace]
    tampered[1] = {**tampered[1], "dimension": "tool_calls"}
    assert verify_runtime_trace(tampered).valid is False


def test_unknown_effect_refuses_false_completion() -> None:
    runtime = OakRuntime(budget_limits=_limits())
    runtime.begin_task("perform one governed effect")
    decision = runtime.authorize_tool(
        "python_execute",
        {"code": "print('effect')"},
        granted=True,
        tainted_sink_approved=True,
    )
    operation = runtime.begin_effect(
        operation_id="call-1",
        principal="Manus",
        tool_name="python_execute",
        arguments={"code": "print('effect')"},
        gate_event_id=decision.event_id,
    )
    runtime.dispatch_effect(operation.operation_id)
    runtime.observe_effect(operation.operation_id, "unknown")
    runtime.record_final_response("the result could not be confirmed")

    assert runtime.finish_task("completed") == "unknown_outcome"

    report = runtime.conformance_report()
    assert report.valid, report.errors
    assert report.unresolved_effects == 1
    assert report.terminal_status == "unknown_outcome"


def test_global_budget_covers_reentry_and_fails_visible() -> None:
    runtime = OakRuntime(budget_limits=_limits(steps=1))
    runtime.begin_task("bounded task")
    runtime.consume_budget("steps", cycle_id="main")

    with pytest.raises(BudgetExceeded, match="steps"):
        runtime.consume_budget("steps", cycle_id="callback")

    runtime.record_final_response("halted")
    assert runtime.finish_task("completed") == "budget_exhausted"
    report = runtime.conformance_report()
    assert report.valid, report.errors
    assert report.budget_exhausted is True


def test_exact_approval_is_bound_to_arguments_state_and_policy() -> None:
    arguments = {"recipient": "owner@example.invalid", "body": "hello"}
    reference = approval_reference("send_email", arguments)
    key = approval_key(
        principal="local-operator",
        tool_name="send_email",
        arguments=arguments,
        state_version="kernel-v1",
        policy_version="policy-v2",
    )

    assert reference.startswith("send_email:")
    assert key != approval_key(
        principal="local-operator",
        tool_name="send_email",
        arguments={**arguments, "recipient": "attacker@example.invalid"},
        state_version="kernel-v1",
        policy_version="policy-v2",
    )
    assert key != approval_key(
        principal="local-operator",
        tool_name="send_email",
        arguments=arguments,
        state_version="kernel-v2",
        policy_version="policy-v2",
    )


def test_empty_or_post_terminal_trace_is_not_vacuously_conformant() -> None:
    assert verify_runtime_trace(()).valid is False

    runtime = OakRuntime(budget_limits=_limits())
    runtime.begin_task("one evidenced result")
    runtime.record_final_response("done")
    runtime.finish_task("completed")
    extended = [*runtime.trace, dict(runtime.trace[-1])]

    report = verify_runtime_trace(extended)
    assert report.valid is False
    assert any("after the terminal" in error for error in report.errors)


def test_provider_must_be_transactionally_absent_before_completion() -> None:
    runtime = OakRuntime(budget_limits=_limits())
    runtime.begin_task("use one MCP provider")
    runtime.record_provider_state("crm", "opening")
    runtime.record_provider_state("crm", "active", published_tool_count=2)
    runtime.record_provider_state("crm", "draining")
    runtime.record_provider_state("crm", "absent")
    runtime.record_final_response("done")

    assert runtime.finish_task("completed") == "completed"
    assert runtime.conformance_report().valid is True

    leaked = OakRuntime(budget_limits=_limits())
    leaked.begin_task("leak one MCP provider")
    leaked.record_provider_state("crm", "opening")
    leaked.record_provider_state("crm", "active", published_tool_count=1)
    leaked.record_final_response("claimed done")

    assert leaked.finish_task("completed") == "failed_visible"
    assert leaked.conformance_report().valid is True


def test_malformed_provider_count_fails_closed_instead_of_crashing() -> None:
    runtime = OakRuntime(budget_limits=_limits())
    runtime.begin_task("reject malformed provider trace")
    runtime.record_provider_state("crm", "opening")
    runtime.record_final_response("stopped")
    runtime.finish_task("failed")
    malformed = [dict(event) for event in runtime.trace]
    malformed[1]["published_tool_count"] = "not-an-integer"

    report = verify_runtime_trace(malformed)

    assert report.valid is False
    assert any("invalid published tool count" in error for error in report.errors)


def test_multiple_failure_causes_remain_visible_under_one_terminal_status() -> None:
    runtime = OakRuntime(budget_limits=_limits(steps=0))
    runtime.begin_task("preserve multiple terminal causes")
    runtime.record_provider_state("crm", "opening")
    runtime.record_provider_state("crm", "active", published_tool_count=1)
    with pytest.raises(BudgetExceeded):
        runtime.consume_budget("steps", cycle_id="main")
    runtime.record_final_response("halted with visible failures")

    assert runtime.finish_task("completed") == "failed_visible"

    terminal = runtime.trace[-1]
    assert terminal["terminal_reasons"] == [
        "provider_not_absent",
        "budget_exhausted",
    ]
    report = runtime.conformance_report()
    assert report.valid is True
    assert report.budget_exhausted is True
    assert set(report.provider_states.values()) == {"active"}


def test_verifier_rejects_prestart_events_and_boolean_budget_numbers() -> None:
    runtime = OakRuntime(budget_limits=_limits())
    runtime.begin_task("strict verifier input types")
    runtime.consume_budget("steps", cycle_id="main")
    runtime.record_final_response("done")
    runtime.finish_task("completed")

    boolean_budget = [dict(event) for event in runtime.trace]
    boolean_budget[1]["delta"] = True
    report = verify_runtime_trace(boolean_budget)
    assert report.valid is False
    assert any("invalid budget delta" in error for error in report.errors)

    before_start = [dict(runtime.trace[2]), *runtime.trace]
    report = verify_runtime_trace(before_start)
    assert report.valid is False
    assert any("trace must start" in error for error in report.errors)
