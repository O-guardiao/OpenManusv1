from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.flow.planning import PlanningFlow
from app.oak.runtime import OakRuntime
from app.tool.planning import PlanningTool


class PlanLLM:
    async def ask_tool(self, **kwargs):
        return SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="planning",
                        arguments=(
                            '{"command":"create","title":"Task",'
                            '"steps":["Inspect evidence","Make change"]}'
                        ),
                    )
                )
            ]
        )


def flow_with(executor) -> PlanningFlow:
    return PlanningFlow.model_construct(
        llm=PlanLLM(),
        planning_tool=PlanningTool(),
        agents={"manus": executor},
        executor_keys=["manus"],
        primary_agent_key="manus",
        active_plan_id="plan-oak-test",
        current_step_index=None,
        oak_runtime=executor.oak_runtime,
    )


@pytest.mark.asyncio
async def test_oak_adds_terminal_verification_gate_to_model_plan() -> None:
    executor = SimpleNamespace(
        name="manus",
        description="executor",
        oak_runtime=OakRuntime(),
    )
    flow = flow_with(executor)

    await flow._create_initial_plan("Change a file")

    steps = flow.planning_tool.plans[flow.active_plan_id]["steps"]
    assert steps[-1] == "[VERIFY] Validate evidence, the negative path, and rollback"
    assert "planner" in flow.oak_runtime.activated_patterns


@pytest.mark.asyncio
async def test_failed_step_is_blocked_instead_of_marked_complete() -> None:
    class FailingExecutor:
        name = "manus"
        description = "executor"
        oak_runtime = OakRuntime()

        async def run(self, prompt: str) -> str:
            return "Error: execution failed"

    executor = FailingExecutor()
    flow = flow_with(executor)
    await flow.planning_tool.execute(
        command="create",
        plan_id=flow.active_plan_id,
        title="Task",
        steps=["Make change"],
    )
    flow.current_step_index = 0

    await flow._execute_step(executor, {"text": "Make change"})

    plan = flow.planning_tool.plans[flow.active_plan_id]
    assert plan["step_statuses"] == ["blocked"]
    assert plan["step_notes"] == ["executor_reported_failure"]


@pytest.mark.asyncio
async def test_verification_step_without_new_evidence_is_blocked() -> None:
    class ClaimOnlyExecutor:
        name = "manus"
        description = "executor"
        oak_runtime = OakRuntime()

        async def run(self, prompt: str) -> str:
            return "Everything passed."

    executor = ClaimOnlyExecutor()
    flow = flow_with(executor)
    await flow.planning_tool.execute(
        command="create",
        plan_id=flow.active_plan_id,
        title="Task",
        steps=["[VERIFY] Run tests and collect evidence"],
    )
    flow.current_step_index = 0

    await flow._execute_step(
        executor,
        {"text": "[VERIFY] Run tests and collect evidence"},
    )

    plan = flow.planning_tool.plans[flow.active_plan_id]
    assert plan["step_statuses"] == ["blocked"]
    assert plan["step_notes"] == ["verification_without_new_evidence"]


@pytest.mark.asyncio
async def test_blocked_plan_is_not_rewritten_as_completed_by_hidden_llm() -> None:
    executor = SimpleNamespace(
        name="manus",
        description="executor",
        oak_runtime=OakRuntime(),
    )
    flow = flow_with(executor)
    await flow.planning_tool.execute(
        command="create",
        plan_id=flow.active_plan_id,
        title="Task",
        steps=["Make change"],
    )
    await flow.planning_tool.execute(
        command="mark_step",
        plan_id=flow.active_plan_id,
        step_index=0,
        step_status="blocked",
        step_notes="negative path failed",
    )

    result = await flow._finalize_plan()

    assert result.startswith("Plan blocked:")
    assert "Plan completed:" not in result
