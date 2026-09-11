from __future__ import annotations

import asyncio
import json

from app.cockpit.session import AgentRunRecorder
from app.cockpit.store import CockpitStore
from app.oak.runtime import OakRuntime


def _finished_runtime(prompt: str, result: str, status: str) -> OakRuntime:
    runtime = OakRuntime()
    runtime.begin_task(prompt)
    runtime.record_final_response(result)
    runtime.finish_task(status)
    return runtime


def test_agent_receipt_is_verifiable_and_does_not_persist_prompt_or_result(tmp_path):
    database = tmp_path / "cockpit.db"
    store = CockpitStore(database)
    prompt = "PRIVATE PROMPT CONTENT"
    result_text = "PRIVATE RESULT CONTENT"
    recorder, created = AgentRunRecorder.admit(
        store,
        prompt=prompt,
        agent_name="Manus",
        allowed_tools=["python_execute"],
        allowed_tainted_tools=[],
        request_key="agent-run-v1",
    )
    assert created is True

    recorder.start()
    runtime = _finished_runtime(prompt, result_text, "completed")
    task = recorder.finish(
        oak_trace=runtime.trace,
        oak_evidence_count=runtime.evidence_count,
        terminal_status="completed",
        result=result_text,
        duration_ms=12,
    )

    exported = store.export_task(task["id"])
    encoded = json.dumps(exported)
    assert task["status"] == "completed"
    assert store.verify_task(task["id"]).valid is True
    assert prompt not in encoded
    assert result_text not in encoded
    assert "agent-run-v1" not in encoded
    evidence = next(event for event in exported["events"] if event["kind"] == "evidence.recorded")
    assert evidence["payload"]["evidence_type"] == "formal.runtime.conformance.v1"
    assert evidence["payload"]["value"]["result_sha256"]
    assert evidence["payload"]["value"]["oak_trace_sha256"]
    assert evidence["payload"]["value"]["formal_conformance"]["valid"] is True
    assert evidence["payload"]["value"]["oak_trace"]

    duplicate, duplicate_created = AgentRunRecorder.admit(
        store,
        prompt=prompt,
        agent_name="Manus",
        allowed_tools=["python_execute"],
        allowed_tainted_tools=[],
        request_key="agent-run-v1",
    )
    assert duplicate_created is False
    assert duplicate.task_id == recorder.task_id


def test_non_successful_agent_terminal_status_is_not_relabelled_completed(tmp_path):
    store = CockpitStore(tmp_path / "cockpit.db")
    recorder, _ = AgentRunRecorder.admit(
        store,
        prompt="bounded task",
        agent_name="Manus",
        allowed_tools=[],
        allowed_tainted_tools=[],
        request_key="agent-max-steps-v1",
    )
    recorder.start()
    result = "Terminated: Reached max steps"
    runtime = _finished_runtime("bounded task", result, "max_steps_reached")
    task = recorder.finish(
        oak_trace=runtime.trace,
        oak_evidence_count=runtime.evidence_count,
        terminal_status="max_steps_reached",
        result=result,
        duration_ms=8,
    )

    assert task["status"] == "failed"
    assert store.verify_task(task["id"]).valid is True


def test_main_records_a_real_cli_lifecycle_without_storing_content(tmp_path, monkeypatch):
    import main as main_module

    runtime = _finished_runtime(
        "DO NOT STORE THIS PROMPT",
        "DO NOT STORE THIS RESULT",
        "completed",
    )

    class FakeAgent:
        name = "Manus"
        oak_runtime = runtime

        async def run(self, prompt):
            assert prompt == "DO NOT STORE THIS PROMPT"
            return "DO NOT STORE THIS RESULT"

        async def cleanup(self):
            return None

    async def fake_create(**kwargs):
        assert kwargs["max_steps"] == 0
        assert isinstance(kwargs["llm"], main_module.LLM)
        return FakeAgent()

    monkeypatch.setattr(main_module.Manus, "create", fake_create)
    database = tmp_path / "agent.db"
    audit_log = tmp_path / "audit.jsonl"
    exit_code = asyncio.run(
        main_module.main(
            [
                "--prompt",
                "DO NOT STORE THIS PROMPT",
                "--audit-log",
                str(audit_log),
                "--cockpit-db",
                str(database),
                "--cockpit-request-key",
                "main-fixture-v1",
                "--max-steps",
                "0",
            ]
        )
    )
    assert exit_code == 0

    store = CockpitStore(database)
    tasks = store.list_tasks()
    assert len(tasks) == 1
    exported = store.export_task(tasks[0]["id"])
    encoded = json.dumps(exported)
    assert tasks[0]["status"] == "completed"
    assert "DO NOT STORE THIS PROMPT" not in encoded
    assert "DO NOT STORE THIS RESULT" not in encoded


def test_main_returns_failure_for_max_steps_and_receipt_stays_valid(tmp_path, monkeypatch):
    import main as main_module

    result = "Terminated: Reached max steps (0)"
    runtime = _finished_runtime("bounded failure", result, "max_steps_reached")

    class MaxStepsAgent:
        name = "Manus"
        oak_runtime = runtime

        async def run(self, prompt):
            return result

        async def cleanup(self):
            return None

    async def fake_create(**kwargs):
        return MaxStepsAgent()

    monkeypatch.setattr(main_module.Manus, "create", fake_create)
    database = tmp_path / "agent.db"
    exit_code = asyncio.run(
        main_module.main(
            [
                "--prompt",
                "bounded failure",
                "--audit-log",
                str(tmp_path / "audit.jsonl"),
                "--cockpit-db",
                str(database),
                "--cockpit-request-key",
                "main-max-steps-v1",
                "--max-steps",
                "0",
            ]
        )
    )

    task = CockpitStore(database).list_tasks()[0]
    assert exit_code == 1
    assert task["status"] == "failed"
    assert CockpitStore(database).verify_task(task["id"]).valid is True


def test_invalid_runtime_trace_can_never_complete_cockpit_task(tmp_path):
    store = CockpitStore(tmp_path / "invalid-runtime.db")
    recorder, _ = AgentRunRecorder.admit(
        store,
        prompt="task",
        agent_name="Manus",
        allowed_tools=[],
        allowed_tainted_tools=[],
        request_key="invalid-runtime-v1",
    )
    recorder.start()

    task = recorder.finish(
        oak_trace=[],
        oak_evidence_count=0,
        terminal_status="completed",
        result="claim without a run",
        duration_ms=1,
    )

    assert task["status"] == "failed"


def test_main_seals_failure_when_cleanup_fails_after_a_completed_run(
    tmp_path, monkeypatch
):
    import main as main_module

    prompt = "cleanup-sensitive task"
    result = "run returned before cleanup"
    runtime = _finished_runtime(prompt, result, "completed")

    class CleanupFailingAgent:
        name = "Manus"
        oak_runtime = runtime

        async def run(self, received_prompt):
            assert received_prompt == prompt
            return result

        async def cleanup(self):
            raise RuntimeError("fixture cleanup failed")

    async def fake_create(**_kwargs):
        return CleanupFailingAgent()

    monkeypatch.setattr(main_module.Manus, "create", fake_create)
    database = tmp_path / "cleanup-failure.db"
    exit_code = asyncio.run(
        main_module.main(
            [
                "--prompt",
                prompt,
                "--audit-log",
                str(tmp_path / "cleanup-audit.jsonl"),
                "--cockpit-db",
                str(database),
                "--cockpit-request-key",
                "cleanup-failure-v1",
                "--max-steps",
                "0",
            ]
        )
    )

    store = CockpitStore(database)
    task = store.list_tasks()[0]
    exported = store.export_task(task["id"])
    receipt = next(
        event["payload"]["value"]
        for event in exported["events"]
        if event["kind"] == "evidence.recorded"
    )
    assert exit_code == 1
    assert task["status"] == "failed"
    assert receipt["terminal_status"] == "failed_cleanup"
    assert receipt["formal_conformance"]["valid"] is True
    assert receipt["terminal_status_agrees"] is False
