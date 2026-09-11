"""Independent causal regression oracles for the harness integration review."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.agent.toolcall import ToolCallAgent
from app.harness.control import ControlInbox
from app.harness.runtime import RunObserver
from app.harness.session import SessionStore
from app.llm import LLM
from app.schema import Function, Memory, ToolCall
from app.tool.policy import ToolPolicy
from app.tool.python_execute import PythonExecute
from app.tool.terminate import Terminate
from app.tool.tool_collection import ToolCollection


@pytest.mark.asyncio
async def test_failed_python_dict_keeps_effect_unresolved_after_partial_write(tmp_path, monkeypatch):
    """A legacy timeout result must not certify a partially performed effect."""
    side_effect = tmp_path / "partial-effect.txt"

    async def partial_then_timeout(self, code, timeout=5):
        side_effect.write_text("partially changed", encoding="utf-8")
        return {"success": False, "observation": "Execution timeout after 5 seconds"}

    calls = []

    async def reply(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(content="", tool_calls=[ToolCall(
                id="python-op", function=Function(name="python_execute", arguments=json.dumps({"code": "pass"}))
            )])
        return SimpleNamespace(content="Finished", tool_calls=[])

    monkeypatch.setattr(PythonExecute, "execute", partial_then_timeout)
    monkeypatch.setattr(LLM, "ask_tool", reply)
    store = SessionStore(tmp_path / "session.db")
    sid = store.create()
    with store.lease(sid):
        store.begin_run(sid, "run-1")
        memory = Memory()
        observer = RunObserver(memory, store=store, session_id=sid, run_id="run-1")
        memory.set_listener(observer.on_message)
        agent = ToolCallAgent(llm=object.__new__(LLM), memory=memory, max_steps=2,
            run_observer=observer, available_tools=ToolCollection(PythonExecute()),
            tool_policy=ToolPolicy.guarded(allowed_tools={"python_execute"}, audit_log_path=tmp_path / "audit.jsonl"))
        await agent.run("Perform the local operation")
        assert side_effect.exists()
        assert any(message.role == "tool" for message in store.load(sid))
        assert store.pending_effects(sid), "Partial timeout was incorrectly resolved as succeeded"


@pytest.mark.asyncio
async def test_steering_arriving_during_terminate_is_not_ignored(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "session.db")
    sid = store.create()
    inbox = ControlInbox(store.database_path, sid)

    async def terminate_with_incoming_steer(self, status):
        inbox.send("steer", "Correction: include the missing license")
        return "Terminated successfully"

    async def reply(self, **kwargs):
        return SimpleNamespace(content="", tool_calls=[ToolCall(
            id="stop-op", function=Function(name="terminate", arguments='{"status":"success"}')
        )])

    monkeypatch.setattr(Terminate, "execute", terminate_with_incoming_steer)
    monkeypatch.setattr(LLM, "ask_tool", reply)
    memory = Memory()
    observer = RunObserver(memory, inbox=inbox)
    agent = ToolCallAgent(llm=object.__new__(LLM), memory=memory, max_steps=1,
        run_observer=observer, available_tools=ToolCollection(Terminate()),
        tool_policy=ToolPolicy.guarded(audit_log_path=tmp_path / "audit.jsonl"))
    await observer.supervise(agent.run("Complete the task"))
    assert agent.oak_runtime.last_task_status != "completed", "Run completed with unapplied steering"


@pytest.mark.asyncio
async def test_explicit_terminate_failure_is_not_completed(tmp_path, monkeypatch):
    async def reply(self, **kwargs):
        return SimpleNamespace(content="", tool_calls=[ToolCall(
            id="stop-failed", function=Function(name="terminate", arguments='{"status":"failure"}')
        )])

    monkeypatch.setattr(LLM, "ask_tool", reply)
    agent = ToolCallAgent(llm=object.__new__(LLM), max_steps=1,
        available_tools=ToolCollection(Terminate()),
        tool_policy=ToolPolicy.guarded(audit_log_path=tmp_path / "audit.jsonl"))
    await agent.run("Attempt the task")
    assert agent.oak_runtime.last_task_status != "completed", "Explicit failure was reported as completed"


@pytest.mark.asyncio
async def test_cancel_after_dispatch_preserves_durable_pending_effect(tmp_path, monkeypatch):
    started = asyncio.Event()

    async def blocked_effect(self, code, timeout=5):
        started.set()
        await asyncio.Event().wait()

    async def reply(self, **kwargs):
        return SimpleNamespace(content="", tool_calls=[ToolCall(
            id="pending-op", function=Function(name="python_execute", arguments='{"code":"pass"}')
        )])

    monkeypatch.setattr(PythonExecute, "execute", blocked_effect)
    monkeypatch.setattr(LLM, "ask_tool", reply)
    store = SessionStore(tmp_path / "session.db")
    sid = store.create()
    inbox = ControlInbox(store.database_path, sid)
    with store.lease(sid):
        store.begin_run(sid, "cancel-run")
        memory = Memory()
        observer = RunObserver(memory, store=store, session_id=sid, run_id="cancel-run", inbox=inbox)
        memory.set_listener(observer.on_message)
        agent = ToolCallAgent(llm=object.__new__(LLM), memory=memory, max_steps=2,
            run_observer=observer, available_tools=ToolCollection(PythonExecute()),
            tool_policy=ToolPolicy.guarded(allowed_tools={"python_execute"}, audit_log_path=tmp_path / "audit.jsonl"))
        task = asyncio.create_task(observer.supervise(agent.run("Perform operation")))
        await asyncio.wait_for(started.wait(), timeout=3)
        inbox.send("cancel")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        assert store.pending_effects(sid)[0]["operation_id"] == "cancel-run:pending-op"
        assert not any(message.role == "tool" for message in store.load(sid))


@pytest.mark.asyncio
async def test_literal_limit_text_in_final_answer_is_not_execution_status(tmp_path, monkeypatch):
    async def reply(self, **kwargs):
        return SimpleNamespace(content="The old log said: Terminated: Reached max steps. Fixed now.", tool_calls=[])

    monkeypatch.setattr(LLM, "ask_tool", reply)
    agent = ToolCallAgent(llm=object.__new__(LLM), max_steps=1,
        tool_policy=ToolPolicy.guarded(audit_log_path=tmp_path / "audit.jsonl"))
    await agent.run("Explain the old log message")
    assert agent.oak_runtime.last_task_status == "completed"
