from types import SimpleNamespace

import pytest

from app.agent.toolcall import ToolCallAgent
from app.harness.control import ControlInbox
from app.harness.runtime import RunObserver
from app.harness.session import SessionStore
from app.llm import LLM
from app.schema import Memory, Message
from app.tool.policy import ToolPolicy


@pytest.mark.asyncio
async def test_final_text_on_last_step_is_completed_once(tmp_path, monkeypatch):
    calls = []
    async def reply(self, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content='Resultado pronto.', tool_calls=[])
    monkeypatch.setattr(LLM, 'ask_tool', reply)
    agent = ToolCallAgent(llm=object.__new__(LLM), max_steps=1,
        tool_policy=ToolPolicy.guarded(audit_log_path=tmp_path / 'audit.jsonl'))
    result = await agent.run('responda uma vez')
    assert result == 'Step 1: Resultado pronto.'
    assert len(calls) == 1
    assert agent.oak_runtime.last_task_status == 'completed'


@pytest.mark.asyncio
async def test_loop_projects_context_without_deleting_transcript(tmp_path, monkeypatch):
    captured = []
    async def reply(self, **kwargs):
        captured.extend(kwargs['messages'])
        return SimpleNamespace(content='Fim.', tool_calls=[])
    monkeypatch.setattr(LLM, 'ask_tool', reply)
    memory = Memory(messages=[Message.user_message('REQUISITO INICIAL'),
        *[Message.assistant_message('x' * 6000) for _ in range(80)]])
    agent = ToolCallAgent(llm=object.__new__(LLM), memory=memory,
        max_steps=1, context_window_tokens=10000,
        tool_policy=ToolPolicy.guarded(audit_log_path=tmp_path / 'audit.jsonl'))
    await agent.run('CORRECAO DO USUARIO')
    assert len(memory.messages) == 83
    assert sum(len(m.content or '') for m in captured) < 10000
    assert any(m.content == 'REQUISITO INICIAL' for m in captured)
    assert any(m.content == 'CORRECAO DO USUARIO' for m in captured)


@pytest.mark.asyncio
async def test_tool_checkpoint_defers_steering_until_complete_turn(tmp_path):
    sid = SessionStore(tmp_path / 'app.db').create()
    inbox = ControlInbox(tmp_path / 'app.db', sid)
    memory = Memory()
    observer = RunObserver(memory, inbox=inbox)
    inbox.send('steer', 'nova instrucao')
    await observer.checkpoint(apply_steering=False)
    assert memory.messages == []
    await observer.checkpoint()
    assert memory.messages[0].content == 'nova instrucao'
