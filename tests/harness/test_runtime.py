import asyncio
import json

import pytest

from app.harness.control import ControlInbox
from app.harness.runtime import RunObserver
from app.harness.session import SessionRecoveryRequired, SessionStore
from app.schema import Function, Memory, Message, ToolCall


def test_observer_persists_before_memory_and_reopens(tmp_path):
    store = SessionStore(tmp_path / 'app.db')
    sid = store.create()
    with store.lease(sid):
        store.begin_run(sid, 'run-1')
        memory = Memory()
        observer = RunObserver(memory, store=store, session_id=sid)
        memory.set_listener(observer.on_message)
        memory.add_message(Message.user_message('Manter requisito inicial'))
        observer.effect_intent('op-1', 'python_execute', 'a' * 64)
        observer.effect_outcome('op-1', 'unknown')
        store.finish_run(sid, 'interrupted')
    assert SessionStore(tmp_path / 'app.db').load(sid)[0].content == 'Manter requisito inicial'
    with store.lease(sid), pytest.raises(SessionRecoveryRequired):
        store.begin_run(sid, 'run-2')


@pytest.mark.asyncio
async def test_controls_pause_steer_resume_and_cancel(tmp_path):
    store = SessionStore(tmp_path / 'app.db')
    sid = store.create()
    inbox = ControlInbox(store.database_path, sid)
    memory = Memory()
    observer = RunObserver(memory, inbox=inbox)
    inbox.send('pause')
    waiting = asyncio.create_task(observer.checkpoint())
    await asyncio.sleep(.05)
    assert not waiting.done()
    inbox.send('steer', 'Preservar o escopo completo')
    inbox.send('continue')
    await asyncio.wait_for(waiting, 2)
    assert memory.messages[-1].content == 'Preservar o escopo completo'
    inbox.send('cancel')
    with pytest.raises(asyncio.CancelledError):
        await observer.checkpoint()


@pytest.mark.asyncio
async def test_control_cancels_cooperative_inflight_work(tmp_path):
    sid = SessionStore(tmp_path / 'app.db').create()
    inbox = ControlInbox(tmp_path / 'app.db', sid)
    observer = RunObserver(Memory(), inbox=inbox)
    cleaned = asyncio.Event()
    async def slow():
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.set()
    work = asyncio.create_task(observer.supervise(slow()))
    await asyncio.sleep(.05)
    inbox.send('cancel')
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(work, 2)
    assert cleaned.is_set()


def test_repair_incomplete_read_turn_does_not_reexecute(tmp_path):
    memory = Memory(messages=[Message.from_tool_calls(tool_calls=[
        ToolCall(id='read-1', function=Function(name='read', arguments='{}'))
    ])])
    observer = RunObserver(memory)
    observer.repair_incomplete_turn()
    assert memory.messages[-1].tool_call_id == 'read-1'
    assert 'interrupted' in memory.messages[-1].content


def test_event_log_does_not_copy_message_content(tmp_path):
    events = tmp_path / 'events.jsonl'
    observer = RunObserver(Memory(), event_path=events)
    observer.on_message(Message.user_message('sensitive prompt'))
    assert 'sensitive prompt' not in events.read_text()
    assert json.loads(events.read_text())['type'] == 'message_recorded'


def test_control_rejects_path_escape_and_unknown_command(tmp_path):
    with pytest.raises(ValueError):
        ControlInbox(tmp_path / 'app.db', '../elsewhere')
    sid = SessionStore(tmp_path / 'app.db').create()
    with pytest.raises(ValueError):
        ControlInbox(tmp_path / 'app.db', sid).send('grant', 'all')


def test_effect_stays_pending_until_tool_observation_is_persisted(tmp_path):
    store = SessionStore(tmp_path / 'app.db')
    sid = store.create()
    with store.lease(sid):
        store.begin_run(sid, 'run-1')
        memory = Memory()
        observer = RunObserver(memory, store=store, session_id=sid, run_id='run-1')
        memory.set_listener(observer.on_message)
        observer.effect_intent('op-1', 'python_execute', 'a' * 64)
        observer.effect_outcome('op-1', 'succeeded', defer_until_observation=True)
        assert store.pending_effects(sid)[0]['operation_id'] == 'run-1:op-1'
        memory.add_message(Message.tool_message('resultado', name='python_execute', tool_call_id='op-1'))
        assert store.pending_effects(sid) == []
        assert store.load(sid)[0].content == 'resultado'
        store.finish_run(sid, 'completed')


def test_control_unicode_accepted_by_sender_is_readable(tmp_path):
    sid = SessionStore(tmp_path / 'app.db').create()
    inbox = ControlInbox(tmp_path / 'app.db', sid)
    text = '\u00e7' * 15000
    inbox.send('steer', text)
    assert next(inbox.pending())[1]['text'] == text
    with pytest.raises(ValueError, match='Encoded control'):
        inbox.send('steer', '\u0000' * 15000 + 'a')
