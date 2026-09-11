"""Bind conversation persistence and observation to the existing agent loop."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone

from app.schema import Memory, Message


class RunObserver:
    def __init__(self, memory: Memory, *, store=None, session_id=None,
                 event_path: Path | None = None, stream_output=False, inbox=None,
                 run_id: str | None = None):
        self.memory = memory
        self.store = store
        self.session_id = session_id
        self.run_id = run_id
        self.event_path = Path(event_path) if event_path else None
        self.stream_output = stream_output
        self.streamed_text = False
        self.inbox = inbox
        self.paused = False
        self.cancelled = False
        self._steering = []
        self._seen = set()
        self._observed_effects = {}
        if self.event_path:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **payload):
        event = {'version': 1, 'type': kind, 'session_id': self.session_id,
                 'run_id': self.run_id,
                 'at': datetime.now(timezone.utc).isoformat(), **payload}
        if self.event_path:
            with self.event_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                stream.flush()
        return event

    def on_message(self, message: Message):
        if self.store:
            self.store.append(self.session_id, message)
        # Keep the intent pending until the corresponding observation is durable.
        # A crash between these writes blocks resume conservatively; reversing
        # their order could hide a successful effect with a missing result.
        if message.role == 'tool' and message.tool_call_id in self._observed_effects:
            status = self._observed_effects.pop(message.tool_call_id)
            self.effect_outcome(message.tool_call_id, status)
        self.emit('message_recorded', role=message.role,
                  index=len(self.memory.messages), characters=len(message.content or ''))

    async def on_model_event(self, event: dict):
        # Text is deliberately included only in the explicitly selected stream log.
        self.emit(event['type'], **{k: v for k, v in event.items() if k != 'type'})
        if self.stream_output and event['type'] == 'model_text_delta':
            self.streamed_text = self.streamed_text or bool(event['text'])
            print(event['text'], end='', flush=True)

    def effect_intent(self, operation_id, tool, args_sha256):
        operation_id = f'{self.run_id}:{operation_id}' if self.run_id else operation_id
        if self.store:
            self.store.record_effect_intent(self.session_id, operation_id, tool, args_sha256)
        self.emit('effect_intent', operation_id=operation_id, tool=tool,
                  arguments_sha256=args_sha256)

    def effect_outcome(self, operation_id, status, *, defer_until_observation=False):
        if defer_until_observation:
            self._observed_effects[operation_id] = status
            return
        operation_id = f'{self.run_id}:{operation_id}' if self.run_id else operation_id
        if self.store:
            self.store.resolve_effect(self.session_id, operation_id, status)
        self.emit('effect_outcome', operation_id=operation_id, status=status)

    def poll_controls(self):
        if not self.inbox:
            return
        for path, command in self.inbox.pending():
            if path in self._seen:
                continue
            self._seen.add(path)
            action = command['action']
            if action == 'steer':
                self._steering.append((path, command['text']))
                continue
            if action == 'pause':
                self.paused = True
            elif action == 'continue':
                self.paused = False
            else:
                self.cancelled = True
            self.emit('control_received', action=action)
            self.inbox.acknowledge(path)

    async def checkpoint(self, *, apply_steering=True):
        while True:
            self.poll_controls()
            if self.cancelled:
                raise asyncio.CancelledError('Operator cancelled the run')
            if not self.paused:
                break
            await asyncio.sleep(.1)
        while apply_steering and self._steering:
            path, text = self._steering[0]
            self.memory.add_message(Message.user_message(text))
            self.inbox.acknowledge(path)
            self._steering.pop(0)
            self.emit('steering_applied')

    async def supervise(self, coroutine):
        work = asyncio.create_task(coroutine)
        try:
            while not work.done():
                self.poll_controls()
                if self.cancelled:
                    work.cancel()
                await asyncio.wait({work}, timeout=.1)
            return await work
        finally:
            if not work.done():
                work.cancel()
                try:
                    await work
                except asyncio.CancelledError:
                    pass

    def repair_incomplete_turn(self):
        """Only used after the persistent pending-effect gate has passed.

        Missing observations are explicit failures; no tool is invoked here.
        A message may have been persisted before a tool's intent was admitted.
        """
        pending = {}
        for message in self.memory.messages:
            for call in message.tool_calls or []:
                pending[call.id] = call.function.name
            if message.role == 'tool':
                pending.pop(message.tool_call_id, None)
        for call_id, name in pending.items():
            self.memory.add_message(Message.tool_message(
                'Previous run interrupted before a durable tool observation. '
                'No successful result is available; inspect current state before acting.',
                name=name, tool_call_id=call_id,
            ))
        if pending:
            self.emit('incomplete_turn_repaired', count=len(pending))
