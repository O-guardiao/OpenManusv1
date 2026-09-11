"""Local operator mailbox. No command can change tool grants.

Cooperative pause takes effect at the next checkpoint; cancel can stop async
I/O. This is not authentication against another process running as the user.
Steering delivery is at least once across a crash between append and ack.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from uuid import UUID, uuid4


class ControlInbox:
    def __init__(self, database: Path, session_id: str):
        if str(UUID(session_id)) != session_id:
            raise ValueError('Canonical session UUID required')
        database = Path(database).resolve()
        self.directory = database.parent / (database.name + '.controls') / session_id

    def send(self, action: str, text: str = '') -> str:
        if action not in {'steer', 'pause', 'continue', 'cancel'}:
            raise ValueError('Unknown control action')
        if not isinstance(text, str) or len(text.encode('utf-8')) > 32_000:
            raise ValueError('Control message exceeds 32000 bytes')
        if action == 'steer' and not text.strip():
            raise ValueError('Steering requires text')
        if action != 'steer' and text:
            raise ValueError('Only steering accepts text')
        document = json.dumps({'version': 1, 'action': action, 'text': text},
                              ensure_ascii=False)
        if len(document.encode('utf-8')) > 40_000:
            raise ValueError('Encoded control request exceeds 40000 bytes')
        self.directory.mkdir(parents=True, exist_ok=True)
        key = f'{time.time_ns():020d}-{uuid4()}'
        pending = self.directory / (key + '.tmp')
        target = self.directory / (key + '.json')
        with pending.open('x', encoding='utf-8') as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, target)
        return key

    def pending(self):
        for path in sorted(self.directory.glob('*.json')):
            if path.is_symlink() or path.stat().st_size > 40_000:
                raise ValueError('Invalid control file')
            document = json.loads(path.read_text(encoding='utf-8'))
            if (not isinstance(document, dict) or set(document) != {'version', 'action', 'text'}
                    or type(document.get('version')) is not int or document['version'] != 1
                    or document.get('action') not in {'steer', 'pause', 'continue', 'cancel'}
                    or not isinstance(document.get('text'), str)
                    or len(document['text'].encode('utf-8')) > 32_000):
                raise ValueError('Invalid control request')
            if document['action'] == 'steer' and not document['text'].strip():
                raise ValueError('Empty steering request')
            if document['action'] != 'steer' and document['text']:
                raise ValueError('Only steering accepts text')
            yield path, document

    def acknowledge(self, path: Path):
        if path.parent != self.directory:
            raise ValueError('Control path outside session')
        path.unlink()
