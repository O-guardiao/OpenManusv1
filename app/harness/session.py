"""Legacy Manus conversation persistence in the cockpit's SQLite database.

Transcripts, images, tool arguments and exports contain sensitive user content.
This module does not capture environment variables, credentials or tool grants.
Backups copy the whole shared database; protect them like the live database.
OS leases coordinate cooperating writers, not hostile local processes. A crash
releases the OS lock; unresolved effects require explicit reconciliation, never
an automatic retry. Restoring creates a different database, never overwrites one.
"""

from __future__ import annotations

from contextlib import contextmanager, closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Iterator
from uuid import UUID, uuid4

from app.schema import Message


class SessionBusy(RuntimeError):
    """Another writer owns the conversation, or a run is already active."""


class SessionRecoveryRequired(RuntimeError):
    """An effect has no trustworthy terminal observation; retry is forbidden."""


class SessionSchemaError(RuntimeError):
    """The conversation schema is unsupported or incomplete."""


_VERSION = 1
_PENDING = {"pending", "unknown", "partial"}
_RESOLVED = {"observed", "failed", "not_dispatched", "succeeded", "failed_before_effect"}
_REGISTRY_LOCK = threading.RLock()
_OWNERS: dict[tuple[str, str], tuple[object, int]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_document(message: Message) -> dict:
    document = message.model_dump(mode="json")
    if message.provenance is not None:
        document["provenance"] = message.provenance.model_dump(mode="json")
    return document


class SessionStore:
    """Append-only conversation records; hold lease() for a complete agent run."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database_path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def _initialize(self) -> None:
        with self._transaction() as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversation_schema'"
            ).fetchone()
            if exists:
                versions = db.execute("SELECT version FROM conversation_schema").fetchall()
                if len(versions) != 1 or versions[0][0] != _VERSION:
                    raise SessionSchemaError("Unsupported conversation schema version")
                return
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name GLOB 'conversation_*'"
            ).fetchone():
                raise SessionSchemaError("Conversation tables exist without a version")
            statements = [
                "CREATE TABLE conversation_schema (version INTEGER PRIMARY KEY)",
                "INSERT INTO conversation_schema VALUES (1)",
                """CREATE TABLE conversation_sessions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    status TEXT NOT NULL, active_run_id TEXT, last_run_id TEXT)""",
                """CREATE TABLE conversation_messages (
                    session_id TEXT NOT NULL REFERENCES conversation_sessions(id),
                    sequence INTEGER NOT NULL, message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, PRIMARY KEY(session_id, sequence))""",
                """CREATE TABLE conversation_effects (
                    session_id TEXT NOT NULL REFERENCES conversation_sessions(id),
                    operation_id TEXT NOT NULL, run_id TEXT NOT NULL, tool TEXT NOT NULL,
                    args_sha256 TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(session_id, operation_id))""",
                """CREATE TRIGGER conversation_messages_no_update BEFORE UPDATE
                    ON conversation_messages BEGIN
                    SELECT RAISE(ABORT, 'conversation transcript is append-only'); END""",
                """CREATE TRIGGER conversation_messages_no_delete BEFORE DELETE
                    ON conversation_messages BEGIN
                    SELECT RAISE(ABORT, 'conversation transcript is append-only'); END""",
            ]
            for statement in statements:
                db.execute(statement)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")

    def create(self) -> str:
        session_id = str(uuid4())
        with self._transaction() as db:
            db.execute(
                "INSERT INTO conversation_sessions VALUES (?, ?, ?, 'idle', NULL, NULL)",
                (session_id, _now(), _now()),
            )
        return session_id

    def _key(self, session_id: str) -> tuple[str, str]:
        if str(UUID(session_id)) != session_id:
            raise ValueError("session_id must be a canonical UUID")
        return os.path.normcase(str(self.database_path)), session_id

    def _owned(self, session_id: str) -> bool:
        with _REGISTRY_LOCK:
            return _OWNERS.get(self._key(session_id)) == (self, threading.get_ident())

    def _require_lease(self, session_id: str) -> None:
        if not self._owned(session_id):
            raise SessionBusy("Hold SessionStore.lease(session_id) while running or resolving effects")

    @contextmanager
    def lease(self, session_id: str) -> Iterator[None]:
        key = self._key(session_id)
        self.get_session(session_id)
        if self._owned(session_id):
            yield
            return
        locks = self.database_path.parent / (self.database_path.name + ".conversation-locks")
        locks.mkdir(exist_ok=True)
        stream = None
        acquired = False
        with _REGISTRY_LOCK:
            if key in _OWNERS:
                raise SessionBusy("Conversation already has an active writer")
            _OWNERS[key] = (self, threading.get_ident())
        try:
            stream = (locks / (session_id + ".lock")).open("a+b")
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as error:
                raise SessionBusy("Conversation is locked by another process") from error
            self._interrupt_unfinished(session_id)
            try:
                yield
            finally:
                self._interrupt_unfinished(session_id)
        finally:
            if stream is not None:
                try:
                    if acquired:
                        stream.seek(0)
                        if os.name == "nt":
                            import msvcrt
                            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                finally:
                    stream.close()
            with _REGISTRY_LOCK:
                _OWNERS.pop(key, None)

    def _interrupt_unfinished(self, session_id: str) -> None:
        with self._transaction() as db:
            db.execute(
                """UPDATE conversation_sessions SET status='interrupted',
                    active_run_id=NULL, updated_at=? WHERE id=? AND status='running'""",
                (_now(), session_id),
            )

    def get_session(self, session_id: str) -> dict:
        self._key(session_id)
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM conversation_sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            result = dict(row)
            result["message_count"] = db.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            result["pending_effect_count"] = db.execute(
                "SELECT COUNT(*) FROM conversation_effects WHERE session_id=? AND status IN ('pending','unknown','partial')",
                (session_id,),
            ).fetchone()[0]
            result["recovery_required"] = result["pending_effect_count"] > 0
            result["incomplete"] = result["status"] in {"running", "interrupted"}
            return result

    def list_sessions(self) -> list[dict]:
        with closing(self._connect()) as db:
            ids = db.execute("SELECT id FROM conversation_sessions ORDER BY updated_at DESC").fetchall()
        return [self.get_session(row[0]) for row in ids]

    def load(self, session_id: str) -> list[Message]:
        self.get_session(session_id)
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT message_json FROM conversation_messages WHERE session_id=? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [Message.model_validate_json(row[0]) for row in rows]

    def append(self, session_id: str, message: Message) -> None:
        document = json.dumps(_message_document(message), ensure_ascii=False)
        with self.lease(session_id):
            with self._transaction() as db:
                db.execute(
                    """INSERT INTO conversation_messages SELECT ?,
                        COALESCE(MAX(sequence), 0)+1, ?, ? FROM conversation_messages
                        WHERE session_id=?""", (session_id, document, _now(), session_id),
                )
                db.execute("UPDATE conversation_sessions SET updated_at=? WHERE id=?", (_now(), session_id))

    def begin_run(self, session_id: str, run_id: str) -> None:
        self._require_lease(session_id)
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        with self._transaction() as db:
            row = db.execute("SELECT active_run_id FROM conversation_sessions WHERE id=?", (session_id,)).fetchone()
            if row[0] is not None:
                raise SessionBusy("A run is already active")
            if self.pending_effects(session_id):
                raise SessionRecoveryRequired("Reconcile pending effects before starting another run")
            db.execute(
                "UPDATE conversation_sessions SET status='running', active_run_id=?, last_run_id=?, updated_at=? WHERE id=?",
                (run_id, run_id, _now(), session_id),
            )

    def finish_run(self, session_id: str, status: str) -> None:
        self._require_lease(session_id)
        if not status.strip() or status in {"running", "idle"}:
            raise ValueError("A terminal run status is required")
        if status == "completed" and self.pending_effects(session_id):
            raise SessionRecoveryRequired("Cannot complete with an unresolved effect")
        with self._transaction() as db:
            changed = db.execute(
                "UPDATE conversation_sessions SET status=?, active_run_id=NULL, updated_at=? WHERE id=? AND active_run_id IS NOT NULL",
                (status, _now(), session_id),
            ).rowcount
            if changed != 1:
                raise SessionBusy("No active run to finish")

    def record_effect_intent(self, session_id: str, operation_id: str, tool: str, args_sha256: str) -> None:
        self._require_lease(session_id)
        if not operation_id.strip() or not tool.strip():
            raise ValueError("operation_id and tool must not be empty")
        if len(args_sha256) != 64 or any(c not in "0123456789abcdef" for c in args_sha256):
            raise ValueError("args_sha256 must be a hexadecimal SHA256")
        with self._transaction() as db:
            row = db.execute("SELECT active_run_id FROM conversation_sessions WHERE id=?", (session_id,)).fetchone()
            if row[0] is None:
                raise SessionBusy("An active run is required before effect intent")
            if db.execute("SELECT 1 FROM conversation_effects WHERE session_id=? AND operation_id=?", (session_id, operation_id)).fetchone():
                raise SessionRecoveryRequired("Operation ID already recorded; do not dispatch again")
            db.execute(
                "INSERT INTO conversation_effects VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (session_id, operation_id, row[0], tool, args_sha256, _now(), _now()),
            )

    def resolve_effect(self, session_id: str, operation_id: str, status: str) -> None:
        """Record a caller's verified observation; this method does not reconcile remotely."""
        self._require_lease(session_id)
        if status not in (_PENDING - {"pending"}) | _RESOLVED:
            raise ValueError("Unsupported effect outcome")
        with self._transaction() as db:
            row = db.execute("SELECT status FROM conversation_effects WHERE session_id=? AND operation_id=?", (session_id, operation_id)).fetchone()
            if row is None:
                raise KeyError(operation_id)
            if row[0] not in _PENDING:
                raise SessionRecoveryRequired("A resolved operation cannot be rewritten")
            db.execute(
                "UPDATE conversation_effects SET status=?, updated_at=? WHERE session_id=? AND operation_id=?",
                (status, _now(), session_id, operation_id),
            )

    def pending_effects(self, session_id: str) -> list[dict]:
        with closing(self._connect()) as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM conversation_effects WHERE session_id=? AND status IN ('pending','unknown','partial') ORDER BY created_at",
                (session_id,),
            ).fetchall()]

    def export_session(self, session_id: str) -> dict:
        return {"schema_version": _VERSION, "sensitive_content": True,
                "session": self.get_session(session_id),
                "messages": [_message_document(message) for message in self.load(session_id)],
                "pending_effects": self.pending_effects(session_id)}

    @staticmethod
    def _copy_database(source: Path, destination: Path) -> Path:
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also rejects aliases of the source and live destinations.
        with destination.open("xb"):
            pass
        try:
            with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src:
                with closing(sqlite3.connect(destination)) as dst:
                    src.backup(dst)
                    if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise SessionSchemaError("Database backup failed integrity check")
            with destination.open("r+b") as stream:
                os.fsync(stream.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return destination

    def backup(self, destination: str | Path) -> Path:
        return self._copy_database(self.database_path, Path(destination))

    @classmethod
    def restore(cls, backup_path: str | Path, new_database_path: str | Path) -> "SessionStore":
        source = Path(backup_path).resolve(strict=True)
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT version FROM conversation_schema").fetchall()
            if rows != [(_VERSION,)]:
                raise SessionSchemaError("Unsupported backup conversation schema")
        return cls(cls._copy_database(source, Path(new_database_path)))
