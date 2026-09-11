"""Durable, append-only task ledger used by the OpenManus cockpit.

The SHA-256 chain detects accidental or unaccounted-for changes.  It is not a
digital signature and therefore does not prove who produced an event.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
from uuid import uuid4


SCHEMA_VERSION = 1
CHAIN_DOMAIN = "openmanus-cockpit-event-v1"
TERMINAL_STATES = {"completed", "failed"}
TRANSITIONS = {
    "task.created": ("", "created"),
    "task.accepted": ("created", "accepted"),
    "run.started": ("accepted", "running"),
    "run.completed": ("running", "completed"),
}


class CockpitError(RuntimeError):
    """Base exception for ledger contract violations."""


class IdempotencyConflict(CockpitError):
    """Raised when a request key is reused for a different input."""


class TransitionError(CockpitError):
    """Raised when an event attempts an invalid state transition."""


class TerminalGateError(TransitionError):
    """Raised when a task is completed without sufficient evidence."""


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    errors: tuple[str, ...]
    event_count: int
    head_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_key_index(value: str) -> str:
    return f"sha256:{_sha256_text(value)}"


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _event_preimage(
    *,
    task_id: str,
    position: int,
    event_id: str,
    kind: str,
    state_before: str,
    state_after: str,
    payload_sha256: str,
    created_at: str,
    prev_hash: str,
) -> str:
    return "\n".join(
        (
            CHAIN_DOMAIN,
            task_id,
            str(position),
            event_id,
            kind,
            state_before,
            state_after,
            payload_sha256,
            created_at,
            prev_hash,
        )
    )


def _event_hash(**fields: Any) -> str:
    return _sha256_text(_event_preimage(**fields))


class CockpitStore:
    """SQLite-backed task and evidence ledger with per-task hash chains."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    subject_json TEXT NOT NULL,
                    subject_digest TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('created', 'accepted', 'running', 'completed', 'failed')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    head_hash TEXT NOT NULL DEFAULT '',
                    evidence_count INTEGER NOT NULL DEFAULT 0 CHECK (evidence_count >= 0),
                    unresolved_effects INTEGER NOT NULL DEFAULT 0 CHECK (unresolved_effects >= 0)
                );

                CREATE TABLE IF NOT EXISTS intake_requests (
                    request_key TEXT PRIMARY KEY,
                    input_digest TEXT NOT NULL,
                    task_id TEXT NOT NULL REFERENCES tasks(id)
                );

                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id),
                    position INTEGER NOT NULL CHECK (position > 0),
                    event_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    state_before TEXT NOT NULL,
                    state_after TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, position)
                );

                CREATE INDEX IF NOT EXISTS idx_events_task_position
                    ON events(task_id, position);
                """
            )
            legacy_keys = connection.execute(
                "SELECT request_key FROM intake_requests WHERE request_key NOT LIKE 'sha256:%'"
            ).fetchall()
            for row in legacy_keys:
                legacy = row["request_key"]
                if len(legacy) == 64 and all(
                    character in "0123456789abcdefABCDEF" for character in legacy
                ):
                    normalized = f"sha256:{legacy.lower()}"
                else:
                    normalized = _request_key_index(legacy)
                try:
                    connection.execute(
                        "UPDATE intake_requests SET request_key = ? WHERE request_key = ?",
                        (normalized, legacy),
                    )
                except sqlite3.IntegrityError as exc:
                    raise CockpitError(
                        "cannot migrate a colliding legacy idempotency key"
                    ) from exc

    def admit_task(
        self,
        *,
        goal: str,
        subject: Mapping[str, Any],
        request_key: str,
    ) -> tuple[dict[str, Any], bool]:
        goal = goal.strip()
        request_key = request_key.strip()
        if not goal:
            raise ValueError("goal must not be empty")
        if not request_key:
            raise ValueError("request_key must not be empty")

        subject_value = dict(subject)
        subject_json = _canonical_json(subject_value)
        request_key_sha256 = _sha256_text(request_key)
        request_key_index = _request_key_index(request_key)
        input_digest = _sha256_text(
            _canonical_json({"goal": goal, "subject": subject_value})
        )

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                prior = connection.execute(
                    "SELECT input_digest, task_id FROM intake_requests WHERE request_key = ?",
                    (request_key_index,),
                ).fetchone()
                if prior is not None:
                    if prior["input_digest"] != input_digest:
                        raise IdempotencyConflict(
                            "request_key already belongs to a different input"
                        )
                    task = self._get_task(connection, prior["task_id"])
                    connection.commit()
                    return task, False

                task_id = str(uuid4())
                created_at = _now()
                connection.execute(
                    """
                    INSERT INTO tasks(
                        id, goal, subject_json, subject_digest, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'created', ?, ?)
                    """,
                    (
                        task_id,
                        goal,
                        subject_json,
                        _sha256_text(subject_json),
                        created_at,
                        created_at,
                    ),
                )
                connection.execute(
                    "INSERT INTO intake_requests(request_key, input_digest, task_id) VALUES (?, ?, ?)",
                    (request_key_index, input_digest, task_id),
                )
                self._append_event(
                    connection,
                    task_id=task_id,
                    kind="task.created",
                    state_before="",
                    state_after="created",
                    payload={
                        "request_key_sha256": request_key_sha256,
                        "input_digest": input_digest,
                    },
                )
                task = self._get_task(connection, task_id)
                connection.commit()
                return task, True
            except Exception:
                connection.rollback()
                raise

    def transition(
        self,
        task_id: str,
        *,
        kind: str,
        state_after: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        expected = TRANSITIONS.get(kind)
        if expected is None or kind == "task.created":
            raise TransitionError(f"unsupported transition event: {kind}")
        expected_before, expected_after = expected
        if state_after != expected_after:
            raise TransitionError(
                f"{kind} must transition to {expected_after}, not {state_after}"
            )

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = self._get_task(connection, task_id)
                if task["status"] != expected_before:
                    raise TransitionError(
                        f"{kind} requires {expected_before}, found {task['status']}"
                    )
                if kind == "run.completed" and task["evidence_count"] <= 0:
                    raise TerminalGateError(
                        "completed tasks require at least one evidence event"
                    )
                if kind == "run.completed" and task["unresolved_effects"] != 0:
                    raise TerminalGateError(
                        "completed tasks cannot have unresolved effects"
                    )
                self._append_event(
                    connection,
                    task_id=task_id,
                    kind=kind,
                    state_before=task["status"],
                    state_after=state_after,
                    payload=dict(payload or {}),
                )
                result = self._get_task(connection, task_id)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def record_evidence(
        self,
        task_id: str,
        *,
        evidence_type: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence_type = evidence_type.strip()
        if not evidence_type:
            raise ValueError("evidence_type must not be empty")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = self._get_task(connection, task_id)
                if task["status"] != "running":
                    raise TransitionError("evidence can only be recorded while running")
                evidence_payload = {
                    "evidence_type": evidence_type,
                    "value": dict(payload),
                }
                self._append_event(
                    connection,
                    task_id=task_id,
                    kind="evidence.recorded",
                    state_before="running",
                    state_after="running",
                    payload=evidence_payload,
                    evidence_delta=1,
                )
                result = self._get_task(connection, task_id)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def mark_effect_unknown(
        self, task_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self._same_state_event(
            task_id,
            kind="effect.unknown",
            payload=payload,
            effect_delta=1,
        )

    def resolve_effect(
        self, task_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            task = self._get_task(connection, task_id)
        if task["unresolved_effects"] <= 0:
            raise TransitionError("there is no unresolved effect to resolve")
        return self._same_state_event(
            task_id,
            kind="effect.resolved",
            payload=payload,
            effect_delta=-1,
        )

    def _same_state_event(
        self,
        task_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
        effect_delta: int,
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = self._get_task(connection, task_id)
                if task["status"] not in {"accepted", "running"}:
                    raise TransitionError(
                        f"{kind} cannot be recorded in state {task['status']}"
                    )
                if task["unresolved_effects"] + effect_delta < 0:
                    raise TransitionError("unresolved effect count cannot be negative")
                self._append_event(
                    connection,
                    task_id=task_id,
                    kind=kind,
                    state_before=task["status"],
                    state_after=task["status"],
                    payload=dict(payload),
                    effect_delta=effect_delta,
                )
                result = self._get_task(connection, task_id)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def complete_task(
        self, task_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.transition(
            task_id,
            kind="run.completed",
            state_after="completed",
            payload=payload,
        )

    def fail_task(
        self, task_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = self._get_task(connection, task_id)
                if task["status"] not in {"created", "accepted", "running"}:
                    raise TransitionError(
                        f"run.failed cannot follow {task['status']}"
                    )
                self._append_event(
                    connection,
                    task_id=task_id,
                    kind="run.failed",
                    state_before=task["status"],
                    state_after="failed",
                    payload=dict(payload),
                )
                result = self._get_task(connection, task_id)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def _append_event(
        self,
        connection: sqlite3.Connection,
        *,
        task_id: str,
        kind: str,
        state_before: str,
        state_after: str,
        payload: Mapping[str, Any],
        evidence_delta: int = 0,
        effect_delta: int = 0,
    ) -> None:
        task = self._get_task(connection, task_id)
        if task["status"] in TERMINAL_STATES:
            raise TransitionError(f"cannot append to terminal task {task_id}")
        position = int(
            connection.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM events WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0]
        )
        event_id = str(uuid4())
        payload_json = _canonical_json(dict(payload))
        payload_sha256 = _sha256_text(payload_json)
        created_at = _now()
        prev_hash = task["head_hash"]
        event_hash = _event_hash(
            task_id=task_id,
            position=position,
            event_id=event_id,
            kind=kind,
            state_before=state_before,
            state_after=state_after,
            payload_sha256=payload_sha256,
            created_at=created_at,
            prev_hash=prev_hash,
        )
        connection.execute(
            """
            INSERT INTO events(
                task_id, position, event_id, kind, state_before, state_after,
                payload_json, payload_sha256, prev_hash, event_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                position,
                event_id,
                kind,
                state_before,
                state_after,
                payload_json,
                payload_sha256,
                prev_hash,
                event_hash,
                created_at,
            ),
        )
        connection.execute(
            """
            UPDATE tasks
            SET status = ?, updated_at = ?, head_hash = ?,
                evidence_count = evidence_count + ?,
                unresolved_effects = unresolved_effects + ?
            WHERE id = ?
            """,
            (
                state_after,
                created_at,
                event_hash,
                evidence_delta,
                effect_delta,
                task_id,
            ),
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            return self._get_task(connection, task_id)

    def _get_task(
        self, connection: sqlite3.Connection, task_id: str
    ) -> dict[str, Any]:
        row = connection.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown task: {task_id}")
        task = dict(row)
        task["subject"] = json.loads(task.pop("subject_json"))
        return task

    def list_tasks(self, *, limit: int = 25) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._get_task(connection, row["id"]) for row in rows]

    def export_task(self, task_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            task = self._get_task(connection, task_id)
            rows = connection.execute(
                "SELECT * FROM events WHERE task_id = ? ORDER BY position",
                (task_id,),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event.pop("seq")
            event["payload_canonical"] = event.pop("payload_json")
            event["payload"] = json.loads(event["payload_canonical"])
            events.append(event)
        return {"schema_version": SCHEMA_VERSION, "task": task, "events": events}

    def verify_task(self, task_id: str) -> VerificationResult:
        return verify_export(self.export_task(task_id))


def verify_export(document: Mapping[str, Any]) -> VerificationResult:
    """Independently recompute the ledger invariants from an exported trace."""

    errors: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    task = document.get("task")
    events = document.get("events")
    if not isinstance(task, Mapping):
        return VerificationResult(False, ("task must be an object",), 0, "")
    if not isinstance(events, list):
        return VerificationResult(False, ("events must be a list",), 0, "")

    task_id = str(task.get("id", ""))
    current_state = ""
    prev_hash = ""
    evidence_count = 0
    unresolved_effects = 0

    for expected_position, raw_event in enumerate(events, start=1):
        if not isinstance(raw_event, Mapping):
            errors.append(f"event {expected_position} must be an object")
            continue
        event = raw_event
        position = event.get("position")
        label = f"event {expected_position}"
        if position != expected_position:
            errors.append(f"{label} position must be {expected_position}")
        if event.get("task_id") != task_id:
            errors.append(f"{label} task_id mismatch")
        if event.get("state_before") != current_state:
            errors.append(f"{label} state_before mismatch")
        if event.get("prev_hash") != prev_hash:
            errors.append(f"{label} prev_hash mismatch")

        payload_canonical = event.get("payload_canonical")
        if not isinstance(payload_canonical, str):
            errors.append(f"{label} payload_canonical must be a string")
            payload_canonical = ""
            parsed_payload: Any = None
        else:
            try:
                parsed_payload = json.loads(payload_canonical)
            except json.JSONDecodeError:
                errors.append(f"{label} payload_canonical is invalid JSON")
                parsed_payload = None
        if event.get("payload") != parsed_payload:
            errors.append(f"{label} payload view does not match canonical payload")
        payload_sha256 = _sha256_text(payload_canonical)
        if event.get("payload_sha256") != payload_sha256:
            errors.append(f"{label} payload_sha256 mismatch")

        kind = str(event.get("kind", ""))
        state_after = str(event.get("state_after", ""))
        expected_transition = TRANSITIONS.get(kind)
        if expected_transition is not None:
            if expected_transition != (current_state, state_after):
                errors.append(f"{label} invalid {kind} transition")
        elif kind == "run.failed":
            if current_state not in {"created", "accepted", "running"} or state_after != "failed":
                errors.append(f"{label} invalid run.failed transition")
        elif kind == "evidence.recorded":
            if current_state != "running" or state_after != current_state:
                errors.append(f"{label} evidence outside running state")
            evidence_count += 1
        elif kind == "effect.unknown":
            if current_state not in {"accepted", "running"} or state_after != current_state:
                errors.append(f"{label} invalid effect.unknown state")
            unresolved_effects += 1
        elif kind == "effect.resolved":
            if current_state not in {"accepted", "running"} or state_after != current_state:
                errors.append(f"{label} invalid effect.resolved state")
            unresolved_effects -= 1
            if unresolved_effects < 0:
                errors.append(f"{label} resolved more effects than were opened")
        else:
            errors.append(f"{label} unknown event kind: {kind}")

        fields = {
            "task_id": task_id,
            "position": expected_position,
            "event_id": str(event.get("event_id", "")),
            "kind": kind,
            "state_before": str(event.get("state_before", "")),
            "state_after": state_after,
            "payload_sha256": str(event.get("payload_sha256", "")),
            "created_at": str(event.get("created_at", "")),
            "prev_hash": str(event.get("prev_hash", "")),
        }
        calculated_hash = _event_hash(**fields)
        if event.get("event_hash") != calculated_hash:
            errors.append(f"{label} event_hash mismatch")
        prev_hash = calculated_hash
        current_state = state_after

        if current_state in TERMINAL_STATES and expected_position != len(events):
            errors.append(f"{label} terminal event is not last")

    if not events:
        errors.append("trace must contain at least one event")
    if current_state == "completed" and evidence_count <= 0:
        errors.append("completed trace has no evidence")
    if current_state == "completed" and unresolved_effects != 0:
        errors.append("completed trace has unresolved effects")
    if task.get("status") != current_state:
        errors.append("task status does not match trace")
    if task.get("head_hash") != prev_hash:
        errors.append("task head_hash does not match trace")
    if task.get("evidence_count") != evidence_count:
        errors.append("task evidence_count does not match trace")
    if task.get("unresolved_effects") != unresolved_effects:
        errors.append("task unresolved_effects does not match trace")

    return VerificationResult(
        valid=not errors,
        errors=tuple(errors),
        event_count=len(events),
        head_hash=prev_hash,
    )
