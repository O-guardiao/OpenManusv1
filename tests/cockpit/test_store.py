from __future__ import annotations

import json
import sqlite3

import pytest

from app.cockpit.store import (
    CockpitStore,
    IdempotencyConflict,
    TerminalGateError,
    verify_export,
)


def test_task_ledger_is_idempotent_and_requires_evidence(tmp_path):
    store = CockpitStore(tmp_path / "cockpit.db")
    subject = {"kind": "repository", "path": "C:/work/example"}

    task, created = store.admit_task(
        goal="Reverse engineer the repository",
        subject=subject,
        request_key="repo-example-v1",
    )
    duplicate, duplicate_created = store.admit_task(
        goal="Reverse engineer the repository",
        subject=subject,
        request_key="repo-example-v1",
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate["id"] == task["id"]
    with sqlite3.connect(store.database_path) as connection:
        stored_key = connection.execute(
            "SELECT request_key FROM intake_requests"
        ).fetchone()[0]
    assert stored_key.startswith("sha256:")
    assert "repo-example-v1" not in stored_key

    with pytest.raises(IdempotencyConflict):
        store.admit_task(
            goal="Reverse engineer a different repository",
            subject={"kind": "repository", "path": "C:/work/other"},
            request_key="repo-example-v1",
        )

    store.transition(task["id"], kind="task.accepted", state_after="accepted")
    store.transition(task["id"], kind="run.started", state_after="running")

    with pytest.raises(TerminalGateError):
        store.complete_task(task["id"], {"decision": "clean_room_only"})
    with pytest.raises(TerminalGateError):
        store.transition(
            task["id"],
            kind="run.completed",
            state_after="completed",
            payload={"decision": "bypass-attempt"},
        )

    store.record_evidence(
        task["id"],
        evidence_type="repository.snapshot",
        payload={"commit": "abc123", "tracked_files": 3},
    )
    store.mark_effect_unknown(task["id"], {"effect": "external-write"})
    with pytest.raises(TerminalGateError):
        store.complete_task(task["id"], {"decision": "clean_room_only"})
    store.resolve_effect(task["id"], {"effect": "external-write", "result": "rolled-back"})
    completed = store.complete_task(
        task["id"], {"decision": "clean_room_only"}
    )

    assert completed["status"] == "completed"
    assert completed["evidence_count"] == 1
    exported = store.export_task(task["id"])
    verification = verify_export(exported)
    assert verification.valid, verification.errors

    tampered_view = json.loads(json.dumps(exported))
    tampered_view["events"][3]["payload"]["value"] = {"forged": True}
    forged_verification = verify_export(tampered_view)
    assert forged_verification.valid is False
    assert any("payload view" in error for error in forged_verification.errors)


def test_tampering_is_detected_by_hash_chain(tmp_path):
    database = tmp_path / "cockpit.db"
    store = CockpitStore(database)
    task, _ = store.admit_task(
        goal="Inspect",
        subject={"kind": "repository", "path": "C:/work/example"},
        request_key="tamper-test",
    )
    store.transition(task["id"], kind="task.accepted", state_after="accepted")
    store.transition(task["id"], kind="run.started", state_after="running")
    store.record_evidence(
        task["id"], evidence_type="repository.snapshot", payload={"files": 1}
    )
    store.complete_task(task["id"], {"decision": "inspect"})

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE events SET payload_json = ? WHERE task_id = ? AND position = 4",
            ('{"files":999}', task["id"]),
        )
        connection.commit()

    verification = store.verify_task(task["id"])
    assert verification.valid is False
    assert any("payload_sha256" in error for error in verification.errors)
