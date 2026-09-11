"""Durability and process ownership tests; no model or external service calls."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app.harness.session import (
    SessionBusy, SessionRecoveryRequired, SessionSchemaError, SessionStore,
)
from app.schema import Message, MessageProvenance


def test_roundtrip_messages_and_provenance(tmp_path):
    path = tmp_path / "cockpit.db"
    store = SessionStore(path)
    session = store.create()
    provenance = MessageProvenance(
        source="search", source_kind="web", trust_tier="untrusted",
        carrier_type="text", content_sha256="a" * 64,
        normalized_sha256="b" * 64, taint_ids=["taint:1"],
        transformations=["framed"],
    )
    message = Message.tool_message("external text", "search", "call-1", provenance=provenance)
    store.append(session, Message.user_message("private prompt"))
    store.append(session, message)
    reopened = SessionStore(path)
    loaded = reopened.load(session)
    assert loaded[1].provenance == provenance
    assert loaded[1].content == message.content
    assert loaded[1].tool_call_id == "call-1"
    document = reopened.export_session(session)
    assert document["messages"][1]["provenance"]["taint_ids"] == ["taint:1"]
    assert "private prompt" in json.dumps(document)
    assert document["sensitive_content"] is True
    assert reopened.list_sessions()[0]["message_count"] == 2
    assert "grants" not in document


def test_run_ownership_and_pending_recovery(tmp_path):
    store = SessionStore(tmp_path / "cockpit.db")
    session = store.create()
    with pytest.raises(SessionBusy):
        store.begin_run(session, "run-1")
    with store.lease(session):
        store.begin_run(session, "run-1")
        with pytest.raises(SessionBusy):
            store.begin_run(session, "run-2")
        store.record_effect_intent(session, "op-1", "python_execute", "a" * 64)
        with pytest.raises(SessionRecoveryRequired):
            store.record_effect_intent(session, "op-1", "python_execute", "a" * 64)
        with pytest.raises(SessionRecoveryRequired):
            store.finish_run(session, "completed")
    assert store.get_session(session)["status"] == "interrupted"
    with store.lease(session):
        with pytest.raises(SessionRecoveryRequired):
            store.begin_run(session, "run-3")
        store.resolve_effect(session, "op-1", "failed_before_effect")
        store.begin_run(session, "run-3")
        store.finish_run(session, "completed")
    assert store.get_session(session)["status"] == "completed"
    assert store.pending_effects(session) == []


def test_live_owner_prevents_second_store_writer(tmp_path):
    store = SessionStore(tmp_path / "cockpit.db")
    second = SessionStore(store.database_path)
    session = store.create()
    with store.lease(session):
        with pytest.raises(SessionBusy):
            with second.lease(session):
                pass
        with pytest.raises(SessionBusy):
            second.append(session, Message.user_message("must not append"))
    assert store.load(session) == []


@pytest.mark.parametrize("status", ["unknown", "partial"])
def test_unresolved_status_and_reentrant_lease(tmp_path, status):
    store = SessionStore(tmp_path / "cockpit.db")
    session = store.create()
    with store.lease(session):
        store.begin_run(session, "one")
        with store.lease(session):
            store.append(session, Message.user_message("nested write"))
            assert store.get_session(session)["status"] == "running"
        store.record_effect_intent(session, "op", "write", "a" * 64)
        store.resolve_effect(session, "op", status)
        store.finish_run(session, "failed")
        assert store.pending_effects(session)[0]["status"] == status
        with pytest.raises(SessionRecoveryRequired):
            store.begin_run(session, "two")
        store.resolve_effect(session, "op", "not_dispatched")
        store.begin_run(session, "two")
        store.finish_run(session, "completed")
    assert store.load(session)[0].content == "nested write"


def test_child_process_lock_and_crash_release(tmp_path):
    store = SessionStore(tmp_path / "cockpit.db")
    session = store.create()
    source = (
        "import sys,time\n"
        "from app.harness.session import SessionStore\n"
        "s=SessionStore(sys.argv[1])\n"
        "with s.lease(sys.argv[2]):\n"
        " s.begin_run(sys.argv[2], 'child-run')\n"
        " s.append(sys.argv[2], __import__('app.schema',fromlist=['Message']).Message.user_message('survives crash'))\n"
        " s.record_effect_intent(sys.argv[2], 'child-op', 'write', 'a'*64)\n"
        " print('LOCKED', flush=True)\n"
        " time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", source, str(store.database_path), session],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    try:
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(SessionBusy):
            with store.lease(session):
                pass
    finally:
        child.kill()
        child.communicate(timeout=10)
    reopened = SessionStore(store.database_path)
    with reopened.lease(session):
        assert reopened.get_session(session)["status"] == "interrupted"
        assert reopened.load(session)[0].content == "survives crash"
        with pytest.raises(SessionRecoveryRequired):
            reopened.begin_run(session, "after-crash")


def test_future_schema_is_refused_without_modification(tmp_path):
    path = tmp_path / "cockpit.db"
    SessionStore(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE conversation_schema SET version=99")
    with pytest.raises(SessionSchemaError):
        SessionStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM conversation_schema").fetchone()[0] == 99


def test_backup_restore_reopens_messages_and_preserves_legacy_tables(tmp_path):
    path = tmp_path / "cockpit.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE legacy_fixture (value TEXT)")
        db.execute("INSERT INTO legacy_fixture VALUES ('preserved')")
    store = SessionStore(path)
    session = store.create()
    store.append(session, Message.user_message("backup survives"))
    backup = tmp_path / "backup.db"
    store.backup(backup)
    restored_path = tmp_path / "restored.db"
    restored = SessionStore.restore(backup, restored_path)
    assert restored.load(session)[0].content == "backup survives"
    assert SessionStore(restored_path).load(session)[0].content == "backup survives"
    with sqlite3.connect(restored_path) as db:
        assert db.execute("SELECT value FROM legacy_fixture").fetchone()[0] == "preserved"
    with pytest.raises(FileExistsError):
        store.backup(path)
    with pytest.raises(FileExistsError):
        SessionStore.restore(backup, path)


def test_foreign_keys_and_append_only_transcript(tmp_path):
    store = SessionStore(tmp_path / "cockpit.db")
    session = store.create()
    store.append(session, Message.user_message("immutable"))
    with sqlite3.connect(store.database_path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE conversation_messages SET message_json='{}'")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM conversation_messages")
    with pytest.raises(KeyError):
        store.load("00000000-0000-0000-0000-000000000000")
