"""CLI ingress and acceptance gates; model calls are replaced at the boundary."""

import asyncio
import json
from pathlib import Path

import pytest


def test_config_override_selects_only_the_explicit_file(tmp_path, monkeypatch):
    from app.config import Config

    path = tmp_path / "isolated.toml"
    path.write_text('[llm]\nmodel="fixture"\n', encoding="utf-8")
    monkeypatch.setenv("OPENMANUS_CONFIG", str(path))
    assert Config._get_config_path() == path.resolve()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        Config._get_config_path()


def test_cli_accepts_opt_in_harness_options(tmp_path):
    from main import build_parser

    args = build_parser().parse_args([
        "--session", "new", "--stream", "--events", str(tmp_path / "events.jsonl"),
        "--context-tokens", "12000", "--acceptance", "acceptance.json",
        "--artifact-root", str(tmp_path),
    ])
    assert args.session == "new"
    assert args.stream is True
    assert args.events == tmp_path / "events.jsonl"
    assert args.context_tokens == 12000
    assert args.acceptance == Path("acceptance.json")
    assert args.artifact_root == tmp_path


@pytest.mark.asyncio
async def test_invalid_acceptance_is_rejected_before_agent_creation(tmp_path, monkeypatch):
    import main as entry

    spec = tmp_path / "acceptance.json"
    spec.write_text(json.dumps({"version": 1, "checks": [
        {"kind": "execute", "path": "anything"}
    ]}), encoding="utf-8")
    created = []

    async def forbidden_create(**kwargs):
        created.append(kwargs)
        raise AssertionError("Model must not be constructed")

    monkeypatch.setattr(entry.Manus, "create", forbidden_create)
    with pytest.raises(ValueError):
        await entry.main([
            "--prompt", "task", "--no-cockpit", "--acceptance", str(spec),
            "--artifact-root", str(tmp_path), "--audit-log", str(tmp_path / "audit.jsonl"),
        ])
    assert not created


@pytest.mark.asyncio
async def test_failed_artifact_acceptance_precedes_oak_finalization(tmp_path, monkeypatch):
    import main as entry
    from app.oak.runtime import OakRuntime

    spec = tmp_path / "acceptance.json"
    spec.write_text(json.dumps({"version": 1, "checks": [
        {"kind": "file_exists", "path": "missing.txt"}
    ]}), encoding="utf-8")
    runtime = OakRuntime()
    finalized = []

    class FakeAgent:
        name = "Manus"
        oak_runtime = runtime

        async def run(self, prompt):
            runtime.begin_task(prompt)
            runtime.record_final_response("Claimed done")
            return "Claimed done"

        async def cleanup(self):
            pass

        def finalize_oak_task(self, status=None):
            finalized.append(status)
            runtime.finish_task(status or "completed")
            return runtime.last_task_status

    async def create(**kwargs):
        return FakeAgent()

    monkeypatch.setattr(entry.Manus, "create", create)
    code = await entry.main([
        "--prompt", "task", "--no-cockpit", "--acceptance", str(spec),
        "--artifact-root", str(tmp_path), "--audit-log", str(tmp_path / "audit.jsonl"),
    ])
    assert finalized == ["failed"]
    assert code == 1


@pytest.mark.asyncio
async def test_cancelled_run_returns_130_and_closes_resources(tmp_path, monkeypatch):
    import main as entry
    from app.oak.runtime import OakRuntime

    runtime = OakRuntime()
    cleaned = []

    class FakeAgent:
        name = "Manus"
        oak_runtime = runtime

        async def run(self, prompt):
            runtime.begin_task(prompt)
            raise asyncio.CancelledError()

        async def cleanup(self):
            cleaned.append(True)

        def finalize_oak_task(self, status=None):
            runtime.finish_task(status or "completed")
            return runtime.last_task_status

    async def create(**kwargs):
        return FakeAgent()

    monkeypatch.setattr(entry.Manus, "create", create)
    code = await entry.main([
        "--prompt", "task", "--no-cockpit", "--audit-log", str(tmp_path / "audit.jsonl"),
    ])
    assert code == 130
    assert cleaned == [True]


def test_session_export_and_restore_do_not_overwrite_existing_destinations(tmp_path):
    import session_main as entry
    from app.harness.session import SessionStore

    database = tmp_path / "session.db"
    store = SessionStore(database)
    session_id = store.create()
    destination = tmp_path / "export.json"
    destination.write_text("preserve me", encoding="utf-8")
    assert entry.main(["--database", str(database), "export", session_id, str(destination)]) == 1
    assert destination.read_text(encoding="utf-8") == "preserve me"
    backup = tmp_path / "backup.db"
    assert entry.main(["--database", str(database), "backup", str(backup)]) == 0
    assert entry.main(["--database", str(database), "restore", str(backup)]) == 1
    assert store.get_session(session_id)["status"] == "idle"


def test_session_control_ingress_keeps_steering_in_the_mailbox(tmp_path):
    import session_main as entry
    from app.harness.control import ControlInbox
    from app.harness.session import SessionStore

    database = tmp_path / "session.db"
    store = SessionStore(database)
    session_id = store.create()
    assert entry.main(["--database", str(database), "steer", session_id, "New direction"]) == 0
    pending = list(ControlInbox(database, session_id).pending())
    assert len(pending) == 1
    assert pending[0][1]["action"] == "steer"
    assert pending[0][1]["text"] == "New direction"
    assert store.load(session_id) == []
