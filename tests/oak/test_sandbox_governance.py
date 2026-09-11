from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.agent import sandbox_agent as sandbox_module
from app.llm import LLM
from app.tool.policy import ToolPolicy


_ARTIFACTS: list[Path] = []


@pytest.fixture(autouse=True)
def cleanup_sandbox_audits():
    yield
    for path in reversed(_ARTIFACTS):
        if path.is_file():
            path.unlink()
    _ARTIFACTS.clear()
    root = Path(__file__).parents[2] / "workspace" / ".test-oak-sandbox"
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def audit_path() -> Path:
    root = Path(__file__).parents[2] / "workspace" / ".test-oak-sandbox"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{uuid4()}.jsonl"
    _ARTIFACTS.append(path)
    return path


def sandbox_agent(policy: ToolPolicy) -> sandbox_module.SandboxManus:
    return sandbox_module.SandboxManus(
        llm=object.__new__(LLM),
        tool_policy=policy,
    )


@pytest.mark.asyncio
async def test_daytona_allocation_is_denied_before_provider_call_without_grant(
) -> None:
    path = audit_path()
    subject = sandbox_agent(ToolPolicy.guarded(audit_log_path=path))

    with pytest.raises(PermissionError, match="allow-tool sandbox_create"):
        await subject.initialize_sandbox_tools(password="do-not-log")

    audit = path.read_text(encoding="utf-8")
    assert "do-not-log" not in audit
    event = json.loads(audit)
    assert event["decision"] == "denied"
    assert event["status"] == "not_executed"


def test_daytona_allocation_grant_passes_oak_pre_effect_gate() -> None:
    path = audit_path()
    policy = ToolPolicy.guarded(
        allowed_tools={"sandbox_create"},
        audit_log_path=path,
    )
    subject = sandbox_agent(policy)

    argument_digest = subject._require_sandbox_creation()

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(argument_digest) == 64
    assert [event["event_type"] for event in events] == [
        "oak_guard",
        "tool_decision",
    ]
    assert events[0]["allowed"] is True
    assert events[0]["risk_score"] == 5


@pytest.mark.asyncio
async def test_missing_vnc_secret_fails_before_daytona_import() -> None:
    path = audit_path()
    policy = ToolPolicy.guarded(
        allowed_tools={"sandbox_create"},
        audit_log_path=path,
    )
    subject = sandbox_agent(policy)

    with pytest.raises(RuntimeError, match="ValueError"):
        await subject.initialize_sandbox_tools(password=None)

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "ValueError"
