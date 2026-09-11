from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.mcp.server import MCPServer
from app.oak.formal_runtime import approval_reference
from app.oak.runtime import OakRuntime
from app.tool.base import BaseTool, ToolResult
from app.tool.policy import ToolPolicy


class ServerEffect(BaseTool):
    name: str = "effect"
    description: str = "Safe test fixture for a server-side effect."
    parameters: dict = {"type": "object", "properties": {"value": {"type": "string"}}}
    calls: int = 0

    async def execute(self, value: str) -> ToolResult:
        self.calls += 1
        return ToolResult(output=f"external:{value}")


_ARTIFACTS: list[Path] = []


@pytest.fixture(autouse=True)
def cleanup_server_audits():
    yield
    for path in reversed(_ARTIFACTS):
        if path.is_file():
            path.unlink()
    _ARTIFACTS.clear()
    root = Path(__file__).parents[2] / "workspace" / ".test-oak-mcp-server"
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def audit_path() -> Path:
    root = Path(__file__).parents[2] / "workspace" / ".test-oak-mcp-server"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{uuid4()}.jsonl"
    _ARTIFACTS.append(path)
    return path


def server(
    *, granted: bool, tainted_capability: bool = False, approved_operations=()
) -> tuple[MCPServer, ServerEffect, Path]:
    path = audit_path()
    policy = ToolPolicy.guarded(
        allowed_tools={"effect"} if granted else set(),
        allowed_tainted_tools={"effect"} if tainted_capability else set(),
        approved_operations=approved_operations,
        audit_log_path=path,
    )
    runtime = OakRuntime(session_id=policy.session_id)
    subject = MCPServer(tool_policy=policy, oak_runtime=runtime)
    effect = ServerEffect()
    subject.tools["effect"] = effect
    return subject, effect, path


@pytest.mark.asyncio
async def test_mcp_server_denies_ungranted_effect_before_execution() -> None:
    subject, effect, _ = server(granted=False)

    result = await subject.execute_tool("effect", {"value": "private"})

    assert "not authorized" in result
    assert effect.calls == 0


@pytest.mark.asyncio
async def test_mcp_server_gate_exception_fails_closed_and_is_visible(
    monkeypatch,
) -> None:
    subject, effect, path = server(granted=True)

    def fail_gate(*args, **kwargs):
        raise RuntimeError("simulated deterministic gate failure")

    monkeypatch.setattr(subject.oak_runtime, "authorize_tool", fail_gate)

    result = await subject.execute_tool("effect", {"value": "private"})

    assert "guard evaluation failed" in result
    assert effect.calls == 0
    assert any(
        event.get("event_type") == "gate_error"
        for event in subject.oak_runtime.trace
    )
    audit = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert audit[-1]["decision"] == "denied"
    assert audit[-1]["reason"] == "guard_error"


@pytest.mark.asyncio
async def test_mcp_server_applies_source_to_sink_guard_and_metadata_only_audit() -> None:
    subject, effect, path = server(granted=True)
    secret = "private-server-value"

    first = await subject.execute_tool("effect", {"value": secret})
    second = await subject.execute_tool("effect", {"value": secret})

    assert "UNTRUSTED EXTERNAL DATA" in first
    assert "requires exact approval" in second
    assert approval_reference("effect", {"value": secret}) in second
    assert effect.calls == 1
    audit = path.read_text(encoding="utf-8")
    assert secret not in audit
    events = [json.loads(line) for line in audit.splitlines()]
    assert any(event["event_type"] == "oak_guard" for event in events)
    assert any(event["event_type"] == "oak_provenance" for event in events)


@pytest.mark.asyncio
async def test_mcp_server_exact_approval_cannot_authorize_changed_arguments() -> None:
    approved = {"value": "approved"}
    subject, effect, _ = server(
        granted=True,
        tainted_capability=True,
        approved_operations={approval_reference("effect", approved)},
    )

    await subject.execute_tool("effect", {"value": "seed taint"})
    allowed = await subject.execute_tool("effect", approved)
    denied = await subject.execute_tool("effect", {"value": "changed"})

    assert "external:approved" in allowed
    assert "requires exact approval" in denied
    assert effect.calls == 2
