import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.toml"
_CREATED_TEST_CONFIG = not _CONFIG_PATH.exists()
if _CREATED_TEST_CONFIG:
    _CONFIG_PATH.write_text(
        '[llm]\nmodel = "gpt-4o"\nbase_url = "http://localhost"\napi_key = "test"\n'
        '\n[daytona]\ndaytona_api_key = "test"\n',
        encoding="utf-8",
    )

try:
    from app.agent.toolcall import ToolCallAgent
    from app.flow import planning as planning_module
    from app.flow.planning import PlanningFlow
    from app.llm import LLM
    from app.schema import Function, ToolCall
    from app.tool.base import BaseTool, ToolResult
    from app.tool.policy import ToolPolicy
    from app.tool.tool_collection import ToolCollection
    from main import build_parser
finally:
    if _CREATED_TEST_CONFIG:
        _CONFIG_PATH.unlink()


_TEST_ARTIFACTS: list[Path] = []


@pytest.fixture(autouse=True)
def cleanup_policy_test_artifacts():
    yield
    for artifact in reversed(_TEST_ARTIFACTS):
        if artifact.is_file():
            artifact.unlink()
    _TEST_ARTIFACTS.clear()
    test_root = Path(__file__).parents[2] / "workspace" / ".test-tool-policy"
    if test_root.is_dir():
        test_root.rmdir()


class RecordingTool(BaseTool):
    name: str = "effect_tool"
    description: str = "Record whether execution happened."
    parameters: dict = {
        "type": "object",
        "properties": {"payload": {"type": "string"}},
        "required": ["payload"],
    }
    calls: int = 0

    async def execute(self, payload: str) -> ToolResult:
        self.calls += 1
        return ToolResult(output=f"accepted:{len(payload)}")


def _agent(tool: RecordingTool, policy: ToolPolicy) -> ToolCallAgent:
    return ToolCallAgent(
        available_tools=ToolCollection(tool),
        llm=object.__new__(LLM),
        tool_policy=policy,
    )


def _command(payload: str) -> ToolCall:
    return ToolCall(
        id="call-1",
        function=Function(
            name="effect_tool",
            arguments=json.dumps({"payload": payload}),
        ),
    )


def _test_path(name: str) -> Path:
    root = Path(__file__).parents[2] / "workspace" / ".test-tool-policy"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{uuid4()}-{name}"
    _TEST_ARTIFACTS.append(path)
    return path


@pytest.mark.asyncio
async def test_ungranted_tool_is_blocked_and_secret_is_not_audited():
    secret = "customer-secret-value"
    raw_arguments = json.dumps({"payload": secret})
    audit_log = _test_path("denied-audit.jsonl")
    tool = RecordingTool()
    agent = _agent(tool, ToolPolicy.guarded(audit_log_path=audit_log))

    result = await agent.execute_tool(_command(secret))

    assert "not authorized" in result
    assert tool.calls == 0
    audit_text = audit_log.read_text(encoding="utf-8")
    assert secret not in audit_text
    event = json.loads(audit_text)
    assert event["decision"] == "denied"
    assert event["status"] == "not_executed"
    assert event["arguments_sha256"] == hashlib.sha256(
        raw_arguments.encode("utf-8")
    ).hexdigest()


@pytest.mark.asyncio
async def test_explicit_grant_executes_and_records_metadata_only():
    secret = "do-not-log-this"
    audit_log = _test_path("allowed-audit.jsonl")
    tool = RecordingTool()
    policy = ToolPolicy.guarded(
        allowed_tools={"effect_tool"},
        audit_log_path=audit_log,
    )
    agent = _agent(tool, policy)

    result = await agent.execute_tool(_command(secret))

    assert "accepted" in result
    assert tool.calls == 1
    audit_text = audit_log.read_text(encoding="utf-8")
    assert secret not in audit_text
    events = [json.loads(line) for line in audit_text.splitlines()]
    tool_events = [event for event in events if event["event_type"] == "tool_decision"]
    assert [(event["decision"], event["status"]) for event in tool_events] == [
        ("allowed", "authorized"),
        ("allowed", "succeeded"),
    ]
    assert {event["event_type"] for event in events} >= {
        "oak_guard",
        "oak_provenance",
        "tool_decision",
    }


def test_model_only_sees_tools_authorized_for_the_session():
    policy = ToolPolicy.guarded(audit_log_path=_test_path("visible-audit.jsonl"))
    agent = _agent(RecordingTool(), policy)

    assert agent.available_tool_params() == []


@pytest.mark.asyncio
async def test_audit_failure_blocks_an_allowed_effect():
    not_a_directory = _test_path("not-a-directory")
    not_a_directory.write_text("occupied", encoding="utf-8")
    tool = RecordingTool()
    policy = ToolPolicy.guarded(
        allowed_tools={"effect_tool"},
        audit_log_path=not_a_directory / "audit.jsonl",
    )
    agent = _agent(tool, policy)

    result = await agent.execute_tool(_command("payload"))

    assert "audit trail unavailable" in result
    assert tool.calls == 0


@pytest.mark.asyncio
async def test_guarded_effect_without_audit_destination_is_blocked():
    tool = RecordingTool()
    policy = ToolPolicy.guarded(allowed_tools={"effect_tool"})
    agent = _agent(tool, policy)

    result = await agent.execute_tool(_command("payload"))

    assert "audit trail unavailable" in result
    assert tool.calls == 0


@pytest.mark.asyncio
async def test_post_effect_audit_failure_marks_outcome_unknown(monkeypatch):
    audit_log = _test_path("post-effect-failure.jsonl")
    tool = RecordingTool()
    policy = ToolPolicy.guarded(
        allowed_tools={"effect_tool"},
        audit_log_path=audit_log,
    )
    agent = _agent(tool, policy)
    original_record = ToolPolicy.record
    calls = 0

    def fail_second_record(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated audit outage")
        return original_record(self, **kwargs)

    monkeypatch.setattr(ToolPolicy, "record", fail_second_record)

    result = await agent.execute_tool(_command("payload"))

    assert tool.calls == 1
    assert "outcome is UNKNOWN" in result
    assert "Do not retry automatically" in result


def test_cli_collects_explicit_session_grants():
    digest = "a" * 64
    args = build_parser().parse_args(
        [
            "--allow-tool",
            "python_execute",
            "--allow-tool",
            "browser_screenshot",
            "--allow-tainted-tool",
            "str_replace_editor",
            "--approve-tainted-operation",
            f"str_replace_editor:{digest}",
        ]
    )

    assert args.allow_tool == ["python_execute", "browser_screenshot"]
    assert args.allow_tainted_tool == ["str_replace_editor"]
    assert args.approve_tainted_operation == [f"str_replace_editor:{digest}"]


def test_session_lifecycle_records_request_hash_and_grants_without_prompt():
    prompt = "private customer task"
    audit_log = _test_path("session-audit.jsonl")
    policy = ToolPolicy.guarded(
        allowed_tools={"python_execute"},
        audit_log_path=audit_log,
    )

    policy.record_session(
        agent_name="Manus",
        status="started",
        request_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        reason="operator_grants_captured",
    )

    audit_text = audit_log.read_text(encoding="utf-8")
    assert prompt not in audit_text
    event = json.loads(audit_text)
    assert event["event_type"] == "session"
    assert event["status"] == "started"
    assert event["granted_tools"] == [
        "ask_human",
        "oak_query",
        "python_execute",
        "terminate",
    ]
    assert event["tainted_sink_grants"] == []


@pytest.mark.asyncio
async def test_planning_flow_does_not_log_raw_arguments_or_results(monkeypatch):
    secret = "private-plan-secret"
    messages: list[str] = []

    class FakeLLM:
        async def ask_tool(self, **kwargs):
            return SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="planning",
                            arguments=json.dumps(
                                {
                                    "command": "create",
                                    "title": secret,
                                    "steps": ["one"],
                                }
                            ),
                        )
                    )
                ]
            )

    class FakePlanningTool:
        def to_param(self):
            return {"type": "function", "function": {"name": "planning"}}

        async def execute(self, **kwargs):
            return f"created:{secret}"

    monkeypatch.setattr(
        planning_module,
        "logger",
        SimpleNamespace(
            info=lambda message: messages.append(str(message)),
            error=lambda message: messages.append(str(message)),
            warning=lambda message: messages.append(str(message)),
        ),
    )
    flow = PlanningFlow.model_construct(
        llm=FakeLLM(),
        planning_tool=FakePlanningTool(),
        agents={"manus": SimpleNamespace(description="executor")},
        executor_keys=["manus"],
        active_plan_id="plan-test",
    )

    await flow._create_initial_plan("ordinary request")

    assert secret not in "\n".join(messages)
