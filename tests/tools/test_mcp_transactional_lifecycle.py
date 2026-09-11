from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from app.tool import mcp as mcp_module
from app.tool.mcp import (
    MCPClientTool,
    MCPClients,
    MCPConnectionError,
    MCPCleanupError,
)


class _AsyncContext:
    def __init__(self, value):
        self.value = value
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.value

    async def __aexit__(self, *exc_info):
        self.exited = True


class _Session(ClientSession):
    def __init__(self, *, fail_list: bool = False):
        self.fail_list = fail_list

    async def initialize(self):
        return SimpleNamespace(instructions="remote instructions")

    async def list_tools(self):
        if self.fail_list:
            raise RuntimeError("remote list failed")
        return ListToolsResult(
            tools=[
                Tool(
                    name="lookup",
                    description="remote lookup",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]
        )


@pytest.mark.asyncio
async def test_connect_failure_restores_absent_state_before_publication(monkeypatch):
    clients = MCPClients()
    transport_context = _AsyncContext((object(), object()))
    session = _Session(fail_list=True)
    session_context = _AsyncContext(session)

    monkeypatch.setattr(
        mcp_module,
        "stdio_client",
        lambda _params: transport_context,
    )
    monkeypatch.setattr(
        mcp_module,
        "ClientSession",
        lambda _read, _write: session_context,
    )

    with pytest.raises(MCPConnectionError, match="before publication"):
        await clients.connect_stdio("fixture", [], server_id="crm")

    assert clients.sessions == {}
    assert clients.exit_stacks == {}
    assert clients.server_instructions == {}
    assert clients.tool_map == {}
    assert clients.lifecycle["crm"] == "absent"
    assert transport_context.exited is True
    assert session_context.exited is True


@pytest.mark.asyncio
async def test_disconnect_unpublishes_before_visible_cleanup_failure():
    clients = MCPClients()
    server_id = "crm"

    class FailingStack:
        async def aclose(self):
            assert server_id not in clients.sessions
            assert not clients.tool_map
            raise RuntimeError("close failed")

    clients.sessions[server_id] = object()
    clients.exit_stacks[server_id] = FailingStack()
    clients.server_instructions[server_id] = "remote"
    clients._call_trackers[server_id] = mcp_module._CallTracker()
    clients.tool_map["mcp_crm_lookup"] = MCPClientTool(
        name="mcp_crm_lookup",
        description="lookup",
            parameters={"type": "object"},
            session=None,
        server_id=server_id,
        original_name="lookup",
    )
    clients.tools = tuple(clients.tool_map.values())

    with pytest.raises(MCPCleanupError, match="failed visibly"):
        await clients.disconnect(server_id)

    assert clients.sessions == {}
    assert clients.exit_stacks == {}
    assert clients.server_instructions == {}
    assert clients.tool_map == {}
    assert clients.lifecycle[server_id] == "failed_cleanup"
    assert clients.cleanup_failures[server_id] == "RuntimeError"
    assert server_id in clients.failed_exit_stacks


@pytest.mark.asyncio
async def test_disconnect_drains_inflight_call_before_disposal():
    clients = MCPClients()
    server_id = "crm"
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = False

    class SlowSession(ClientSession):
        def __init__(self):
            pass

        async def call_tool(self, _name, _arguments):
            entered.set()
            await release.wait()
            return CallToolResult(
                content=[TextContent(type="text", text="done")]
            )

    class Stack:
        async def aclose(self):
            nonlocal closed
            closed = True

    tracker = mcp_module._CallTracker()
    session = SlowSession()
    tool = MCPClientTool(
        name="mcp_crm_lookup",
        description="lookup",
        parameters={"type": "object"},
        session=session,
        server_id=server_id,
        original_name="lookup",
        call_tracker=tracker,
    )
    clients.sessions[server_id] = session
    clients.exit_stacks[server_id] = Stack()
    clients._call_trackers[server_id] = tracker
    clients.tool_map[tool.name] = tool
    clients.tools = (tool,)

    call = asyncio.create_task(tool.execute())
    await entered.wait()
    disconnect = asyncio.create_task(clients.disconnect(server_id))
    await asyncio.sleep(0)

    assert tool.name not in clients.tool_map
    assert clients.lifecycle[server_id] == "draining"
    assert closed is False

    release.set()
    result = await call
    await disconnect

    assert result.output == "done"
    assert closed is True
    assert clients.lifecycle[server_id] == "absent"
