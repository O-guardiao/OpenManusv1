import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import ImageContent, ListToolsResult, TextContent
from pydantic import Field

from app.logger import logger
from app.oak.security import sanitize_mcp_metadata
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


def _identifier_digest(value: str) -> str:
    """Return a stable log token without exposing remote-controlled metadata."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class MCPConnectionError(RuntimeError):
    """A provider failed before it could be published atomically."""


class MCPCleanupError(RuntimeError):
    """A provider was unpublished, but one or more resources did not close."""


class _CallTracker:
    """Admission/drain gate for calls racing with provider disposal."""

    def __init__(self) -> None:
        self.accepting = True
        self.inflight = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> bool:
        async with self._condition:
            if not self.accepting:
                return False
            self.inflight += 1
            return True

    async def release(self) -> None:
        async with self._condition:
            self.inflight = max(0, self.inflight - 1)
            self._condition.notify_all()

    async def stop_and_drain(self, timeout_seconds: float) -> None:
        async def wait_until_empty() -> None:
            async with self._condition:
                self.accepting = False
                await self._condition.wait_for(lambda: self.inflight == 0)

        await asyncio.wait_for(wait_until_empty(), timeout=timeout_seconds)


class MCPClientTool(BaseTool):
    """Represents a tool proxy that can be called on the MCP server from the client side."""

    session: Optional[ClientSession] = None
    server_id: str = ""  # Add server identifier
    original_name: str = ""
    metadata_sha256: str = ""
    metadata_trust_tier: str = "untrusted"
    metadata_transformations: List[str] = Field(default_factory=list)
    call_tracker: Any = None

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool by making a remote call to the MCP server."""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        if self.call_tracker is not None and not await self.call_tracker.acquire():
            return ToolResult(error="MCP provider is draining")

        try:
            logger.info(
                f"Executing MCP tool ({_identifier_digest(self.original_name)})"
            )
            result = await self.session.call_tool(self.original_name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            image = next(
                (
                    item.data
                    for item in result.content
                    if isinstance(item, ImageContent)
                ),
                None,
            )
            return ToolResult(
                output=content_str
                or ("Image returned." if image else "No output returned."),
                base64_image=image,
            )
        except Exception as error:
            return ToolResult(
                error=f"MCP tool failed with {type(error).__name__}"
            )
        finally:
            if self.call_tracker is not None:
                await self.call_tracker.release()


class MCPClients(ToolCollection):
    """
    A collection of tools that connects to multiple MCP servers and manages available tools through the Model Context Protocol.
    """

    sessions: Dict[str, ClientSession] = {}
    exit_stacks: Dict[str, AsyncExitStack] = {}
    server_instructions: Dict[str, str] = {}
    description: str = "MCP client tools for server interaction"

    def __init__(self):
        super().__init__()  # Initialize with empty tools list
        self.name = "mcp"  # Keep name for backward compatibility
        self.sessions = {}
        self.exit_stacks = {}
        self.server_instructions = {}
        self.lifecycle: Dict[str, str] = {}
        self.cleanup_failures: Dict[str, str] = {}
        self.failed_exit_stacks: Dict[str, AsyncExitStack] = {}
        self._call_trackers: Dict[str, _CallTracker] = {}
        self.disconnect_timeout_seconds = 30.0

    async def connect_sse(self, server_url: str, server_id: str = "") -> None:
        """Connect to an MCP server using SSE transport."""
        if not server_url:
            raise ValueError("Server URL is required.")

        server_id = server_id or server_url

        # Always ensure clean disconnection before new connection
        if server_id in self.sessions:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.lifecycle[server_id] = "opening"
        try:
            streams_context = sse_client(url=server_url)
            streams = await exit_stack.enter_async_context(streams_context)
            session = await exit_stack.enter_async_context(ClientSession(*streams))
            await self._publish_initialized_provider(
                server_id=server_id,
                session=session,
                exit_stack=exit_stack,
                tool_name_prefix=True,
            )
        except BaseException as error:
            await self._rollback_connect(server_id, exit_stack, error)

    async def connect_stdio(
        self,
        command: str,
        args: List[str],
        server_id: str = "",
        tool_name_prefix: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """Connect to an MCP server using stdio transport."""
        if not command:
            raise ValueError("Server command is required.")

        server_id = server_id or command

        # Always ensure clean disconnection before new connection
        if server_id in self.sessions:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.lifecycle[server_id] = "opening"
        try:
            server_params = StdioServerParameters(command=command, args=args, env=env)
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await self._publish_initialized_provider(
                server_id=server_id,
                session=session,
                exit_stack=exit_stack,
                tool_name_prefix=tool_name_prefix,
            )
        except BaseException as error:
            await self._rollback_connect(server_id, exit_stack, error)

    async def _rollback_connect(
        self,
        server_id: str,
        exit_stack: AsyncExitStack,
        original_error: BaseException,
    ) -> None:
        """Restore the observable pre-state and close locally owned resources."""

        self._unpublish_provider(server_id)
        cleanup_error: BaseException | None = None
        try:
            await exit_stack.aclose()
        except BaseException as error:
            cleanup_error = error
            self.cleanup_failures[server_id] = type(error).__name__
            self.lifecycle[server_id] = "failed_cleanup"
        else:
            self.lifecycle[server_id] = "absent"
        if isinstance(original_error, asyncio.CancelledError):
            raise original_error
        detail = type(original_error).__name__
        if cleanup_error is not None:
            detail += f"; rollback={type(cleanup_error).__name__}"
        raise MCPConnectionError(
            "MCP provider initialization failed before publication: " + detail
        ) from original_error

    async def _publish_initialized_provider(
        self,
        *,
        server_id: str,
        session: ClientSession,
        exit_stack: AsyncExitStack,
        tool_name_prefix: bool,
    ) -> None:
        tracker = _CallTracker()
        instructions, staged_tools, remote_tool_names = await self._stage_tools(
            server_id=server_id,
            session=session,
            tool_name_prefix=tool_name_prefix,
            call_tracker=tracker,
        )
        self.sessions[server_id] = session
        self.exit_stacks[server_id] = exit_stack
        self._call_trackers[server_id] = tracker
        if instructions:
            self.server_instructions[server_id] = instructions
        self.tool_map.update(staged_tools)
        self.tools = tuple(self.tool_map.values())
        self.cleanup_failures.pop(server_id, None)
        self.lifecycle[server_id] = "active"
        self._log_connected(server_id, remote_tool_names)

    async def _initialize_and_list_tools(
        self, server_id: str, tool_name_prefix: bool = True
    ) -> None:
        """Compatibility path for an already-owned test or embedded session."""
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(
                "Session not initialized for MCP server "
                f"({_identifier_digest(server_id)})"
            )

        tracker = self._call_trackers.setdefault(server_id, _CallTracker())
        instructions, staged_tools, remote_names = await self._stage_tools(
            server_id=server_id,
            session=session,
            tool_name_prefix=tool_name_prefix,
            call_tracker=tracker,
        )
        self._remove_server_tools(server_id)
        if instructions:
            self.server_instructions[server_id] = instructions
        else:
            self.server_instructions.pop(server_id, None)
        self.tool_map.update(staged_tools)
        self.tools = tuple(self.tool_map.values())
        self.lifecycle[server_id] = "active"
        self._log_connected(server_id, remote_names)

    async def _stage_tools(
        self,
        *,
        server_id: str,
        session: ClientSession,
        tool_name_prefix: bool,
        call_tracker: _CallTracker,
    ) -> tuple[str, Dict[str, MCPClientTool], list[str]]:
        """Initialize and validate a provider without mutating published state."""

        initialization = await session.initialize()
        response = await session.list_tools()
        staged_tools: Dict[str, MCPClientTool] = {}
        existing_names = {
            name
            for name, tool in self.tool_map.items()
            if getattr(tool, "server_id", None) != server_id
        }
        for tool in response.tools:
            original_name = tool.name
            tool_name = (
                f"mcp_{server_id}_{original_name}"
                if tool_name_prefix
                else original_name
            )
            tool_name = self.sanitize_tool_name(tool_name)
            if not tool_name:
                raise ValueError("MCP tool name is empty after sanitization")
            if tool_name in existing_names or tool_name in staged_tools:
                raise ValueError(
                    "MCP tool name collision after sanitization: "
                    f"{_identifier_digest(tool_name)}"
                )
            description, input_schema, metadata_sha256, transformations = (
                sanitize_mcp_metadata(tool.description, tool.inputSchema)
            )

            server_tool = MCPClientTool(
                name=tool_name,
                description=description,
                parameters=input_schema,
                session=session,
                server_id=server_id,
                original_name=original_name,
                metadata_sha256=metadata_sha256,
                metadata_transformations=transformations,
                call_tracker=call_tracker,
            )
            staged_tools[tool_name] = server_tool
        return (
            str(initialization.instructions or ""),
            staged_tools,
            [str(tool.name) for tool in response.tools],
        )

    def _log_connected(self, server_id: str, remote_names: list[str]) -> None:
        tool_names_sha256 = hashlib.sha256(
            json.dumps(
                sorted(remote_names),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        logger.info(
            "Connected to MCP server "
            f"({_identifier_digest(server_id)}) with {len(remote_names)} tools "
            f"({tool_names_sha256})"
        )

    def _remove_server_tools(self, server_id: str) -> None:
        self.tool_map = {
            name: tool
            for name, tool in self.tool_map.items()
            if getattr(tool, "server_id", None) != server_id
        }
        self.tools = tuple(self.tool_map.values())

    def _unpublish_provider(self, server_id: str) -> None:
        self.sessions.pop(server_id, None)
        self.exit_stacks.pop(server_id, None)
        self.server_instructions.pop(server_id, None)
        self._call_trackers.pop(server_id, None)
        self._remove_server_tools(server_id)

    @staticmethod
    def sanitize_tool_name(name: str) -> str:
        """Sanitize tool name to match MCPClientTool requirements."""
        import re

        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Truncate to 64 characters if needed
        if len(sanitized) > 64:
            sanitized = sanitized[:64]

        return sanitized

    async def list_tools(self) -> ListToolsResult:
        """List all available tools."""
        tools_result = ListToolsResult(tools=[])
        for session in self.sessions.values():
            response = await session.list_tools()
            tools_result.tools += response.tools
        return tools_result

    async def disconnect(self, server_id: str = "") -> None:
        """Disconnect from a specific MCP server or all servers if no server_id provided."""
        if server_id:
            if server_id not in self.sessions and server_id not in self.exit_stacks:
                return
            exit_stack = self.exit_stacks.get(server_id)
            tracker = self._call_trackers.get(server_id)
            self.lifecycle[server_id] = "draining"
            self._remove_server_tools(server_id)
            try:
                if tracker is not None:
                    await tracker.stop_and_drain(self.disconnect_timeout_seconds)
                # Once calls are drained, remove every public handle before the
                # first disposer await. Observers can no longer call a dead provider.
                self._unpublish_provider(server_id)
                if exit_stack is not None:
                    await exit_stack.aclose()
            except BaseException as error:
                self._unpublish_provider(server_id)
                if exit_stack is not None:
                    self.failed_exit_stacks[server_id] = exit_stack
                self.cleanup_failures[server_id] = type(error).__name__
                self.lifecycle[server_id] = "failed_cleanup"
                logger.error(
                    "MCP disconnect failed for server "
                    f"({_identifier_digest(server_id)}) with "
                    f"{type(error).__name__}"
                )
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise MCPCleanupError(
                    "MCP provider cleanup failed visibly: " + type(error).__name__
                ) from error
            self.failed_exit_stacks.pop(server_id, None)
            self.cleanup_failures.pop(server_id, None)
            self.lifecycle[server_id] = "absent"
            logger.info(
                "Disconnected from MCP server "
                f"({_identifier_digest(server_id)})"
            )
        else:
            # Disconnect from all servers in a deterministic order
            failures: list[tuple[str, str]] = []
            server_ids = sorted(set(self.sessions) | set(self.exit_stacks))
            for sid in server_ids:
                try:
                    await self.disconnect(sid)
                except MCPCleanupError as error:
                    failures.append((sid, type(error.__cause__).__name__))
            self.tool_map = {
                name: tool
                for name, tool in self.tool_map.items()
                if not isinstance(tool, MCPClientTool)
            }
            self.tools = tuple(self.tool_map.values())
            if failures:
                summary = ", ".join(
                    f"{_identifier_digest(sid)}:{error_type}"
                    for sid, error_type in failures
                )
                raise MCPCleanupError(
                    f"MCP cleanup failed for {len(failures)} provider(s): {summary}"
                )
            logger.info("Disconnected from all MCP servers")
