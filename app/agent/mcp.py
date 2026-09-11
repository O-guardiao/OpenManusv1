import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, PrivateAttr

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.oak.tool import OakQuery
from app.prompt.mcp import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message
from app.tool.mcp import MCPClients
from app.tool.policy import ToolPolicy


class MCPAgent(ToolCallAgent):
    """Agent for interacting with MCP (Model Context Protocol) servers.

    This agent connects to an MCP server using either SSE or stdio transport
    and makes the server's tools available through the agent's tool interface.
    """

    name: str = "mcp_agent"
    description: str = "An agent that connects to an MCP server and uses its tools."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Initialize MCP tool collection
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    available_tools: MCPClients = None  # Will be set in initialize()
    tool_policy: ToolPolicy = Field(
        default_factory=lambda: ToolPolicy.guarded(
            audit_log_path=config.workspace_root / "audit" / "mcp-agent-events.jsonl"
        )
    )

    max_steps: int = 20
    connection_type: str = "stdio"  # "stdio" or "sse"

    # Track tool schemas to detect changes
    tool_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _refresh_tools_interval: int = 5  # Refresh tools every N steps
    _oak_provider_ids: set[str] = PrivateAttr(default_factory=set)

    # Special tool names that should trigger termination
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])

    async def initialize(
        self,
        connection_type: Optional[str] = None,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        server_id: str = "",
        tool_name_prefix: bool = True,
    ) -> None:
        """Initialize the MCP connection.

        Args:
            connection_type: Type of connection to use ("stdio" or "sse")
            server_url: URL of the MCP server (for SSE connection)
            command: Command to run (for stdio connection)
            args: Arguments for the command (for stdio connection)
        """
        if connection_type:
            self.connection_type = connection_type

        resolved_server_id = server_id or command or server_url or ""
        track_provider = self.oak_runtime.task_active
        if track_provider:
            self.oak_runtime.record_provider_state(
                resolved_server_id, "opening"
            )

        try:
            # Connect to the MCP server based on connection type
            if self.connection_type == "sse":
                if not server_url:
                    raise ValueError("Server URL is required for SSE connection")
                await self.mcp_clients.connect_sse(
                    server_url=server_url, server_id=server_id
                )
            elif self.connection_type == "stdio":
                if not command:
                    raise ValueError("Command is required for stdio connection")
                await self.mcp_clients.connect_stdio(
                    command=command,
                    args=args or [],
                    server_id=server_id,
                    tool_name_prefix=tool_name_prefix,
                )
            else:
                raise ValueError(
                    f"Unsupported connection type: {self.connection_type}"
                )

            # Set available_tools to our MCP instance
            self.available_tools = self.mcp_clients
            self.available_tools.add_tool(OakQuery(runtime=self.oak_runtime))

            # Store initial tool schemas
            await self._refresh_tools()
            instructions = self.mcp_clients.server_instructions.get(
                resolved_server_id, ""
            )
        except BaseException as original_error:
            cleanup_error: BaseException | None = None
            if resolved_server_id in self.mcp_clients.sessions:
                try:
                    await self.mcp_clients.disconnect(resolved_server_id)
                except BaseException as error:
                    cleanup_error = error
            if track_provider:
                self.oak_runtime.record_provider_state(
                    resolved_server_id,
                    "failed_cleanup" if cleanup_error else "absent",
                    cleanup_error_type=(
                        type(cleanup_error).__name__ if cleanup_error else ""
                    ),
                )
            raise

        if track_provider:
            self.oak_runtime.record_provider_state(
                resolved_server_id,
                "active",
                published_tool_count=len(self.tool_schemas),
            )
            self._oak_provider_ids.add(resolved_server_id)
            self.defer_oak_finish = True

        # Tool names and schemas already reach the model through the typed tool
        # envelope. Do not duplicate them in history or elevate server metadata.
        if instructions:
            integrated = self.oak_runtime.integrate_external_content(
                source=f"mcp:{resolved_server_id}:instructions",
                content=instructions,
                carrier_type="mcp_metadata",
                source_kind="mcp",
                max_content_length=self.max_observe if self.max_observe else None,
            )
            self.memory.add_message(
                Message.user_message(
                    integrated.model_content,
                    provenance=integrated.provenance,
                )
            )

    async def _refresh_tools(self) -> Tuple[List[str], List[str]]:
        """Refresh the list of available tools from the MCP server.

        Returns:
            A tuple of (added_tools, removed_tools)
        """
        if not self.mcp_clients.sessions:
            return [], []

        # Get current tool schemas directly from the server
        response = await self.mcp_clients.list_tools()
        current_tools = {tool.name: tool.inputSchema for tool in response.tools}

        # Determine added, removed, and changed tools
        current_names = set(current_tools.keys())
        previous_names = set(self.tool_schemas.keys())

        added_tools = list(current_names - previous_names)
        removed_tools = list(previous_names - current_names)

        # Check for schema changes in existing tools
        changed_tools = []
        for name in current_names.intersection(previous_names):
            if current_tools[name] != self.tool_schemas.get(name):
                changed_tools.append(name)

        # Update stored schemas
        self.tool_schemas = current_tools

        # Log and notify about changes
        if added_tools:
            digest = hashlib.sha256(
                json.dumps(sorted(added_tools)).encode("utf-8")
            ).hexdigest()
            logger.info(f"Added {len(added_tools)} MCP tools ({digest})")
        if removed_tools:
            digest = hashlib.sha256(
                json.dumps(sorted(removed_tools)).encode("utf-8")
            ).hexdigest()
            logger.info(f"Removed {len(removed_tools)} MCP tools ({digest})")
        if changed_tools:
            digest = hashlib.sha256(
                json.dumps(sorted(changed_tools)).encode("utf-8")
            ).hexdigest()
            logger.info(f"Changed {len(changed_tools)} MCP tools ({digest})")

        return added_tools, removed_tools

    async def think(self) -> bool:
        """Process current state and decide next action."""
        # Check MCP session and tools availability
        if not self.mcp_clients.sessions or not self.mcp_clients.tool_map:
            logger.info("MCP service is no longer available, ending interaction")
            self.state = AgentState.FINISHED
            return False

        # Refresh tools periodically
        if self.current_step % self._refresh_tools_interval == 0:
            await self._refresh_tools()
            # All tools removed indicates shutdown
            if not self.mcp_clients.tool_map:
                logger.info("MCP service has shut down, ending interaction")
                self.state = AgentState.FINISHED
                return False

        # Use the parent class's think method
        return await super().think()

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        # Terminate if the tool name is 'terminate'
        return name.lower() == "terminate"

    async def cleanup(self) -> None:
        """Clean up MCP connection when done."""
        tracked = tuple(sorted(self._oak_provider_ids))
        if self.oak_runtime.task_active:
            for provider_id in tracked:
                self.oak_runtime.record_provider_state(
                    provider_id, "draining"
                )
        try:
            if self.mcp_clients.sessions:
                await self.mcp_clients.disconnect()
                logger.info("MCP connection closed")
        except BaseException as error:
            if self.oak_runtime.task_active:
                for provider_id in tracked:
                    self.oak_runtime.record_provider_state(
                        provider_id,
                        "failed_cleanup",
                        cleanup_error_type=type(error).__name__,
                    )
            raise
        else:
            if self.oak_runtime.task_active:
                for provider_id in tracked:
                    self.oak_runtime.record_provider_state(
                        provider_id, "absent"
                    )
            self._oak_provider_ids.clear()

    async def run(self, request: Optional[str] = None) -> str:
        """Run one request; the owning runner controls connection lifecycle."""

        return await super().run(request)
