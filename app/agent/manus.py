import hashlib
import os
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.oak.tool import OakQuery
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.policy import ToolPolicy
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor


_BROWSER_USE_SERVER_ID = "browser_use"
_BROWSER_USE_COMMAND = "uvx"
_BROWSER_USE_ARGS = ["browser-use", "--cli-mcp"]
_BROWSER_USE_ENV_VARS = (
    "BROWSER_USE_API_KEY",
    "BROWSER_USE_CLOUD_API_URL",
    "BU_BROWSER_ID",
    "BU_CDP_URL",
    "BU_CDP_WS",
    "BU_NAME",
)
_BROWSER_USE_TRANSPORT_INSTRUCTIONS = """\
Browser Use CLI 3.0 is exposed here as MCP tools. When the Browser Use skill
shows `browser-use <<'PY'`, pass the Python body to `browser_exec` instead.
Use `browser_screenshot` when visual inspection is needed. Both tools use the
same persistent browser-harness session as CLI 3.0.
"""


def _identifier_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _browser_use_env() -> Dict[str, str]:
    return {name: value for name in _BROWSER_USE_ENV_VARS if (value := os.getenv(name))}


def mcp_connection_granted(
    policy: ToolPolicy,
    server_id: str,
    *,
    tool_name_prefix: bool,
) -> bool:
    """Require an already-declared tool grant before implicit MCP connection."""

    if policy.default_allow:
        return True
    if server_id == _BROWSER_USE_SERVER_ID:
        return any(
            policy.allows(name) for name in ("browser_exec", "browser_screenshot")
        )
    if not tool_name_prefix:
        return False
    safe_server_id = MCPClients.sanitize_tool_name(server_id).lower()
    prefix = f"mcp_{safe_server_id}_"
    return any(name.startswith(prefix) for name in policy.allowed_tools)


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )
    tool_policy: ToolPolicy = Field(
        default_factory=lambda: ToolPolicy.guarded(
            audit_log_path=config.workspace_root / "audit" / "tool-events.jsonl"
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    mcp_instruction_servers: set[str] = Field(default_factory=set, exclude=True)
    _initialized: bool = False

    @model_validator(mode="after")
    def bind_intrinsic_oak_tool(self) -> "Manus":
        """Expose the same verified runtime used by the execution controller."""

        if "oak_query" not in self.available_tools.tool_map:
            self.available_tools.add_tool(OakQuery(runtime=self.oak_runtime))
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        oak_task_request = kwargs.pop("oak_task_request", None)
        instance = cls(**kwargs)
        if oak_task_request is not None:
            instance.start_owned_oak_task(str(oak_task_request))
        try:
            await instance.initialize_mcp_servers()
            instance._initialized = True
        except BaseException:
            cleanup_status = "failed"
            try:
                await instance.cleanup()
            except Exception:
                cleanup_status = "failed_cleanup"
            if instance.oak_runtime.task_active:
                instance.finalize_oak_task(cleanup_status)
            raise
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        browser_tools_granted = any(
            self.tool_policy.allows(name)
            for name in ("browser_exec", "browser_screenshot")
        )
        if (
            browser_tools_granted
            and _BROWSER_USE_SERVER_ID not in config.mcp_config.servers
            and os.getenv("OPENMANUS_DISABLE_BROWSER_USE", "").lower()
            not in {"1", "true", "yes"}
        ):
            try:
                await self.connect_mcp_server(
                    _BROWSER_USE_COMMAND,
                    _BROWSER_USE_SERVER_ID,
                    use_stdio=True,
                    stdio_args=_BROWSER_USE_ARGS,
                    tool_name_prefix=False,
                    stdio_env=_browser_use_env(),
                )
                logger.info("Connected to Browser Use CLI 3.0 through MCP")
            except Exception as error:
                logger.error(
                    "Browser Use MCP connection failed with "
                    f"{type(error).__name__}"
                )

        for server_id, server_config in config.mcp_config.servers.items():
            if server_id == _BROWSER_USE_SERVER_ID and not browser_tools_granted:
                logger.info(
                    "Browser Use MCP not started because no browser tool was granted"
                )
                continue
            tool_name_prefix = server_id != _BROWSER_USE_SERVER_ID
            if not mcp_connection_granted(
                self.tool_policy,
                server_id,
                tool_name_prefix=tool_name_prefix,
            ):
                logger.info(
                    "MCP server "
                    f"({_identifier_digest(server_id)}) not connected because "
                    "none of its prefixed tools was granted"
                )
                continue
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            "Connected to configured SSE MCP server "
                            f"({_identifier_digest(server_id)})"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                            tool_name_prefix=tool_name_prefix,
                            stdio_env=(
                                _browser_use_env()
                                if server_id == _BROWSER_USE_SERVER_ID
                                else None
                            ),
                        )
                        logger.info(
                            "Connected to configured stdio MCP server "
                            f"({_identifier_digest(server_id)})"
                        )
            except Exception as error:
                logger.error(
                    "MCP connection failed for server "
                    f"({_identifier_digest(server_id)}) with "
                    f"{type(error).__name__}"
                )

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: Optional[List[str]] = None,
        tool_name_prefix: bool = True,
        stdio_env: Optional[Dict[str, str]] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        resolved_server_id = server_id or server_url
        if not mcp_connection_granted(
            self.tool_policy,
            resolved_server_id,
            tool_name_prefix=tool_name_prefix,
        ):
            raise PermissionError(
                "MCP server has no granted tool prefix "
                f"({_identifier_digest(resolved_server_id)})"
            )
        if resolved_server_id in self.mcp_clients.sessions:
            await self.disconnect_mcp_server(resolved_server_id)
        if self.oak_runtime.task_active:
            self.oak_runtime.record_provider_state(resolved_server_id, "opening")
        try:
            if use_stdio:
                await self.mcp_clients.connect_stdio(
                    server_url,
                    stdio_args or [],
                    resolved_server_id,
                    tool_name_prefix=tool_name_prefix,
                    env=stdio_env,
                )
            else:
                await self.mcp_clients.connect_sse(server_url, resolved_server_id)
        except BaseException:
            if self.oak_runtime.task_active:
                lifecycle = self.mcp_clients.lifecycle.get(
                    resolved_server_id, "absent"
                )
                terminal_state = (
                    lifecycle
                    if lifecycle in {"absent", "failed_cleanup"}
                    else "failed_cleanup"
                )
                self.oak_runtime.record_provider_state(
                    resolved_server_id,
                    terminal_state,
                    cleanup_error_type=self.mcp_clients.cleanup_failures.get(
                        resolved_server_id, ""
                    ),
                )
            raise
        self.connected_servers[resolved_server_id] = server_url

        # Update available tools with only the new tools from this server
        new_tools = [
            tool
            for tool in self.mcp_clients.tools
            if tool.server_id == resolved_server_id
        ]
        self.available_tools.add_tools(*new_tools)
        if self.oak_runtime.task_active:
            self.oak_runtime.record_provider_state(
                resolved_server_id,
                "active",
                published_tool_count=len(new_tools),
            )

        instructions = self.mcp_clients.server_instructions.get(resolved_server_id)
        if instructions and resolved_server_id not in self.mcp_instruction_servers:
            transport_instructions = (
                f"{_BROWSER_USE_TRANSPORT_INSTRUCTIONS}\n"
                if resolved_server_id == _BROWSER_USE_SERVER_ID
                else ""
            )
            integrated = self.oak_runtime.integrate_external_content(
                source=(
                    "mcp:"
                    f"{_identifier_digest(resolved_server_id)}:instructions"
                ),
                content=f"{transport_instructions}{instructions}",
                carrier_type="mcp_metadata",
                source_kind="mcp",
                max_content_length=self.max_observe,
            )
            self.memory.add_message(
                Message.user_message(
                    integrated.model_content,
                    provenance=integrated.provenance,
                )
            )
            self.mcp_instruction_servers.add(resolved_server_id)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""
        targets = (
            [server_id]
            if server_id
            else sorted(self.mcp_clients.sessions)
        )
        if self.oak_runtime.task_active:
            for target in targets:
                self.oak_runtime.record_provider_state(target, "draining")
        disconnect_error: BaseException | None = None
        try:
            await self.mcp_clients.disconnect(server_id)
        except BaseException as error:
            disconnect_error = error
        if self.oak_runtime.task_active:
            for target in targets:
                lifecycle = self.mcp_clients.lifecycle.get(target, "absent")
                terminal_state = (
                    lifecycle
                    if lifecycle in {"absent", "failed_cleanup"}
                    else "failed_cleanup"
                )
                self.oak_runtime.record_provider_state(
                    target,
                    terminal_state,
                    cleanup_error_type=self.mcp_clients.cleanup_failures.get(
                        target, ""
                    ),
                )
        if disconnect_error is not None:
            raise disconnect_error
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # Rebuild available tools without the disconnected server's tools
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """Clean up Manus agent resources."""
        if self.mcp_clients.sessions:
            await self.disconnect_mcp_server()
        self._initialized = False

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        return await super().think()
