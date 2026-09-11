import hashlib
import json
import time
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import Field, PrivateAttr, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.oak.formal_runtime import (
    BudgetExceeded,
    EffectConflict,
    approval_reference,
)
from app.oak.tool import OakQuery
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.mcp import MCPClients, MCPClientTool


def _identifier_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SandboxManus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""

    name: str = "SandboxManus"
    description: str = "A versatile agent that can solve various tasks using multiple sandbox-tools including MCP-based tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            # PythonExecute(),
            # BrowserUseTool(),
            # StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    mcp_instruction_servers: set[str] = Field(default_factory=set, exclude=True)
    _initialized: bool = False
    sandbox_link: Optional[dict[str, dict[str, str]]] = Field(default_factory=dict)
    _sandbox_gate_event_id: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def initialize_helper(self) -> "SandboxManus":
        """Initialize basic components synchronously."""
        self.browser_context_helper = BrowserContextHelper(self)
        if "oak_query" not in self.available_tools.tool_map:
            self.available_tools.add_tool(OakQuery(runtime=self.oak_runtime))
        return self

    @classmethod
    async def create(cls, **kwargs) -> "SandboxManus":
        """Factory method to create and properly initialize a Manus instance."""
        oak_task_request = kwargs.pop("oak_task_request", None)
        instance = cls(**kwargs)
        if oak_task_request is not None:
            instance.start_owned_oak_task(str(oak_task_request))
        try:
            await instance.initialize_mcp_servers()
            await instance.initialize_sandbox_tools()
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

    def _require_sandbox_creation(self) -> str:
        """Persist a monotonic pre-effect decision for Daytona allocation."""

        name = "sandbox_create"
        arguments = {"provider": "daytona", "public": True}
        argument_digest = hashlib.sha256(
            json.dumps(arguments, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if not self.tool_policy.allows(name):
            self._record_tool_event(
                name=name,
                decision="denied",
                status="not_executed",
                reason="not_granted_for_session",
                arguments_sha256=argument_digest,
            )
            raise PermissionError(
                "Daytona sandbox creation requires --allow-tool sandbox_create"
            )
        decision = self.oak_runtime.authorize_tool(
            name,
            arguments,
            granted=True,
            tainted_sink_approved=self.tool_policy.allows_tainted(name, arguments),
            principal=self.tool_policy.principal,
            policy_version=self.tool_policy.policy_version,
        )
        self.tool_policy.record_oak_guard(
            agent_name=self.name,
            tool_name=name,
            allowed=decision.allowed,
            reason=decision.reason,
            source=decision.profile.source,
            sink=decision.profile.sink,
            risk_score=decision.profile.risk_score,
            taint_count=len(decision.taint_ids),
            outcome=decision.outcome,
            arguments_sha256=decision.arguments_sha256,
            approval_key_value=decision.approval_key,
            state_version=decision.state_version,
        )
        if not decision.allowed:
            raise PermissionError(
                "Daytona sandbox creation requires exact approval "
                f"{approval_reference(name, arguments)} after untrusted input"
            )
        self._sandbox_gate_event_id = decision.event_id
        self._record_tool_event(
            name=name,
            decision="allowed",
            status="authorized",
            reason=decision.reason,
            arguments_sha256=argument_digest,
        )
        return argument_digest

    async def initialize_sandbox_tools(
        self,
        password: Optional[str] = config.daytona.VNC_password,
    ) -> None:
        argument_digest = self._require_sandbox_creation()
        started_at = time.perf_counter()
        effect = None
        provider_opened = False
        provider_active = False
        try:
            if not password:
                raise ValueError("VNC password must be configured explicitly")

            from app.daytona.sandbox import create_sandbox
            from app.daytona.tool_base import SandboxToolsBase
            from app.tool.sandbox.sb_browser_tool import SandboxBrowserTool
            from app.tool.sandbox.sb_files_tool import SandboxFilesTool
            from app.tool.sandbox.sb_shell_tool import SandboxShellTool
            from app.tool.sandbox.sb_vision_tool import SandboxVisionTool

            if self.oak_runtime.task_active:
                arguments = {"provider": "daytona", "public": True}
                effect = self.oak_runtime.begin_effect(
                    operation_id=f"sandbox-create:{uuid4()}",
                    principal=self.tool_policy.principal,
                    tool_name="sandbox_create",
                    arguments=arguments,
                    gate_event_id=self._sandbox_gate_event_id,
                )
                self.tool_policy.record_effect_intent(
                    agent_name=self.name,
                    operation_id=effect.operation_id,
                    tool_name="sandbox_create",
                    arguments_sha256=effect.arguments_sha256,
                    idempotency_key=effect.idempotency_key,
                    gate_event_id=self._sandbox_gate_event_id,
                )
                self.oak_runtime.dispatch_effect(effect.operation_id)
                self.oak_runtime.record_provider_state(
                    "daytona-sandbox", "opening"
                )
                provider_opened = True

            # 创建新沙箱
            sandbox = create_sandbox(password=password)
            self.sandbox = sandbox
            vnc_link = sandbox.get_preview_link(6080)
            website_link = sandbox.get_preview_link(8080)
            vnc_url = vnc_link.url if hasattr(vnc_link, "url") else str(vnc_link)
            website_url = (
                website_link.url if hasattr(website_link, "url") else str(website_link)
            )

            # Get the actual sandbox_id from the created sandbox
            actual_sandbox_id = sandbox.id if hasattr(sandbox, "id") else "new_sandbox"
            if not self.sandbox_link:
                self.sandbox_link = {}
            self.sandbox_link[actual_sandbox_id] = {
                "vnc": vnc_url,
                "website": website_url,
            }
            sandbox_id_sha256 = hashlib.sha256(
                actual_sandbox_id.encode("utf-8")
            ).hexdigest()
            logger.info(
                "Daytona sandbox created with two preview endpoints "
                f"({sandbox_id_sha256})"
            )
            SandboxToolsBase._urls_printed = True
            sb_tools = [
                SandboxBrowserTool(sandbox),
                SandboxFilesTool(sandbox),
                SandboxShellTool(sandbox),
                SandboxVisionTool(sandbox),
            ]
            self.available_tools.add_tools(*sb_tools)

            if self.oak_runtime.task_active:
                self.oak_runtime.record_provider_state(
                    "daytona-sandbox",
                    "active",
                    published_tool_count=len(sb_tools),
                )
                provider_active = True

            self.oak_runtime.record_tool_outcome(
                "sandbox_create",
                "succeeded",
                "daytona_allocation",
                operation_id=effect.operation_id if effect else None,
            )
            observation = self.oak_runtime.integrate_external_content(
                source="sandbox_create",
                content=(
                    "Daytona sandbox allocated; "
                    f"id_sha256={sandbox_id_sha256}; preview_endpoint_count=2"
                ),
                carrier_type="service_metadata",
                source_kind="sandbox",
            )
            self._record_tool_event(
                name="sandbox_create",
                decision="allowed",
                status="succeeded",
                reason="daytona_allocation",
                arguments_sha256=argument_digest,
                duration_ms=round((time.perf_counter() - started_at) * 1000),
            )
            self.tool_policy.record_provenance(
                agent_name=self.name,
                tool_name="sandbox_create",
                content_sha256=observation.provenance.content_sha256,
                normalized_sha256=observation.provenance.normalized_sha256,
                carrier_type=observation.provenance.carrier_type,
                trust_tier=observation.provenance.trust_tier,
                taint_count=len(observation.provenance.taint_ids),
                transformations=observation.provenance.transformations,
            )
            if effect is not None:
                self.tool_policy.record_effect_outcome(
                    agent_name=self.name,
                    operation_id=effect.operation_id,
                    tool_name="sandbox_create",
                    status="succeeded",
                    idempotency_key=effect.idempotency_key,
                )
                self.oak_runtime.observe_effect(effect.operation_id, "succeeded")

        except (Exception, BudgetExceeded, EffectConflict) as e:
            self.oak_runtime.record_tool_outcome(
                "sandbox_create",
                "failed",
                type(e).__name__,
                operation_id=effect.operation_id if effect else None,
            )
            if effect is not None:
                self.oak_runtime.observe_effect(effect.operation_id, "unknown")
                try:
                    self.tool_policy.record_effect_outcome(
                        agent_name=self.name,
                        operation_id=effect.operation_id,
                        tool_name="sandbox_create",
                        status="unknown",
                        idempotency_key=effect.idempotency_key,
                    )
                except OSError:
                    pass
            if (
                provider_opened
                and not provider_active
                and self.oak_runtime.task_active
            ):
                self.oak_runtime.record_provider_state(
                    "daytona-sandbox",
                    "failed_cleanup",
                    cleanup_error_type=type(e).__name__,
                )
            try:
                self._record_tool_event(
                    name="sandbox_create",
                    decision="allowed",
                    status="failed",
                    reason=type(e).__name__,
                    arguments_sha256=argument_digest,
                    duration_ms=round((time.perf_counter() - started_at) * 1000),
                )
            except OSError:
                pass
            logger.error(
                f"Sandbox initialization failed with {type(e).__name__}"
            )
            raise RuntimeError(
                f"sandbox_create failed with {type(e).__name__}"
            ) from None

    def _configured_mcp_granted(self, server_id: str) -> bool:
        if self.tool_policy.default_allow:
            return True
        safe_server_id = MCPClients.sanitize_tool_name(server_id).lower()
        prefix = f"mcp_{safe_server_id}_"
        return any(name.startswith(prefix) for name in self.tool_policy.allowed_tools)

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        for server_id, server_config in config.mcp_config.servers.items():
            if not self._configured_mcp_granted(server_id):
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
    ) -> None:
        """Connect to an MCP server and add its tools."""
        resolved_server_id = server_id or server_url
        if not self._configured_mcp_granted(resolved_server_id):
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
                    server_url, stdio_args or [], resolved_server_id
                )
            else:
                await self.mcp_clients.connect_sse(
                    server_url, resolved_server_id
                )
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
            integrated = self.oak_runtime.integrate_external_content(
                source=(
                    "mcp:"
                    f"{_identifier_digest(resolved_server_id)}:instructions"
                ),
                content=instructions,
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
        targets = [server_id] if server_id else sorted(self.mcp_clients.sessions)
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

    async def delete_sandbox(self, sandbox_id: str) -> None:
        """Delete a sandbox by ID."""
        sandbox_id_sha256 = hashlib.sha256(sandbox_id.encode("utf-8")).hexdigest()
        try:
            from app.daytona.sandbox import delete_sandbox

            await delete_sandbox(sandbox_id)
            logger.info(f"Sandbox deleted successfully ({sandbox_id_sha256})")
            if sandbox_id in self.sandbox_link:
                del self.sandbox_link[sandbox_id]
        except Exception as e:
            logger.error(f"Sandbox deletion failed with {type(e).__name__}")
            raise

    async def cleanup(self):
        """Clean up Manus agent resources."""
        failures: list[tuple[str, str]] = []
        if self.browser_context_helper:
            try:
                await self.browser_context_helper.cleanup_browser()
            except Exception as error:
                failures.append(("browser", type(error).__name__))
        if self.mcp_clients.sessions:
            try:
                await self.disconnect_mcp_server()
            except Exception as error:
                failures.append(("mcp", type(error).__name__))
        sandbox = getattr(self, "sandbox", None)
        if sandbox is not None and getattr(sandbox, "id", None):
            if self.oak_runtime.task_active:
                self.oak_runtime.record_provider_state(
                    "daytona-sandbox", "draining"
                )
            try:
                await self.delete_sandbox(sandbox.id)
            except Exception as error:
                failures.append(("daytona", type(error).__name__))
                if self.oak_runtime.task_active:
                    self.oak_runtime.record_provider_state(
                        "daytona-sandbox",
                        "failed_cleanup",
                        cleanup_error_type=type(error).__name__,
                    )
            else:
                self.sandbox = None
                if self.oak_runtime.task_active:
                    self.oak_runtime.record_provider_state(
                        "daytona-sandbox", "absent"
                    )
        self._initialized = False
        if failures:
            summary = ", ".join(
                f"{resource}:{error_type}"
                for resource, error_type in failures
            )
            raise RuntimeError(f"sandbox cleanup failed: {summary}")

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == "sandbox_browser"
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result
