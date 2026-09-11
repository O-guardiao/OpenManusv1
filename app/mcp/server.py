import logging
import sys


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])

import argparse
import asyncio
import atexit
import hashlib
import json
from inspect import Parameter, Signature
from pathlib import Path
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from app.config import config
from app.logger import logger
from app.oak.runtime import OakRuntime
from app.oak.formal_runtime import BudgetExceeded, EffectConflict, approval_reference
from app.oak.tool import OakQuery
from app.tool.base import BaseTool
from app.tool.bash import Bash
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate
from app.tool.policy import ToolPolicy


class MCPServer:
    """MCP Server implementation with tool registration and management."""

    def __init__(
        self,
        name: str = "openmanus",
        *,
        tool_policy: ToolPolicy | None = None,
        oak_runtime: OakRuntime | None = None,
    ):
        self.server = FastMCP(name)
        self.tool_policy = tool_policy or ToolPolicy.guarded(
            audit_log_path=config.workspace_root / "audit" / "mcp-server-events.jsonl"
        )
        self.oak_runtime = oak_runtime or OakRuntime(
            session_id=self.tool_policy.session_id
        )
        self.oak_runtime.session_id = self.tool_policy.session_id
        self.tools: Dict[str, BaseTool] = {}

        # Initialize standard tools
        self.tools["bash"] = Bash()
        self.tools["editor"] = StrReplaceEditor()
        self.tools["terminate"] = Terminate()
        self.tools["oak_query"] = OakQuery(runtime=self.oak_runtime)

    async def execute_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        """Execute one registered server tool through the same monotonic guard."""

        tool = self.tools.get(tool_name)
        if tool is None:
            return f"Error: Unknown MCP server tool '{tool_name}'"
        argument_bytes = json.dumps(
            kwargs, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        argument_digest = hashlib.sha256(argument_bytes).hexdigest()
        if not self.tool_policy.allows(tool_name):
            try:
                self.tool_policy.record(
                    agent_name="MCPServer",
                    tool_name=tool_name,
                    decision="denied",
                    status="not_executed",
                    reason="not_granted_for_server_process",
                    arguments_sha256=argument_digest,
                )
            except OSError:
                pass
            return f"Error: MCP server tool '{tool_name}' is not authorized"

        try:
            decision = self.oak_runtime.authorize_tool(
                tool_name,
                kwargs,
                granted=True,
                tainted_sink_approved=self.tool_policy.allows_tainted(
                    tool_name, kwargs
                ),
                principal=self.tool_policy.principal,
                policy_version=self.tool_policy.policy_version,
            )
        except Exception as error:
            self.oak_runtime.record_gate_failure(
                tool_name,
                kwargs,
                error_type=type(error).__name__,
            )
            try:
                self.tool_policy.record(
                    agent_name="MCPServer",
                    tool_name=tool_name,
                    decision="denied",
                    status="not_executed",
                    reason="guard_error",
                    arguments_sha256=argument_digest,
                )
            except OSError:
                pass
            return (
                f"Error: MCP server tool '{tool_name}' was not executed: "
                "guard evaluation failed"
            )
        try:
            self.tool_policy.record_oak_guard(
                agent_name="MCPServer",
                tool_name=tool_name,
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
        except OSError:
            self.oak_runtime.record_gate_failure(
                tool_name,
                kwargs,
                error_type="GuardAuditUnavailable",
            )
            return f"Error: MCP server tool '{tool_name}' audit trail unavailable"
        if not decision.allowed:
            return (
                f"Error: MCP server tool '{tool_name}' requires exact approval "
                f"{approval_reference(tool_name, kwargs)}"
            )

        try:
            self.tool_policy.record(
                agent_name="MCPServer",
                tool_name=tool_name,
                decision="allowed",
                status="authorized",
                reason=decision.reason,
                arguments_sha256=argument_digest,
            )
        except OSError:
            self.oak_runtime.record_gate_failure(
                tool_name,
                kwargs,
                error_type="AuthorizationAuditUnavailable",
            )
            return f"Error: MCP server tool '{tool_name}' audit trail unavailable"

        effect = None
        if decision.profile.sink:
            try:
                effect = self.oak_runtime.begin_effect(
                    operation_id=f"mcp-server:{uuid4()}",
                    principal=self.tool_policy.principal,
                    tool_name=tool_name,
                    arguments=kwargs,
                    gate_event_id=decision.event_id,
                )
                self.tool_policy.record_effect_intent(
                    agent_name="MCPServer",
                    operation_id=effect.operation_id,
                    tool_name=tool_name,
                    arguments_sha256=effect.arguments_sha256,
                    idempotency_key=effect.idempotency_key,
                    gate_event_id=decision.event_id,
                )
                self.oak_runtime.dispatch_effect(effect.operation_id)
            except (OSError, BudgetExceeded, EffectConflict) as error:
                if effect is not None:
                    self.oak_runtime.observe_effect(
                        effect.operation_id, "failed_before_effect"
                    )
                return (
                    f"Error: MCP server tool '{tool_name}' was not executed: "
                    f"{type(error).__name__}"
                )

        effect_observed = False
        started_at = time.perf_counter()
        try:
            result = await tool.execute(**kwargs)
            status = "failed" if getattr(result, "error", None) else "succeeded"
            self.oak_runtime.record_tool_outcome(
                tool_name,
                status,
                "mcp_tool_result",
                operation_id=effect.operation_id if effect else None,
            )
            integrated = self.oak_runtime.integrate_tool_result(tool_name, str(result))
            try:
                self.tool_policy.record(
                    agent_name="MCPServer",
                    tool_name=tool_name,
                    decision="allowed",
                    status=status,
                    reason="mcp_tool_result",
                    arguments_sha256=argument_digest,
                    duration_ms=round((time.perf_counter() - started_at) * 1000),
                )
                self.tool_policy.record_provenance(
                    agent_name="MCPServer",
                    tool_name=tool_name,
                    content_sha256=integrated.provenance.content_sha256,
                    normalized_sha256=integrated.provenance.normalized_sha256,
                    carrier_type=integrated.provenance.carrier_type,
                    trust_tier=integrated.provenance.trust_tier,
                    taint_count=len(integrated.provenance.taint_ids),
                    transformations=integrated.provenance.transformations,
                )
                if effect is not None:
                    effect_status = "succeeded" if status == "succeeded" else "unknown"
                    self.tool_policy.record_effect_outcome(
                        agent_name="MCPServer",
                        operation_id=effect.operation_id,
                        tool_name=tool_name,
                        status=effect_status,
                        idempotency_key=effect.idempotency_key,
                    )
                    self.oak_runtime.observe_effect(
                        effect.operation_id, effect_status
                    )
                    effect_observed = True
            except OSError:
                if effect is not None and not effect_observed:
                    try:
                        self.oak_runtime.observe_effect(
                            effect.operation_id, "unknown"
                        )
                        effect_observed = True
                    except (EffectConflict, ValueError) as observation_error:
                        self.oak_runtime.record_gate_failure(
                            tool_name,
                            kwargs,
                            error_type=type(observation_error).__name__,
                        )
                return (
                    f"Error: MCP server tool '{tool_name}' may have executed; "
                    "outcome UNKNOWN, do not retry automatically"
                )
            return integrated.model_content
        except Exception as error:
            self.oak_runtime.record_tool_outcome(
                tool_name,
                "failed",
                type(error).__name__,
                operation_id=effect.operation_id if effect else None,
            )
            if effect is not None and not effect_observed:
                try:
                    self.oak_runtime.observe_effect(
                        effect.operation_id, "unknown"
                    )
                    effect_observed = True
                except (EffectConflict, ValueError) as observation_error:
                    self.oak_runtime.record_gate_failure(
                        tool_name,
                        kwargs,
                        error_type=type(observation_error).__name__,
                    )
                try:
                    self.tool_policy.record_effect_outcome(
                        agent_name="MCPServer",
                        operation_id=effect.operation_id,
                        tool_name=tool_name,
                        status="unknown",
                        idempotency_key=effect.idempotency_key,
                    )
                except OSError:
                    pass
            try:
                self.tool_policy.record(
                    agent_name="MCPServer",
                    tool_name=tool_name,
                    decision="allowed",
                    status="failed",
                    reason=type(error).__name__,
                    arguments_sha256=argument_digest,
                    duration_ms=round((time.perf_counter() - started_at) * 1000),
                )
            except OSError:
                pass
            return f"Error: MCP server tool '{tool_name}' failed with {type(error).__name__}"

    def register_tool(self, tool: BaseTool, method_name: Optional[str] = None) -> None:
        """Register a tool with parameter validation and documentation."""
        tool_name = method_name or tool.name
        tool_param = tool.to_param()
        tool_function = tool_param["function"]

        # Define the async function to be registered
        async def tool_method(**kwargs):
            logger.info(f"Executing governed MCP tool {tool_name}")
            return await self.execute_tool(tool_name, kwargs)

        # Set method metadata
        tool_method.__name__ = tool_name
        tool_method.__doc__ = self._build_docstring(tool_function)
        tool_method.__signature__ = self._build_signature(tool_function)

        # Store parameter schema (important for tools that access it programmatically)
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        tool_method._parameter_schema = {
            param_name: {
                "description": param_details.get("description", ""),
                "type": param_details.get("type", "any"),
                "required": param_name in required_params,
            }
            for param_name, param_details in param_props.items()
        }

        # Register with server
        self.server.tool()(tool_method)
        logger.info(f"Registered tool: {tool_name}")

    def _build_docstring(self, tool_function: dict) -> str:
        """Build a formatted docstring from tool function metadata."""
        description = tool_function.get("description", "")
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])

        # Build docstring (match original format)
        docstring = description
        if param_props:
            docstring += "\n\nParameters:\n"
            for param_name, param_details in param_props.items():
                required_str = (
                    "(required)" if param_name in required_params else "(optional)"
                )
                param_type = param_details.get("type", "any")
                param_desc = param_details.get("description", "")
                docstring += (
                    f"    {param_name} ({param_type}) {required_str}: {param_desc}\n"
                )

        return docstring

    def _build_signature(self, tool_function: dict) -> Signature:
        """Build a function signature from tool function metadata."""
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])

        parameters = []

        # Follow original type mapping
        for param_name, param_details in param_props.items():
            param_type = param_details.get("type", "")
            default = Parameter.empty if param_name in required_params else None

            # Map JSON Schema types to Python types (same as original)
            annotation = Any
            if param_type == "string":
                annotation = str
            elif param_type == "integer":
                annotation = int
            elif param_type == "number":
                annotation = float
            elif param_type == "boolean":
                annotation = bool
            elif param_type == "object":
                annotation = dict
            elif param_type == "array":
                annotation = list

            # Create parameter with same structure as original
            param = Parameter(
                name=param_name,
                kind=Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
            parameters.append(param)

        return Signature(parameters=parameters)

    async def cleanup(self) -> None:
        """Clean up server resources."""
        logger.info("Cleaning up resources")

    def register_all_tools(self) -> None:
        """Register all tools with the server."""
        for method_name, tool in self.tools.items():
            if self.tool_policy.allows(method_name):
                self.register_tool(tool, method_name=method_name)

    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server."""
        # Register all tools
        self.register_all_tools()

        # Register cleanup function (match original behavior)
        atexit.register(lambda: asyncio.run(self.cleanup()))

        # Start server (with same logging as original)
        logger.info(f"Starting OpenManus server ({transport} mode)")
        self.server.run(transport=transport)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenManus MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Communication method: stdio or http (default: stdio)",
    )
    parser.add_argument(
        "--allow-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Authorize one exact MCP server tool for this process.",
    )
    parser.add_argument(
        "--allow-tainted-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Grant a sensitive server tool; exact calls still need approval.",
    )
    parser.add_argument(
        "--approve-tainted-operation",
        action="append",
        default=[],
        metavar="TOOL:SHA256",
        help="Approve one exact canonical argument set for a tainted sink.",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=config.workspace_root / "audit" / "mcp-server-events.jsonl",
        help="Metadata-only MCP server audit path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create and run server (maintaining original flow)
    policy = ToolPolicy.guarded(
        allowed_tools=args.allow_tool,
        allowed_tainted_tools=args.allow_tainted_tool,
        approved_operations=args.approve_tainted_operation,
        audit_log_path=args.audit_log,
    )
    server = MCPServer(tool_policy=policy)
    server.run(transport=args.transport)
