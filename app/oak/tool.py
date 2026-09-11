"""Read-only tool exposing the intrinsic kernel to the agent itself."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, Optional

from app.oak.runtime import OakRuntime
from app.tool.base import BaseTool, ToolResult


class OakQuery(BaseTool):
    name: str = "oak_query"
    description: str = (
        "Query the verified local ontology kernel, active provenance state, and "
        "evidence summary. This tool is read-only and cannot approve effects or "
        "modify the frozen kernel."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["status", "tool_profile", "patterns_for_tool"],
            },
            "tool_name": {
                "type": "string",
                "description": "Required for tool_profile and patterns_for_tool.",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    }
    runtime: Any

    async def execute(
        self,
        *,
        operation: Literal["status", "tool_profile", "patterns_for_tool"],
        tool_name: Optional[str] = None,
    ) -> ToolResult:
        runtime: OakRuntime = self.runtime
        if operation == "status":
            return self.success_response(runtime.status())
        if not tool_name:
            return self.fail_response(f"tool_name is required for {operation}")
        if operation == "tool_profile":
            return self.success_response(asdict(runtime.tool_profile(tool_name)))
        if operation == "patterns_for_tool":
            return self.success_response(
                runtime.query("patterns_for_tool", {"tool_name": tool_name})
            )
        return self.fail_response(f"unsupported OaK operation: {operation}")
