#!/usr/bin/env python
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Iterable

from app.agent.mcp import MCPAgent
from app.config import config
from app.llm import LLM
from app.logger import logger
from app.tool.policy import ToolPolicy


class MCPRunner:
    """Runner class for MCP Agent with proper path handling and configuration."""

    def __init__(
        self,
        *,
        allowed_tools: Iterable[str] = (),
        allowed_tainted_tools: Iterable[str] = (),
        approved_operations: Iterable[str] = (),
        audit_log: Path | None = None,
        server_audit_log: Path | None = None,
        max_steps: int | None = None,
    ):
        self.root_path = config.root_path
        self.server_reference = config.mcp_config.server_reference
        self.allowed_tools = list(allowed_tools)
        self.allowed_tainted_tools = list(allowed_tainted_tools)
        self.approved_operations = list(approved_operations)
        self.server_audit_log = server_audit_log or (
            config.workspace_root / "audit" / "mcp-server-events.jsonl"
        )
        self.connection_type = "stdio"
        self.server_url: str | None = None
        self.last_terminal_status = ""
        policy = ToolPolicy.guarded(
            allowed_tools=self.allowed_tools,
            allowed_tainted_tools=self.allowed_tainted_tools,
            approved_operations=self.approved_operations,
            audit_log_path=(
                audit_log
                or config.workspace_root / "audit" / "mcp-agent-events.jsonl"
            ),
        )
        agent_options = {"tool_policy": policy}
        if max_steps is not None:
            agent_options["max_steps"] = max_steps
        if max_steps == 0:
            agent_options["llm"] = object.__new__(LLM)
        self.agent = MCPAgent(**agent_options)

    async def initialize(
        self,
        connection_type: str,
        server_url: str | None = None,
    ) -> None:
        """Initialize the MCP agent with the appropriate connection."""
        logger.info(f"Initializing MCPAgent with {connection_type} connection...")

        if connection_type == "stdio":
            server_args = ["-m", self.server_reference]
            for name in self.allowed_tools:
                server_args.extend(["--allow-tool", name])
            for name in self.allowed_tainted_tools:
                server_args.extend(["--allow-tainted-tool", name])
            for reference in self.approved_operations:
                server_args.extend(["--approve-tainted-operation", reference])
            server_args.extend(["--audit-log", str(self.server_audit_log)])
            await self.agent.initialize(
                connection_type="stdio",
                command=sys.executable,
                args=server_args,
                tool_name_prefix=False,
            )
        else:  # sse
            await self.agent.initialize(connection_type="sse", server_url=server_url)

        logger.info(f"Connected to MCP server via {connection_type}")

    async def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        print("\nMCP Agent Interactive Mode (type 'exit' to quit)\n")
        while True:
            user_input = input("\nEnter your request: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            response = await self._run_request(user_input)
            print(f"\nAgent: {response}")

    async def run_single_prompt(self, prompt: str) -> None:
        """Run the agent with a single prompt."""
        await self._run_request(prompt)

    async def run_default(self) -> None:
        """Run the agent in default mode."""
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await self._run_request(prompt)
        if self.last_terminal_status == "completed":
            logger.info("Request processing completed.")
        else:
            logger.warning(
                f"Request ended with status: {self.last_terminal_status}."
            )

    async def _run_request(self, prompt: str) -> str:
        """Own one provider/task lifecycle from opening through terminal."""

        self.agent.defer_oak_finish = True
        self.agent.start_owned_oak_task(prompt)
        response = ""
        run_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            await self.initialize(self.connection_type, self.server_url)
            response = await self.agent.run(prompt)
        except BaseException as error:
            run_error = error
        try:
            await self.agent.cleanup()
        except BaseException as error:
            cleanup_error = error
        self.last_terminal_status = self.agent.finalize_oak_task(
            "failed_cleanup"
            if cleanup_error
            else "failed" if run_error else None
        )
        if cleanup_error is not None:
            raise cleanup_error
        if run_error is not None:
            raise run_error
        return response

    async def cleanup(self) -> None:
        """Clean up agent resources."""
        if self.agent.mcp_clients.sessions:
            await self.agent.cleanup()
        if self.agent.oak_runtime.task_active:
            self.agent.finalize_oak_task("failed")
        logger.info("Session ended")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the MCP Agent")
    parser.add_argument(
        "--connection",
        "-c",
        choices=["stdio", "sse"],
        default="stdio",
        help="Connection type: stdio or sse",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/sse",
        help="URL for SSE connection",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument("--prompt", "-p", help="Single prompt to execute and exit")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override the request step budget; zero performs a no-model lifecycle smoke.",
    )
    parser.add_argument(
        "--allow-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Authorize the same exact tool in the local MCP agent and server.",
    )
    parser.add_argument(
        "--allow-tainted-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Grant a sensitive source-to-sink capability.",
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
        default=config.workspace_root / "audit" / "mcp-agent-events.jsonl",
        help="Metadata-only client-side audit path.",
    )
    parser.add_argument(
        "--server-audit-log",
        type=Path,
        default=config.workspace_root / "audit" / "mcp-server-events.jsonl",
        help="Metadata-only stdio server audit path.",
    )
    return parser.parse_args()


async def run_mcp() -> int:
    """Main entry point for the MCP runner."""
    args = parse_args()
    if args.max_steps is not None and args.max_steps < 0:
        raise ValueError("--max-steps must be zero or greater")
    runner = MCPRunner(
        allowed_tools=args.allow_tool,
        allowed_tainted_tools=args.allow_tainted_tool,
        approved_operations=args.approve_tainted_operation,
        audit_log=args.audit_log,
        server_audit_log=args.server_audit_log,
        max_steps=args.max_steps,
    )
    runner.connection_type = args.connection
    runner.server_url = args.server_url

    try:
        if args.prompt:
            await runner.run_single_prompt(args.prompt)
        elif args.interactive:
            await runner.run_interactive()
        else:
            await runner.run_default()

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error running MCPAgent: {str(e)}", exc_info=True)
        return 1
    finally:
        await runner.cleanup()
    return 0 if runner.last_terminal_status in {"", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_mcp()))
