import argparse
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Sequence

from app.agent.sandbox_agent import SandboxManus
from app.config import config
from app.logger import logger
from app.tool.policy import ToolPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run governed Daytona Manus")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    parser.add_argument(
        "--allow-tool",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Authorize one exact tool. sandbox_create is required before Daytona "
            "allocation."
        ),
    )
    parser.add_argument(
        "--allow-tainted-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Grant one sensitive tool; exact calls still need approval.",
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
        default=config.workspace_root / "audit" / "sandbox-tool-events.jsonl",
        help="Metadata-only JSONL audit path.",
    )
    return parser


async def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    prompt = args.prompt if args.prompt else input("Enter your prompt: ")
    if not prompt.strip():
        logger.warning("Empty prompt provided.")
        return

    policy = ToolPolicy.guarded(
        allowed_tools=args.allow_tool,
        allowed_tainted_tools=args.allow_tainted_tool,
        approved_operations=args.approve_tainted_operation,
        audit_log_path=args.audit_log,
    )
    request_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    started_at = time.perf_counter()
    policy.record_session(
        agent_name="SandboxManus",
        status="started",
        request_sha256=request_sha256,
        reason="operator_grants_captured",
    )

    agent = None
    cleanup_attempted = False
    try:
        agent = await SandboxManus.create(
            tool_policy=policy,
            defer_oak_finish=True,
            oak_task_request=prompt,
        )
        logger.warning("Processing your request...")
        await agent.run(prompt)
        cleanup_error = ""
        try:
            await agent.cleanup()
        except Exception as error:
            cleanup_error = type(error).__name__
        finally:
            cleanup_attempted = True
        terminal_status = agent.finalize_oak_task(
            "failed_cleanup" if cleanup_error else None
        )
        policy.record_session(
            agent_name=agent.name,
            status=terminal_status,
            request_sha256=request_sha256,
            reason=f"agent_run_{terminal_status}",
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        if terminal_status == "completed":
            logger.info("Request processing completed.")
        else:
            logger.warning(f"Request ended with status: {terminal_status}.")
    except KeyboardInterrupt:
        cleanup_error = ""
        if agent is not None and not cleanup_attempted:
            try:
                await agent.cleanup()
            except Exception as error:
                cleanup_error = type(error).__name__
            finally:
                cleanup_attempted = True
        terminal_status = "interrupted_cleanup_failed" if cleanup_error else "interrupted"
        if agent is not None and agent.oak_runtime.task_active:
            agent.finalize_oak_task(terminal_status)
        policy.record_session(
            agent_name=agent.name if agent else "SandboxManus",
            status=terminal_status,
            request_sha256=request_sha256,
            reason="operator_interrupt",
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        logger.warning("Operation interrupted.")
    except Exception as error:
        cleanup_error = ""
        if agent is not None and not cleanup_attempted:
            try:
                await agent.cleanup()
            except Exception as cleanup_exception:
                cleanup_error = type(cleanup_exception).__name__
            finally:
                cleanup_attempted = True
        terminal_status = "failed_cleanup" if cleanup_error else "failed"
        if agent is not None and agent.oak_runtime.task_active:
            agent.finalize_oak_task(terminal_status)
        policy.record_session(
            agent_name=agent.name if agent else "SandboxManus",
            status=terminal_status,
            request_sha256=request_sha256,
            reason=type(error).__name__,
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        raise
    finally:
        if agent is not None and not cleanup_attempted:
            try:
                await agent.cleanup()
            except Exception as cleanup_error:
                logger.error(
                    "Final sandbox cleanup failed: "
                    f"{type(cleanup_error).__name__}"
                )


if __name__ == "__main__":
    asyncio.run(main())
