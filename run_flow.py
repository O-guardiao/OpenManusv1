import argparse
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Sequence

from app.agent.data_analysis import DataAnalysis
from app.agent.manus import Manus
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.tool.policy import ToolPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenManus planning flow")
    parser.add_argument("--prompt", help="Input prompt for the flow")
    parser.add_argument(
        "--allow-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Authorize one exact Manus tool name for this process.",
    )
    parser.add_argument(
        "--allow-tainted-tool",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Separately approve one sensitive Manus tool after untrusted data "
            "enters this process."
        ),
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=config.workspace_root / "audit" / "tool-events.jsonl",
        help="Metadata-only JSONL audit path.",
    )
    return parser


async def run_flow(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    prompt = args.prompt if args.prompt else input("Enter your prompt: ")
    if not prompt.strip():
        logger.warning("Empty prompt provided.")
        return

    policy = ToolPolicy.guarded(
        allowed_tools=args.allow_tool,
        allowed_tainted_tools=args.allow_tainted_tool,
        audit_log_path=args.audit_log,
    )
    request_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    started_at = time.perf_counter()
    policy.record_session(
        agent_name="PlanningFlow",
        status="started",
        request_sha256=request_sha256,
        reason="operator_grants_captured",
    )
    agents = {}
    try:
        agents = {
            "manus": Manus(tool_policy=policy),
        }
        if config.run_flow_config.use_data_analysis_agent:
            agents["data_analysis"] = DataAnalysis(tool_policy=policy)

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(f"Flow returned {len(result)} characters")
            policy.record_session(
                agent_name="PlanningFlow",
                status=flow.terminal_status,
                request_sha256=request_sha256,
                reason=f"flow_{flow.terminal_status}",
                duration_ms=round((time.perf_counter() - started_at) * 1000),
            )
        except asyncio.TimeoutError:
            policy.record_session(
                agent_name="PlanningFlow",
                status="timed_out",
                request_sha256=request_sha256,
                reason="one_hour_deadline",
                duration_ms=round((time.perf_counter() - started_at) * 1000),
            )
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        policy.record_session(
            agent_name="PlanningFlow",
            status="interrupted",
            request_sha256=request_sha256,
            reason="operator_interrupt",
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        logger.info("Operation cancelled by user.")
    except Exception as error:
        policy.record_session(
            agent_name="PlanningFlow",
            status="failed",
            request_sha256=request_sha256,
            reason=type(error).__name__,
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        logger.error(f"Planning flow failed with {type(error).__name__}")
    finally:
        for agent in agents.values():
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_flow())
