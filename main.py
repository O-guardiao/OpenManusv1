import argparse
import asyncio
from contextlib import nullcontext
import hashlib
import time
from pathlib import Path
from typing import Sequence
from uuid import UUID, uuid4

from app.agent.manus import Manus
from app.cockpit.session import AgentRunRecorder
from app.cockpit.store import CockpitStore
from app.config import config
from app.harness.control import ControlInbox
from app.harness.runtime import RunObserver
from app.harness.session import SessionStore
from app.harness.verification import AcceptanceSpec
from app.llm import LLM
from app.logger import logger
from app.schema import Memory
from app.tool.policy import ToolPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Manus agent with a prompt")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    parser.add_argument(
        "--allow-tool",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Authorize one exact tool name for this session. Repeat for multiple "
            "tools; ask_human and terminate are always available."
        ),
    )
    parser.add_argument(
        "--allow-tainted-tool",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Grant one sensitive tool capability. This does not approve any "
            "particular call after untrusted data enters the session."
        ),
    )
    parser.add_argument(
        "--approve-tainted-operation",
        action="append",
        default=[],
        metavar="TOOL:SHA256",
        help=(
            "Approve one exact canonical argument set for a tainted sink. "
            "Use the reference returned by a denied call; repeat as needed."
        ),
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=config.workspace_root / "audit" / "tool-events.jsonl",
        help="Metadata-only JSONL audit path.",
    )
    parser.add_argument(
        "--cockpit-db",
        type=Path,
        default=config.workspace_root / "cockpit" / "openmanus.db",
        help="SQLite database for the portable task receipt.",
    )
    parser.add_argument(
        "--cockpit-request-key",
        help="Stable idempotency key. A unique key is generated when omitted.",
    )
    parser.add_argument(
        "--no-cockpit",
        action="store_true",
        help="Disable the intrinsic SQLite task receipt for this run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override the agent step limit; zero is useful for a no-model smoke run.",
    )
    parser.add_argument(
        "--session", metavar="new|UUID",
        help="Opt in to storing conversation content in --cockpit-db; resume by UUID.",
    )
    parser.add_argument("--stream", action="store_true", help="Display model text as it arrives.")
    parser.add_argument("--events", type=Path, help="Optional JSONL event file; streamed text is included.")
    parser.add_argument("--context-tokens", type=int, default=32768, help="Model input context budget.")
    parser.add_argument("--acceptance", type=Path, help="Declarative JSON artifact acceptance specification.")
    parser.add_argument("--artifact-root", type=Path, default=config.workspace_root,
                        help="Root directory for acceptance artifacts.")
    return parser


async def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps is not None and args.max_steps < 0:
        raise ValueError("--max-steps must be zero or greater")
    if args.context_tokens < 1:
        raise ValueError("--context-tokens must be greater than zero")
    if args.session is not None and args.session != "new":
        if str(UUID(args.session)) != args.session:
            raise ValueError("--session must be new or a canonical UUID")
    acceptance = (
        AcceptanceSpec.from_file(args.acceptance, args.artifact_root)
        if args.acceptance else None
    )
    policy = ToolPolicy.guarded(
        allowed_tools=args.allow_tool,
        allowed_tainted_tools=args.allow_tainted_tool,
        approved_operations=args.approve_tainted_operation,
        audit_log_path=args.audit_log,
    )
    prompt = args.prompt if args.prompt else input("Enter your prompt: ")
    if not prompt.strip():
        logger.warning("Empty prompt provided.")
        return 2

    store = SessionStore(args.cockpit_db) if args.session else None
    session_id = store.create() if store and args.session == "new" else args.session
    if store:
        print(f"Session: {session_id}", flush=True)
    run_id = str(uuid4())
    terminal_status = "failed"
    run_started = False
    memory = None
    with store.lease(session_id) if store else nullcontext():
        try:
            if store:
                store.begin_run(session_id, run_id)
                run_started = True
            memory = Memory(messages=store.load(session_id) if store else [])
            observer = RunObserver(
                memory, store=store, session_id=session_id, event_path=args.events,
                stream_output=args.stream,
                inbox=ControlInbox(args.cockpit_db, session_id) if store else None,
                run_id=run_id,
            )
            memory.set_listener(observer.on_message)
            if store:
                observer.repair_incomplete_turn()
            exit_code, terminal_status = await _execute_run(
                args, policy, prompt, observer, acceptance
            )
            return exit_code
        finally:
            if memory is not None:
                memory.set_listener(None)
            if store and run_started:
                store.finish_run(session_id, terminal_status)


async def _execute_run(args, policy, prompt, observer, acceptance) -> tuple[int, str]:
    """Execute under the optional conversation lease; retain OaK and receipt gates."""

    request_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    started_at = time.perf_counter()
    recorder = None
    if not args.no_cockpit:
        request_key = args.cockpit_request_key or f"manus:{uuid4()}"
        recorder, created = AgentRunRecorder.admit(
            CockpitStore(args.cockpit_db),
            prompt=prompt,
            agent_name="Manus",
            allowed_tools=args.allow_tool,
            allowed_tainted_tools=args.allow_tainted_tool,
            approved_operations=args.approve_tainted_operation,
            request_key=request_key,
        )
        if not created:
            policy.record_session(
                agent_name="Manus",
                status="deduplicated",
                request_sha256=request_sha256,
                reason=f"cockpit_task_{recorder.task_id}",
            )
            logger.warning(
                f"Request already recorded as cockpit task {recorder.task_id}; skipped."
            )
            prior_task = recorder.store.get_task(recorder.task_id)
            status = prior_task["status"]
            return (0 if status == "completed" else 1), status
        recorder.start()
        logger.info(f"Cockpit task: {recorder.task_id}")
    policy.record_session(
        agent_name="Manus",
        status="started",
        request_sha256=request_sha256,
        reason="operator_grants_captured",
    )
    agent = None
    exit_code = 1
    terminal_status = "failed"
    cleanup_attempted = False
    try:
        agent_options = {
            "tool_policy": policy,
            "defer_oak_finish": True,
            "oak_task_request": prompt,
            "memory": observer.memory,
            "run_observer": observer,
            "stream_responses": args.stream,
            "context_window_tokens": args.context_tokens,
        }
        if args.max_steps is not None:
            agent_options["max_steps"] = args.max_steps
        if args.max_steps == 0:
            # A bounded lifecycle smoke must not initialize a provider or tokenizer.
            agent_options["llm"] = object.__new__(LLM)
        agent = await Manus.create(**agent_options)
        logger.warning("Processing your request...")
        result = await observer.supervise(agent.run(prompt))
        cleanup_error_type = ""
        try:
            await agent.cleanup()
        except Exception as cleanup_error:
            cleanup_error_type = type(cleanup_error).__name__
            terminal_status = "failed_cleanup"
            logger.error(
                "Agent resources did not close cleanly: "
                f"{cleanup_error_type}"
            )
        finally:
            cleanup_attempted = True
        acceptance_failed = False
        if acceptance is not None:
            report = acceptance.validate()
            observer.emit("acceptance_checked", **report)
            acceptance_failed = not report["passed"]
            if acceptance_failed:
                logger.error("Artifact acceptance checks failed.")
        terminal_override = (
            "failed_cleanup" if cleanup_error_type else "failed" if acceptance_failed else None
        )
        if hasattr(agent, "finalize_oak_task"):
            terminal_status = agent.finalize_oak_task(terminal_override)
        else:
            terminal_status = (
                terminal_override
                if terminal_override
                else agent.oak_runtime.last_task_status or "returned"
            )
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        if recorder is not None:
            recorder.finish(
                oak_trace=agent.oak_runtime.trace,
                oak_evidence_count=agent.oak_runtime.evidence_count,
                terminal_status=terminal_status,
                result=result,
                duration_ms=duration_ms,
            )
        policy.record_session(
            agent_name=agent.name,
            status=terminal_status,
            request_sha256=request_sha256,
            reason=f"agent_run_{terminal_status}",
            duration_ms=duration_ms,
        )
        if terminal_status == "completed":
            logger.info("Request processing completed.")
        else:
            logger.warning(f"Request ended with status: {terminal_status}.")
        final_text = next((
            message.content for message in reversed(observer.memory.messages)
            if message.role == "assistant" and not message.tool_calls and message.content
        ), result)
        if args.stream and observer.streamed_text:
            print()
        elif final_text:
            print(final_text, flush=True)
        observer.emit("run_finished", terminal_status=terminal_status)
        exit_code = 0 if terminal_status == "completed" else 1
    except (KeyboardInterrupt, asyncio.CancelledError) as interruption:
        cleanup_error_type = ""
        if agent is not None and not cleanup_attempted:
            try:
                await agent.cleanup()
            except Exception as cleanup_error:
                cleanup_error_type = type(cleanup_error).__name__
                logger.error(
                    "Cleanup after interruption failed: "
                    f"{cleanup_error_type}"
                )
            finally:
                cleanup_attempted = True
        if (
            agent is not None
            and hasattr(agent, "finalize_oak_task")
            and agent.oak_runtime.task_active
        ):
            agent.finalize_oak_task(
                "failed_cleanup" if cleanup_error_type else "interrupted"
            )
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        interrupted_status = (
            "interrupted_cleanup_failed"
            if cleanup_error_type
            else "interrupted"
        )
        if recorder is not None:
            recorder.fail(
                terminal_status=interrupted_status,
                duration_ms=duration_ms,
                error_type=cleanup_error_type or type(interruption).__name__,
                oak_trace=agent.oak_runtime.trace if agent else (),
                oak_evidence_count=agent.oak_runtime.evidence_count if agent else 0,
            )
        policy.record_session(
            agent_name=agent.name if agent else "Manus",
            status=interrupted_status,
            request_sha256=request_sha256,
            reason="operator_interrupt",
            duration_ms=duration_ms,
        )
        logger.warning("Operation interrupted.")
        terminal_status = interrupted_status
        observer.emit("run_finished", terminal_status=terminal_status)
        exit_code = 130
    except Exception as error:
        cleanup_error_type = ""
        if agent is not None and not cleanup_attempted:
            try:
                await agent.cleanup()
            except Exception as cleanup_error:
                cleanup_error_type = type(cleanup_error).__name__
                logger.error(
                    "Cleanup after run failure also failed: "
                    f"{cleanup_error_type}"
                )
            finally:
                cleanup_attempted = True
        if (
            agent is not None
            and hasattr(agent, "finalize_oak_task")
            and agent.oak_runtime.task_active
        ):
            agent.finalize_oak_task(
                "failed_cleanup" if cleanup_error_type else "failed"
            )
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        failed_status = "failed_cleanup" if cleanup_error_type else "failed"
        terminal_status = failed_status
        if recorder is not None:
            recorder.fail(
                terminal_status=failed_status,
                duration_ms=duration_ms,
                error_type=(
                    f"{type(error).__name__}+{cleanup_error_type}"
                    if cleanup_error_type
                    else type(error).__name__
                ),
                oak_trace=agent.oak_runtime.trace if agent else (),
                oak_evidence_count=agent.oak_runtime.evidence_count if agent else 0,
            )
        policy.record_session(
            agent_name=agent.name if agent else "Manus",
            status=failed_status,
            request_sha256=request_sha256,
            reason="unhandled_exception",
            duration_ms=duration_ms,
        )
        raise
    finally:
        if agent is not None and not cleanup_attempted:
            try:
                await agent.cleanup()
            except Exception as cleanup_error:
                logger.error(
                    "Final cleanup failed visibly: "
                    f"{type(cleanup_error).__name__}"
                )
    return exit_code, terminal_status


def cli() -> None:
    """Synchronous console-script entrypoint for the packaged async runner."""
    raise SystemExit(asyncio.run(main()))


if __name__ == "__main__":
    cli()
