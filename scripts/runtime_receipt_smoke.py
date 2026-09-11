#!/usr/bin/env python3
"""Create one offline, replayable OaK runtime receipt for cross-language checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from app.cockpit.session import AgentRunRecorder
from app.cockpit.store import CockpitStore
from app.oak.runtime import OakRuntime


def run(output_directory: Path) -> dict[str, object]:
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    database = output_directory / "runtime-smoke.db"
    prompt = "offline runtime conformance smoke"
    result = "offline receipt generated"

    runtime = OakRuntime()
    runtime.begin_task(prompt)
    runtime.consume_budget("steps", cycle_id="smoke")
    runtime.record_provider_state("fixture-provider", "opening")
    runtime.record_provider_state(
        "fixture-provider", "active", published_tool_count=1
    )
    runtime.record_provider_state("fixture-provider", "draining")
    runtime.record_provider_state("fixture-provider", "absent")
    runtime.record_final_response(result)
    terminal_status = runtime.finish_task("completed")

    store = CockpitStore(database)
    recorder, created = AgentRunRecorder.admit(
        store,
        prompt=prompt,
        agent_name="OfflineSmoke",
        allowed_tools=[],
        allowed_tainted_tools=[],
        request_key=f"offline-runtime-smoke:{uuid4()}",
    )
    if not created:
        raise RuntimeError("offline smoke request unexpectedly deduplicated")
    recorder.start()
    task = recorder.finish(
        oak_trace=runtime.trace,
        oak_evidence_count=runtime.evidence_count,
        terminal_status=terminal_status,
        result=result,
        duration_ms=0,
    )
    document = store.export_task(task["id"])
    document["verification"] = store.verify_task(task["id"]).to_dict()
    export_path = output_directory / f"runtime-receipt-{task['id']}.json"
    export_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    conformance = runtime.conformance_report()
    return {
        "task_id": task["id"],
        "task_status": task["status"],
        "runtime_valid": conformance.valid,
        "runtime_event_count": conformance.event_count,
        "cockpit_valid": document["verification"]["valid"],
        "database": str(database),
        "export": str(export_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    report = run(args.output_directory)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(
        report["task_status"] != "completed"
        or not report["runtime_valid"]
        or not report["cockpit_valid"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
