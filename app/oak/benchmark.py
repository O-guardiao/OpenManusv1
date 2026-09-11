"""Measure local OaK overhead without invoking an LLM or an external tool."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from app.oak.runtime import OakRuntime


def _milliseconds(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def run(iterations: int = 200) -> dict:
    if iterations < 10:
        raise ValueError("iterations must be at least 10")
    started = time.perf_counter()
    runtime = OakRuntime()
    initialization_ms = _milliseconds(started)

    started = time.perf_counter()
    runtime.tool_profile("python_execute")
    first_query_ms = _milliseconds(started)

    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        runtime.would_authorize_tool(
            "python_execute",
            granted=True,
            tainted_sink_approved=False,
        )
        samples.append(_milliseconds(started))
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "schema_version": 1,
        "iterations": iterations,
        "kernel_initialization_ms": round(initialization_ms, 3),
        "first_typed_query_ms": round(first_query_ms, 3),
        "cached_guard_mean_ms": round(statistics.fmean(samples), 6),
        "cached_guard_p95_ms": round(ordered[p95_index], 6),
        "external_model_calls": 0,
        "external_tool_calls": 0,
        "marginal_api_cost_usd": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.iterations), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
