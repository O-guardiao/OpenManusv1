"""Adapter that turns a real Manus run into a content-safe cockpit receipt."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from app.cockpit.store import CockpitStore
from app.oak.formal_runtime import RUNTIME_CONTRACT_VERSION, verify_runtime_trace


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


_FORBIDDEN_RUNTIME_KEYS = {
    "arguments",
    "content",
    "error_message",
    "output",
    "prompt",
    "request",
    "result",
}


def _assert_metadata_only(value: Any, *, path: str = "trace") -> None:
    """Fail closed if a future runtime event attempts to persist raw content."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_RUNTIME_KEYS:
                raise ValueError(f"raw content field is forbidden in {path}: {key}")
            _assert_metadata_only(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_metadata_only(item, path=f"{path}[{index}]")


@dataclass
class AgentRunRecorder:
    store: CockpitStore
    task_id: str
    request_sha256: str

    @classmethod
    def admit(
        cls,
        store: CockpitStore,
        *,
        prompt: str,
        agent_name: str,
        allowed_tools: Iterable[str],
        allowed_tainted_tools: Iterable[str],
        approved_operations: Iterable[str] = (),
        request_key: str,
    ) -> tuple["AgentRunRecorder", bool]:
        request_sha256 = _sha256(prompt)
        subject = {
            "kind": "agent_run",
            "agent": agent_name,
            "request_sha256": request_sha256,
            "allowed_tools": sorted(set(allowed_tools)),
            "allowed_tainted_tools": sorted(set(allowed_tainted_tools)),
            "approved_operations": sorted(set(approved_operations)),
        }
        task, created = store.admit_task(
            goal="Execute one governed Manus task and issue a portable receipt",
            subject=subject,
            request_key=request_key,
        )
        return cls(store, task["id"], request_sha256), created

    def start(self) -> dict[str, Any]:
        self.store.transition(
            self.task_id,
            kind="task.accepted",
            state_after="accepted",
            payload={"policy": "intrinsic-oak-tool-policy-v1"},
        )
        return self.store.transition(
            self.task_id,
            kind="run.started",
            state_after="running",
            payload={"executor": "Manus", "content_storage": "hashes_only"},
        )

    def _receipt(
        self,
        *,
        oak_trace: Iterable[Mapping[str, Any]],
        oak_evidence_count: int,
        terminal_status: str,
        result: str,
        duration_ms: int,
    ) -> dict[str, Any]:
        trace = [dict(event) for event in oak_trace]
        _assert_metadata_only(trace)
        conformance = verify_runtime_trace(trace)
        event_types = Counter(str(event.get("event_type", "unknown")) for event in trace)
        task_starts = [
            event for event in trace if event.get("event_type") == "task_started"
        ]
        final_claims = [
            event for event in trace if event.get("event_type") == "final_evidence"
        ]
        result_sha256 = _sha256(result)
        return {
            "request_sha256": self.request_sha256,
            "result_sha256": result_sha256,
            "result_characters": len(result),
            "oak_trace_sha256": _sha256(_canonical(trace)),
            "oak_trace": trace,
            "oak_event_count": len(trace),
            "oak_event_types": dict(sorted(event_types.items())),
            "oak_evidence_count": int(oak_evidence_count),
            "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
            "formal_conformance": conformance.to_dict(),
            "evidence_count_agrees": (
                int(oak_evidence_count) == conformance.evidence_count
            ),
            "terminal_status_agrees": (
                terminal_status == conformance.terminal_status
            ),
            "request_digest_agrees": (
                len(task_starts) == 1
                and task_starts[0].get("request_sha256") == self.request_sha256
            ),
            "result_digest_agrees": any(
                event.get("content_sha256") == result_sha256
                for event in final_claims
            ),
            "terminal_status": terminal_status,
            "duration_ms": int(duration_ms),
            "content_storage": "hashes_only",
        }

    def finish(
        self,
        *,
        oak_trace: Iterable[Mapping[str, Any]],
        oak_evidence_count: int,
        terminal_status: str,
        result: str,
        duration_ms: int,
    ) -> dict[str, Any]:
        receipt = self._receipt(
            oak_trace=oak_trace,
            oak_evidence_count=oak_evidence_count,
            terminal_status=terminal_status,
            result=result,
            duration_ms=duration_ms,
        )
        self.store.record_evidence(
            self.task_id,
            evidence_type="formal.runtime.conformance.v1",
            payload=receipt,
        )
        conformance = receipt["formal_conformance"]
        if (
            terminal_status == "completed"
            and conformance["valid"]
            and receipt["evidence_count_agrees"]
            and receipt["terminal_status_agrees"]
            and receipt["request_digest_agrees"]
            and receipt["result_digest_agrees"]
        ):
            return self.store.complete_task(
                self.task_id,
                {"decision": "agent_run_completed", "receipt": "hashes_only"},
            )
        return self.store.fail_task(
            self.task_id,
            {
                "reason": (
                    terminal_status
                    if terminal_status != "completed"
                    else "runtime_conformance_failed"
                ),
                "receipt": "metadata_only_replayable_trace",
            },
        )

    def fail(
        self,
        *,
        terminal_status: str,
        duration_ms: int,
        error_type: str,
        oak_trace: Iterable[Mapping[str, Any]] = (),
        oak_evidence_count: int = 0,
    ) -> dict[str, Any]:
        task = self.store.get_task(self.task_id)
        if task["status"] in {"completed", "failed"}:
            return task
        trace = list(oak_trace)
        if task["status"] == "running" and trace:
            self.store.record_evidence(
                self.task_id,
                evidence_type="formal.runtime.conformance.v1",
                payload=self._receipt(
                    oak_trace=trace,
                    oak_evidence_count=oak_evidence_count,
                    terminal_status=terminal_status,
                    result="",
                    duration_ms=duration_ms,
                ),
            )
        return self.store.fail_task(
            self.task_id,
            {
                "reason": terminal_status,
                "error_type": error_type,
                "duration_ms": int(duration_ms),
                "content_storage": "hashes_only",
            },
        )
