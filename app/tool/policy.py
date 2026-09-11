"""Session-scoped tool authorization and privacy-preserving audit records."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from app.oak.formal_runtime import approval_key, approval_reference


CONTROL_TOOLS = {"ask_human", "oak_query", "terminate"}


class ToolPolicy(BaseModel):
    """Decide which tools a session may use and record each decision.

    ToolCallAgent and its specialized agents use ``guarded`` by default, so host
    and MCP effects are denied unless the operator grants the exact tool name for
    that session. ``permissive`` exists only for explicit compatibility/testing.
    """

    default_allow: bool = True
    allowed_tools: set[str] = Field(default_factory=set)
    allowed_tainted_tools: set[str] = Field(default_factory=set)
    approved_operations: set[str] = Field(default_factory=set)
    audit_log_path: Optional[Path] = None
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    principal: str = "local-operator"
    policy_version: str = "intrinsic-oak-tool-policy-v2"

    _write_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @field_validator("allowed_tools", "allowed_tainted_tools", mode="before")
    @classmethod
    def normalize_tool_names(cls, value: Iterable[str] | None) -> set[str]:
        normalized = set()
        for name in value or ():
            clean_name = str(name).strip().lower()
            if not clean_name:
                raise ValueError("Tool grants must be non-empty names")
            normalized.add(clean_name)
        return normalized

    @field_validator("approved_operations", mode="before")
    @classmethod
    def normalize_operation_approvals(
        cls, value: Iterable[str] | None
    ) -> set[str]:
        normalized: set[str] = set()
        for item in value or ():
            reference = str(item).strip().lower()
            tool, separator, digest = reference.partition(":")
            if (
                not separator
                or not tool
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    "Exact approvals must use TOOL:CANONICAL_ARGUMENTS_SHA256"
                )
            normalized.add(f"{tool}:{digest}")
        return normalized

    @classmethod
    def permissive(cls) -> "ToolPolicy":
        return cls(default_allow=True)

    @classmethod
    def guarded(
        cls,
        *,
        allowed_tools: Iterable[str] | None = None,
        allowed_tainted_tools: Iterable[str] | None = None,
        approved_operations: Iterable[str] | None = None,
        audit_log_path: str | Path | None = None,
        principal: str = "local-operator",
    ) -> "ToolPolicy":
        tainted_grants = set(allowed_tainted_tools or ())
        grants = set(allowed_tools or ()) | tainted_grants | CONTROL_TOOLS
        return cls(
            default_allow=False,
            allowed_tools=grants,
            allowed_tainted_tools=tainted_grants,
            approved_operations=set(approved_operations or ()),
            audit_log_path=Path(audit_log_path) if audit_log_path else None,
            principal=principal,
        )

    def allows(self, tool_name: str) -> bool:
        return self.default_allow or tool_name.strip().lower() in self.allowed_tools

    def allows_tainted(
        self,
        tool_name: str,
        arguments: dict | None = None,
    ) -> bool:
        """Require approval bound to the exact canonical arguments.

        ``allowed_tainted_tools`` remains a capability grant for compatibility;
        it is deliberately insufficient authority for a tainted sink effect.
        """

        if self.default_allow:
            return True
        normalized = tool_name.strip().lower()
        if normalized not in self.allowed_tainted_tools:
            return False
        if arguments is None:
            return any(
                reference.startswith(f"{normalized}:")
                for reference in self.approved_operations
            )
        return approval_reference(normalized, arguments) in self.approved_operations

    def exact_approval_key(
        self,
        tool_name: str,
        arguments: dict,
        *,
        state_version: str,
    ) -> str:
        if not self.allows_tainted(tool_name, arguments):
            return ""
        return approval_key(
            principal=self.principal,
            tool_name=tool_name,
            arguments=arguments,
            state_version=state_version,
            policy_version=self.policy_version,
        )

    def record_oak_guard(
        self,
        *,
        agent_name: str,
        tool_name: str,
        allowed: bool,
        reason: str,
        source: bool,
        sink: bool,
        risk_score: int,
        taint_count: int,
        outcome: str = "",
        arguments_sha256: str = "",
        approval_key_value: str = "",
        state_version: str = "",
    ) -> None:
        """Persist the monotonic OaK pre-effect decision without payloads."""

        if self.audit_log_path is None:
            if self.default_allow:
                return
            raise OSError("guarded policies require an audit log destination")
        self._append_event(
            {
                "schema_version": 1,
                "event_type": "oak_guard",
                "event_id": str(uuid4()),
                "session_id": self.session_id,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "agent": agent_name,
                "tool": tool_name,
                "allowed": allowed,
                "reason": reason,
                "source": source,
                "sink": sink,
                "risk_score": risk_score,
                "taint_count": taint_count,
                "outcome": outcome or ("allow" if allowed else "reject"),
                "arguments_sha256": arguments_sha256,
                "approval_key": approval_key_value,
                "state_version": state_version,
                "policy_version": self.policy_version,
            }
        )

    def record_effect_intent(
        self,
        *,
        agent_name: str,
        operation_id: str,
        tool_name: str,
        arguments_sha256: str,
        idempotency_key: str,
        gate_event_id: str,
    ) -> None:
        """Append and fsync intent before a potentially external effect."""

        if self.audit_log_path is None:
            if self.default_allow:
                return
            raise OSError("guarded policies require an audit log destination")
        self._append_event(
            {
                "schema_version": 1,
                "event_type": "effect_intent",
                "event_id": str(uuid4()),
                "session_id": self.session_id,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "agent": agent_name,
                "operation_id": operation_id,
                "tool": tool_name,
                "arguments_sha256": arguments_sha256,
                "idempotency_key": idempotency_key,
                "gate_event_id": gate_event_id,
                "durability": "flush+fsync",
            }
        )

    def record_effect_outcome(
        self,
        *,
        agent_name: str,
        operation_id: str,
        tool_name: str,
        status: str,
        idempotency_key: str,
    ) -> None:
        if self.audit_log_path is None:
            if self.default_allow:
                return
            raise OSError("guarded policies require an audit log destination")
        self._append_event(
            {
                "schema_version": 1,
                "event_type": "effect_outcome",
                "event_id": str(uuid4()),
                "session_id": self.session_id,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "agent": agent_name,
                "operation_id": operation_id,
                "tool": tool_name,
                "status": status,
                "idempotency_key": idempotency_key,
            }
        )

    def record_provenance(
        self,
        *,
        agent_name: str,
        tool_name: str,
        content_sha256: str,
        normalized_sha256: str,
        carrier_type: str,
        trust_tier: str,
        taint_count: int,
        transformations: Iterable[str],
    ) -> None:
        """Persist model-boundary provenance without the observation text."""

        if self.audit_log_path is None:
            if self.default_allow:
                return
            raise OSError("guarded policies require an audit log destination")
        self._append_event(
            {
                "schema_version": 1,
                "event_type": "oak_provenance",
                "event_id": str(uuid4()),
                "session_id": self.session_id,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "agent": agent_name,
                "tool": tool_name,
                "content_sha256": content_sha256,
                "normalized_sha256": normalized_sha256,
                "carrier_type": carrier_type,
                "trust_tier": trust_tier,
                "taint_count": taint_count,
                "transformations": sorted(set(transformations)),
            }
        )

    def record(
        self,
        *,
        agent_name: str,
        tool_name: str,
        decision: str,
        status: str,
        arguments_sha256: str,
        reason: str,
        duration_ms: int | None = None,
    ) -> None:
        """Append an audit event without arguments, outputs, prompts, or secrets."""

        if self.audit_log_path is None:
            if self.default_allow:
                return
            raise OSError("guarded policies require an audit log destination")

        event = {
            "schema_version": 1,
            "event_type": "tool_decision",
            "event_id": str(uuid4()),
            "session_id": self.session_id,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "tool": tool_name,
            "decision": decision,
            "status": status,
            "reason": reason,
            "arguments_sha256": arguments_sha256,
        }
        if duration_ms is not None:
            event["duration_ms"] = duration_ms

        self._append_event(event)

    def record_session(
        self,
        *,
        agent_name: str,
        status: str,
        request_sha256: str,
        reason: str,
        duration_ms: int | None = None,
    ) -> None:
        """Record a task lifecycle event without persisting the request text."""

        if self.audit_log_path is None:
            if self.default_allow:
                return
            raise OSError("guarded policies require an audit log destination")

        event = {
            "schema_version": 1,
            "event_type": "session",
            "event_id": str(uuid4()),
            "session_id": self.session_id,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "status": status,
            "reason": reason,
            "request_sha256": request_sha256,
            "granted_tools": sorted(self.allowed_tools),
            "tainted_sink_grants": sorted(self.allowed_tainted_tools),
        }
        if duration_ms is not None:
            event["duration_ms"] = duration_ms

        self._append_event(event)

    def _append_event(self, event: dict) -> None:
        if self.audit_log_path is None:
            return

        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self._write_lock:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
