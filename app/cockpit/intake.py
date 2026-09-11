"""Deterministic repository intake for the reverse-engineering cockpit."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.cockpit.store import CockpitStore
from app.oak.kernel import verify_frozen_kernel
from app.oak.runtime import DEFAULT_KERNEL_ROOT


MANIFEST_NAMES = {
    "cargo.toml",
    "go.mod",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
}


class RepositoryInspectionError(RuntimeError):
    """Raised when a repository cannot be inspected reproducibly."""


def _sanitize_remote_url(value: str) -> str:
    value = value.strip()
    if "://" in value:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc += f":{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    if "@" in value and ":" in value.split("@", 1)[1]:
        return value.split("@", 1)[1]
    return value


def _git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown git error"
        raise RepositoryInspectionError(
            f"git {' '.join(arguments)} failed for {repository}: {detail}"
        )
    return process.stdout


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        timeout=20,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).decode("utf-8", errors="replace").strip()
        raise RepositoryInspectionError(
            f"git {' '.join(arguments)} failed for {repository}: {detail or 'unknown git error'}"
        )
    return process.stdout


def _evidence_files(tracked: list[str]) -> list[str]:
    candidates: list[str] = []
    for relative in tracked:
        if "/" in relative:
            continue
        upper_name = relative.upper()
        lower_name = relative.lower()
        if (
            upper_name.startswith("LICENSE")
            or upper_name.startswith("COPYING")
            or upper_name.startswith("NOTICE")
            or upper_name.startswith("PROVENANCE")
            or lower_name in MANIFEST_NAMES
        ):
            candidates.append(relative)
    return sorted(candidates, key=str.casefold)


def inspect_repository(repository_path: str | Path) -> dict[str, Any]:
    """Return a content-addressed snapshot without importing source code."""

    started_at = time.perf_counter()
    repository = Path(repository_path).resolve(strict=True)
    if not repository.is_dir():
        raise RepositoryInspectionError(f"not a directory: {repository}")

    commit = _git(repository, "rev-parse", "HEAD").strip()
    origin_process = subprocess.run(
        ["git", "-C", str(repository), "remote", "get-url", "origin"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    origin = (
        _sanitize_remote_url(origin_process.stdout)
        if origin_process.returncode == 0
        else ""
    )

    tracked_output = _git(repository, "ls-tree", "-r", "--name-only", "-z", "HEAD")
    tracked = [item for item in tracked_output.split("\0") if item]
    extensions = Counter()
    for relative in tracked:
        suffix = Path(relative).suffix.lower()
        extensions[suffix or "<none>"] += 1

    evidence_files = _evidence_files(tracked)
    evidence_hashes = {
        relative: hashlib.sha256(
            _git_bytes(repository, "show", f"HEAD:{relative}")
        ).hexdigest()
        for relative in evidence_files
    }
    license_files = [
        name
        for name in evidence_hashes
        if name.upper().startswith(("LICENSE", "COPYING"))
    ]
    notice_files = [
        name
        for name in evidence_hashes
        if name.upper().startswith(("NOTICE", "PROVENANCE"))
    ]
    kernel_report = verify_frozen_kernel(DEFAULT_KERNEL_ROOT)

    warnings: list[str] = []
    if not license_files:
        warnings.append(
            "No root source license was found; reuse must remain clean-room until rights are established."
        )
    if not origin:
        warnings.append("No origin remote was available in the inspected clone.")

    return {
        "scanner_version": "repository-head-v2",
        "snapshot_scope": "git_commit_HEAD",
        "repository": str(repository),
        "origin": origin,
        "commit": commit,
        "tracked_files": len(tracked),
        "extension_counts": dict(sorted(extensions.items())),
        "evidence_file_sha256": evidence_hashes,
        "license_files": license_files,
        "notice_and_provenance_files": notice_files,
        "reuse_policy": "licensed_source" if license_files else "clean_room_only",
        "warnings": warnings,
        "measurement": {
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "llm_calls": 0,
            "llm_cost_usd": 0.0,
        },
        "oak_kernel": {
            "verified": kernel_report.get("status") == "verified",
            "assurance_level": kernel_report.get("assurance_level"),
            "bundle_sha256": kernel_report.get("bundle_sha256"),
        },
    }


class RepositoryIntakeService:
    """Orchestrate one repository inspection as a durable task trace."""

    GOAL = "Inspect a repository and produce a governed clean-room reuse decision"

    def __init__(self, store: CockpitStore):
        self.store = store

    def run(
        self, repository_path: str | Path, *, request_key: str
    ) -> dict[str, Any]:
        repository = Path(repository_path).resolve(strict=True)
        task, created = self.store.admit_task(
            goal=self.GOAL,
            subject={"kind": "repository", "path": str(repository)},
            request_key=request_key,
        )
        if not created:
            return task

        try:
            self.store.transition(
                task["id"],
                kind="task.accepted",
                state_after="accepted",
                payload={"policy": "clean-room-contract-extraction-v1"},
            )
            self.store.transition(
                task["id"],
                kind="run.started",
                state_after="running",
                payload={"scanner": "deterministic-git-intake-v1"},
            )
            snapshot = inspect_repository(repository)
            self.store.record_evidence(
                task["id"],
                evidence_type="repository.snapshot",
                payload=snapshot,
            )
            return self.store.complete_task(
                task["id"],
                {
                    "decision": snapshot["reuse_policy"],
                    "kernel_assurance": snapshot["oak_kernel"]["assurance_level"],
                },
            )
        except Exception as exc:
            current = self.store.get_task(task["id"])
            if current["status"] not in {"completed", "failed"}:
                self.store.fail_task(
                    task["id"],
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            raise
