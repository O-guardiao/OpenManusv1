from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from app.cockpit.intake import RepositoryIntakeService, inspect_repository
from app.cockpit.store import CockpitStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repository_snapshot_uses_real_git_and_oak_evidence():
    snapshot = inspect_repository(PROJECT_ROOT)

    assert snapshot["commit"]
    assert snapshot["tracked_files"] > 100
    assert snapshot["extension_counts"][".py"] > 0
    assert snapshot["reuse_policy"] == "licensed_source"
    assert snapshot["oak_kernel"]["verified"] is True


def test_repository_intake_persists_a_completed_verifiable_trace(tmp_path):
    store = CockpitStore(tmp_path / "cockpit.db")
    service = RepositoryIntakeService(store)

    task = service.run(PROJECT_ROOT, request_key="openmanus-self-v1")
    duplicate = service.run(PROJECT_ROOT, request_key="openmanus-self-v1")

    assert task["status"] == "completed"
    assert duplicate["id"] == task["id"]
    assert store.verify_task(task["id"]).valid is True
    events = store.export_task(task["id"])["events"]
    assert [event["kind"] for event in events] == [
        "task.created",
        "task.accepted",
        "run.started",
        "evidence.recorded",
        "run.completed",
    ]


def test_snapshot_is_anchored_to_head_not_dirty_or_untracked_files(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Cockpit Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "remote",
            "add",
            "origin",
            "https://user:secret@example.com/org/repository.git",
        ],
        check=True,
    )
    license_path = repository / "LICENSE"
    license_path.write_text("committed license\n", encoding="utf-8")
    (repository / "package.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "LICENSE", "package.json"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )

    license_path.write_text("dirty replacement\n", encoding="utf-8")
    (repository / "NOTICE.md").write_text("untracked notice\n", encoding="utf-8")
    snapshot = inspect_repository(repository)

    expected = hashlib.sha256(b"committed license\n").hexdigest()
    assert snapshot["evidence_file_sha256"]["LICENSE"] == expected
    assert snapshot["tracked_files"] == 2
    assert snapshot["notice_and_provenance_files"] == []
    assert snapshot["origin"] == "https://example.com/org/repository.git"
