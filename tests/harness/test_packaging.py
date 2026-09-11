"""Regression guards for state exclusion and non-destructive build output."""

import importlib.util
from pathlib import Path

import pytest


_SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "build_harness.py"
_SPEC = importlib.util.spec_from_file_location("build_harness", _SOURCE)
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def test_output_refuses_existing_content_without_overwriting(tmp_path):
    destination = tmp_path / "output"
    destination.mkdir()
    marker = destination / "important.whl"
    marker.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError):
        builder.prepare_output(destination)
    assert marker.read_bytes() == b"existing artifact"


def test_staging_includes_namespace_sources_but_excludes_local_secrets_and_state(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    for name in builder.ROOT_FILES:
        (source / name).write_text("fixture", encoding="utf-8")
    included = ["app/harness/session.py", "app/sandbox/core/sandbox.py", *builder.DATA_FILES]
    excluded = ["config/config.toml", "config/mcp.json", "workspace/private.jsonl",
                "workspace/cockpit.db", "logs/private.log", "app/__pycache__/stale.pyc",
                "app/oak/bundles/openmanus-v1/private.json", "app/cockpit/static/private.js"]
    for name in included + excluded:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    manifest = builder.stage_source(source, destination)
    assert all(name in manifest and (destination / name).is_file() for name in included)
    assert all(name not in manifest and not (destination / name).exists() for name in excluded)
