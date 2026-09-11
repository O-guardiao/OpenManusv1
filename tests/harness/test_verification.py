import hashlib
import json

import pytest


def write_spec(root, checks, **extra):
    path = root / "acceptance.json"
    path.write_text(json.dumps({"version": 1, "checks": checks, **extra}), encoding="utf-8")
    return path


def verify(path, root):
    from app.harness.verification import verify_acceptance

    return verify_acceptance(path, root)


def test_artifact_checks_inspect_real_files_and_json_pointer(tmp_path):
    data = b'{"a/b":{"~key":[true,42]},"name":"done"}'
    (tmp_path / "result.json").write_bytes(data)
    checks = [
        {"kind": "file_exists", "path": "result.json"},
        {"kind": "text_contains", "path": "result.json", "value": '"done"'},
        {"kind": "json_equals", "path": "result.json", "pointer": "/a~1b/~0key/1", "value": 42},
        {"kind": "sha256", "path": "result.json", "value": hashlib.sha256(data).hexdigest()},
    ]
    result = verify(write_spec(tmp_path, checks), tmp_path)
    assert result["passed"] is True
    assert result["assurance"] == "artifact_checks"
    assert all(check["passed"] for check in result["checks"])
    assert '"done"' not in json.dumps(result)


@pytest.mark.parametrize("check", [
    {"kind": "execute", "path": "x"},
    {"kind": "file_exists", "path": "../x"},
    {"kind": "file_exists", "path": "/absolute"},
    {"kind": "file_exists", "path": "C:\\outside"},
    {"kind": "file_exists", "path": "x", "value": True},
    {"kind": "text_contains", "path": "x", "value": ""},
    {"kind": "text_contains", "path": "x", "value": 3},
    {"kind": "sha256", "path": "x", "value": "not-a-hash"},
    {"kind": "json_equals", "path": "x", "pointer": "bad", "value": 2},
    {"kind": "json_equals", "path": "x", "pointer": "/a~2", "value": 2},
    {"kind": "file_exists", "path": "x", "command": "echo forbidden"},
])
def test_invalid_schema_rejected_before_any_artifact_read(tmp_path, check):
    with pytest.raises(ValueError):
        verify(write_spec(tmp_path, [{"kind": "file_exists", "path": "missing"}, check]), tmp_path)


@pytest.mark.parametrize("payload", [
    {"version": True, "checks": [{"kind": "file_exists", "path": "x"}]},
    {"version": 2, "checks": [{"kind": "file_exists", "path": "x"}]},
    {"version": 1, "checks": []},
    {"version": 1, "checks": [{"kind": "file_exists", "path": "x"}] * 51},
    {"version": 1, "checks": [{"kind": "file_exists", "path": "x"}], "shell": "x"},
])
def test_invalid_top_level_schema(tmp_path, payload):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        verify(path, tmp_path)


def test_failures_are_metadata_only_and_directory_is_not_file(tmp_path):
    (tmp_path / "secret.txt").write_text("PRIVATE-CONTENT", encoding="utf-8")
    (tmp_path / "directory").mkdir()
    checks = [
        {"kind": "file_exists", "path": "missing"},
        {"kind": "file_exists", "path": "directory"},
        {"kind": "text_contains", "path": "secret.txt", "value": "OTHER-PRIVATE-CONTENT"},
    ]
    result = verify(write_spec(tmp_path, checks), tmp_path)
    assert result["passed"] is False
    assert all(not item["passed"] for item in result["checks"])
    assert "PRIVATE-CONTENT" not in json.dumps(result)


def test_read_limit_and_invalid_json_do_not_return_content(tmp_path):
    (tmp_path / "large.txt").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    (tmp_path / "invalid.json").write_text("secret broken JSON", encoding="utf-8")
    checks = [
        {"kind": "text_contains", "path": "large.txt", "value": "x"},
        {"kind": "json_equals", "path": "invalid.json", "value": {}},
    ]
    result = verify(write_spec(tmp_path, checks), tmp_path)
    assert [item["error"] for item in result["checks"]] == ["file_too_large", "invalid_json"]
    assert "secret" not in json.dumps(result)


def test_json_pointer_missing_and_boolean_not_equal_number(tmp_path):
    (tmp_path / "data.json").write_text('{"items":[true]}', encoding="utf-8")
    checks = [
        {"kind": "json_equals", "path": "data.json", "pointer": "/items/0", "value": 1},
        {"kind": "json_equals", "path": "data.json", "pointer": "/missing", "value": None},
        {"kind": "json_equals", "path": "data.json", "pointer": "/items/01", "value": True},
    ]
    result = verify(write_spec(tmp_path, checks), tmp_path)
    assert not any(item["passed"] for item in result["checks"])


def test_symlink_escape_is_rejected_when_available(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Host does not allow symlink creation")
    result = verify(write_spec(root, [{"kind": "file_exists", "path": "link.txt"}]), root)
    assert result["checks"][0]["error"] == "outside_workspace"


def test_invalid_later_check_prevents_earlier_artifact_execution(tmp_path, monkeypatch):
    import app.harness.verification as module

    def forbidden(*args):
        pytest.fail("Artifact inspected before complete schema validation")
    monkeypatch.setattr(module, "_check_artifact", forbidden)
    path = write_spec(tmp_path, [{"kind": "file_exists", "path": "x"}, {"kind": "invalid", "path": "y"}])
    with pytest.raises(ValueError):
        module.verify_acceptance(path, tmp_path)


def test_nonfinite_overflow_in_spec_is_rejected(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text('{"version":1,"checks":[{"kind":"json_equals","path":"x","value":1e999}]}', encoding="utf-8")
    with pytest.raises(ValueError):
        verify(path, tmp_path)
