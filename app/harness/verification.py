"""Bounded, declarative artifact acceptance checks. No shell or model execution."""

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


MAX_FILE_BYTES = 5 * 1024 * 1024
KINDS = frozenset({"file_exists", "text_contains", "json_equals", "sha256"})


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _number(value: str) -> Decimal:
    number = Decimal(value)
    if not math.isfinite(float(number)):
        raise ValueError("JSON number exceeds supported finite range")
    return number


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _json(data: str) -> Any:
    return json.loads(data, parse_constant=_reject_constant, parse_float=_number,
                      object_pairs_hook=_object)


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Check path must be a nonempty relative path")
    windows, posix = PureWindowsPath(value), PurePosixPath(value)
    parts = value.replace("\\", "/").split("/")
    if (windows.drive or windows.root or posix.is_absolute()
            or ".." in parts or ":" in value or any(part.endswith((" ", ".")) for part in parts)):
        raise ValueError("Check path is not a confined relative path")
    # Reject Windows device paths even when validating a portable spec on Unix.
    devices = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
               *(f"LPT{i}" for i in range(1, 10))}
    if any(part.split(".")[0].upper() in devices for part in parts):
        raise ValueError("Check path names a reserved device")
    return value.replace("\\", "/")


def _pointer_tokens(pointer: Any) -> list[str]:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("JSON pointer must be empty or begin with slash")
    tokens = pointer.split("/")[1:] if pointer else []
    for token in tokens:
        index = 0
        while index < len(token):
            if token[index] == "~":
                if index + 1 >= len(token) or token[index + 1] not in "01":
                    raise ValueError("Invalid JSON pointer escape")
                index += 1
            index += 1
    return [token.replace("~1", "/").replace("~0", "~") for token in tokens]


def _validate_check(check: Any) -> dict:
    if not isinstance(check, dict):
        raise ValueError("Each check must be an object")
    kind = check.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("Unknown acceptance check kind")
    allowed = {"kind", "path"}
    if kind != "file_exists":
        allowed.add("value")
    if kind == "json_equals":
        allowed.add("pointer")
    if set(check) - allowed or "path" not in check:
        raise ValueError("Invalid acceptance check fields")
    result = dict(check)
    result["path"] = _relative_path(check["path"])
    if kind != "file_exists" and "value" not in check:
        raise ValueError("Acceptance check requires a value")
    value = check.get("value")
    if kind == "text_contains" and (not isinstance(value, str) or not value):
        raise ValueError("text_contains requires a nonempty string")
    if kind == "sha256":
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError("sha256 requires a 64-character hexadecimal value")
        result["value"] = value.lower()
    if kind == "json_equals":
        _pointer_tokens(check.get("pointer", ""))
    return result


def _json_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_equal(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def _at_pointer(value: Any, pointer: str) -> Any:
    for token in _pointer_tokens(pointer):
        if isinstance(value, dict):
            value = value[token]
        elif isinstance(value, list):
            if not token or not token.isascii() or not token.isdecimal() or (len(token) > 1 and token[0] == "0"):
                raise KeyError("Invalid list index")
            value = value[int(token)]
        else:
            raise KeyError("Pointer crosses a scalar")
    return value


def _check_artifact(check: dict, root: Path) -> str | None:
    try:
        target = (root / check["path"]).resolve()
        if not target.is_relative_to(root):
            return "outside_workspace"
        metadata = target.stat()
        if not stat.S_ISREG(metadata.st_mode):
            return "not_file"
        if check["kind"] == "file_exists":
            return None
        if metadata.st_size > MAX_FILE_BYTES:
            return "file_too_large"
        with target.open("rb") as stream:
            # Recheck the opened object and bound the read if the file grows.
            if os.fstat(stream.fileno()).st_size > MAX_FILE_BYTES:
                return "file_too_large"
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            return "file_too_large"
    except FileNotFoundError:
        return "missing_file"
    except (OSError, RuntimeError, ValueError):
        return "read_failed"
    if check["kind"] == "sha256":
        return None if hashlib.sha256(data).hexdigest() == check["value"] else "value_mismatch"
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError:
        return "invalid_utf8"
    if check["kind"] == "text_contains":
        return None if check["value"] in text else "value_mismatch"
    try:
        value = _json(text)
    except (ValueError, RecursionError):
        return "invalid_json"
    try:
        actual = _at_pointer(value, check.get("pointer", ""))
    except (KeyError, IndexError, ValueError):
        return "pointer_missing"
    try:
        return None if _json_equal(actual, check["value"]) else "value_mismatch"
    except RecursionError:
        return "invalid_json"


@dataclass(frozen=True)
class AcceptanceSpec:
    workspace_root: Path
    _checks: tuple[dict, ...]

    @classmethod
    def from_file(cls, path: str | Path, workspace_root: Path) -> "AcceptanceSpec":
        root = Path(workspace_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Workspace root must be a directory")
        spec_path = Path(path)
        if not spec_path.is_absolute():
            spec_path = root / spec_path
        try:
            if not spec_path.is_file() or spec_path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Acceptance specification is missing or too large")
            with spec_path.open("rb") as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("Acceptance specification is too large")
            payload = _json(data.decode("utf-8-sig"))
        except (OSError, UnicodeError, ValueError, RecursionError):
            raise ValueError("Cannot read a valid acceptance specification") from None
        if not isinstance(payload, dict) or set(payload) != {"version", "checks"}:
            raise ValueError("Invalid acceptance specification fields")
        if type(payload["version"]) is not int or payload["version"] != 1:
            raise ValueError("Acceptance specification version must be 1")
        checks = payload["checks"]
        if not isinstance(checks, list) or not 1 <= len(checks) <= 50:
            raise ValueError("Acceptance requires between 1 and 50 checks")
        # Validate the entire schema before inspecting any artifact.
        return cls(root, tuple(_validate_check(check) for check in checks))

    def validate(self) -> dict:
        results = []
        for index, check in enumerate(self._checks):
            error = _check_artifact(check, self.workspace_root)
            result = {"index": index, "kind": check["kind"], "path": check["path"],
                      "passed": error is None}
            if error:
                result["error"] = error
            results.append(result)
        return {"passed": all(item["passed"] for item in results),
                "checks": results, "assurance": "artifact_checks"}


def verify_acceptance(spec_path: str | Path, workspace_root: Path) -> dict:
    return AcceptanceSpec.from_file(spec_path, workspace_root).validate()
