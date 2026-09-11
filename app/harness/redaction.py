"""Bounded credential redaction for configured application logs only.

Known configuration values are read at emission time, never from an environment
scan. This does not cover transcripts, opt-in stream/event output, direct print,
unconfigured stdlib loggers, or secrets in unknown encodings or unknown fields.
"""

from collections.abc import Mapping
import re
import sys


REDACTED = "[REDACTED]"
FAILED = "[LOG REDACTION FAILED]"
_AUTHORIZATION = re.compile(
    r"(?im)(\b(?:proxy-)?authorization[\"']?\s*[:=]\s*)[^\r\n,;}]+"
)
_BEARER = re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]+")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|(?:access[_-]?|refresh[_-]?)?token|password|secret)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}\]]+)"
)


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def configured_secrets() -> tuple[str, ...]:
    """Read only explicitly known credential fields after config initialization."""
    config = _field(sys.modules.get("app.config"), "config")
    if config is None:
        return ()
    llms = _field(config, "llm", {}) or {}
    values = [_field(settings, "api_key") for settings in llms.values()]
    daytona = _field(config, "daytona")
    values.extend((_field(daytona, "daytona_api_key"), _field(daytona, "VNC_password")))
    proxy = _field(_field(config, "browser_config"), "proxy")
    values.append(_field(proxy, "password"))
    return tuple(sorted(
        {value for value in values if isinstance(value, str) and value.strip()},
        key=len, reverse=True,
    ))


def _sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    return normalized in {"token", "authorization", "proxyauthorization"} or normalized.endswith(
        ("apikey", "password", "secret", "accesstoken", "refreshtoken", "authtoken", "bearertoken")
    )


def redact_text(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        text = text.replace(secret, REDACTED)
    text = _AUTHORIZATION.sub(lambda match: match.group(1) + REDACTED, text)
    text = _BEARER.sub("Bearer " + REDACTED, text)
    return _SENSITIVE_ASSIGNMENT.sub(lambda match: match.group(1) + REDACTED, text)


def redact_tree(value, secrets: tuple[str, ...], *, depth=0):
    """Return detached safe values so later rendering cannot expose raw objects."""
    if depth > 24:
        return REDACTED
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, bytes):
        return redact_text(value.decode("utf-8", errors="replace"), secrets)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key = str(key)
            result[redact_text(key, secrets)] = (
                REDACTED if _sensitive_key(key)
                else redact_tree(item, secrets, depth=depth + 1)
            )
        return result
    if isinstance(value, (tuple, list, set, frozenset)):
        return [redact_tree(item, secrets, depth=depth + 1) for item in value]
    if value is None or type(value) in (bool, int, float):
        return value
    return redact_text(str(value), secrets)


def redact_loguru_record(record: dict) -> None:
    """Patch before every sink; exceptions become a redacted type/message summary."""
    try:
        secrets = configured_secrets()
        message = redact_text(record["message"], secrets)
        extra = redact_tree(record["extra"], secrets)
        exception = record["exception"]
        if exception is not None:
            name = getattr(exception.type, "__name__", "Exception")
            summary = redact_text(f"{name}: {exception.value}", secrets)
            message = f"{message}\n{summary}"
        record["message"] = message
        record["extra"] = extra
    except Exception:
        # No original message, extras, exception or redactor failure details escape.
        record["message"] = FAILED
        record["extra"] = {}
    finally:
        record["exception"] = None


def redact_structlog_event(_logger, _method_name, event_dict: dict) -> dict:
    """Final processor before rendering, including merged context and exception text."""
    try:
        return redact_tree(event_dict, configured_secrets())
    except Exception:
        return {"event": FAILED, "level": "error"}
