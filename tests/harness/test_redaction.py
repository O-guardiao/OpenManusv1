"""Canaries inspect emitted application logs, never real configuration secrets."""

import importlib
from io import StringIO
import sys
from types import SimpleNamespace

import pytest
import structlog


API_KEY = "fixture-api-credential-726"
DAYTONA_KEY = "fixture-daytona-credential-384"
VNC_PASSWORD = "fixture-vnc-password-643"
PROXY_PASSWORD = "fixture-proxy-password-973"
KNOWN = (API_KEY, DAYTONA_KEY, VNC_PASSWORD, PROXY_PASSWORD)


@pytest.fixture
def fake_config(monkeypatch):
    config_module = importlib.import_module("app.config")
    fake = SimpleNamespace(
        llm={"default": SimpleNamespace(api_key=API_KEY)},
        daytona=SimpleNamespace(daytona_api_key=DAYTONA_KEY, VNC_password=VNC_PASSWORD),
        browser_config=SimpleNamespace(proxy=SimpleNamespace(password=PROXY_PASSWORD)),
    )
    monkeypatch.setattr(config_module, "config", fake)
    return fake


@pytest.fixture
def loguru_sink(tmp_path, monkeypatch, fake_config):
    module = importlib.import_module("app.logger")
    sink = StringIO()
    with monkeypatch.context() as patch:
        patch.setattr(module, "PROJECT_ROOT", tmp_path)
        logger = module.define_log_level(name="redaction-canary")
        logger.add(sink, format="{message} {extra}")
        yield logger, sink
    module.define_log_level()


def test_loguru_redacts_known_values_headers_and_nested_extras_before_sinks(loguru_sink, tmp_path):
    logger, sink = loguru_sink
    logger.bind(
        nested={"api_key": "unconfigured-extra-key", "safe": "retained detail",
                "entries": [{"password": "unconfigured-extra-password"}]},
        header="Authorization: Bearer unconfigured-header-credential",
    ).info("Safe operation: {}", " ".join(KNOWN))
    outputs = [sink.getvalue()] + [path.read_text(encoding="utf-8") for path in (tmp_path / "logs").glob("*.log")]
    assert len(outputs) > 1
    for output in outputs:
        assert "Safe operation" in output
        for secret in KNOWN + ("unconfigured-extra-key", "unconfigured-extra-password",
                               "unconfigured-header-credential"):
            assert secret not in output
    assert "retained detail" in sink.getvalue()


def test_loguru_exception_keeps_type_without_secret_or_locals(loguru_sink):
    logger, sink = loguru_sink
    private_local = "fixture-local-must-never-appear"
    try:
        raise ValueError("Failure using " + API_KEY)
    except ValueError:
        logger.exception("Harmless failure description")
    output = sink.getvalue()
    assert "ValueError" in output
    assert "Harmless failure description" in output
    assert API_KEY not in output
    assert private_local not in output


def test_loguru_picks_up_changed_credentials_without_reconfiguration(loguru_sink, fake_config):
    logger, sink = loguru_sink
    fake_config.llm["default"].api_key = "fixture-rotated-credential-617"
    logger.info("Rotated value {}", fake_config.llm["default"].api_key)
    assert "Rotated value" in sink.getvalue()
    assert fake_config.llm["default"].api_key not in sink.getvalue()


@pytest.mark.parametrize("mode", ["LOCAL", "JSON"])
def test_structlog_stderr_redacts_tree_and_exception_without_locals(monkeypatch, fake_config, mode):
    saved = structlog.get_config().copy()
    sink = StringIO()
    try:
        with monkeypatch.context() as patch:
            patch.setenv("ENV_MODE", mode)
            patch.setattr(sys, "stderr", sink)
            module = importlib.reload(importlib.import_module("app.utils.logger"))
            # Observe the module's actual processor pipeline at a real stderr sink.
            structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
                                cache_logger_on_first_use=False)
            logger = structlog.get_logger()
            private_local = "fixture-structlog-local-must-never-appear"
            try:
                raise RuntimeError("Provider rejected " + API_KEY)
            except RuntimeError:
                logger.exception(
                    "Harmless structured failure", credentials=list(KNOWN),
                    data={"token": "unconfigured-structured-token", "safe": "retained detail"},
                    request="Authorization: Bearer unconfigured-bearer-credential",
                )
            output = sink.getvalue()
            assert "Harmless structured failure" in output
            assert "retained detail" in output
            assert "RuntimeError" in output
            for secret in KNOWN + ("unconfigured-structured-token", "unconfigured-bearer-credential", private_local):
                assert secret not in output
    finally:
        structlog.configure(**saved)


def test_redaction_failure_never_emits_original_loguru_record(loguru_sink, monkeypatch):
    from app.harness import redaction

    logger, sink = loguru_sink

    def broken_redactor():
        raise RuntimeError("fixture-internal-secret")

    monkeypatch.setattr(redaction, "configured_secrets", broken_redactor)
    logger.bind(password="must-not-leak").info("Original payload {}", API_KEY)
    output = sink.getvalue()
    assert "REDACTION FAILED" in output
    assert "Original payload" not in output
    assert "must-not-leak" not in output
    assert "fixture-internal-secret" not in output
    assert API_KEY not in output


def test_redaction_failure_never_emits_original_structlog_record(monkeypatch, fake_config):
    from app.harness import redaction

    saved = structlog.get_config().copy()
    sink = StringIO()
    try:
        module = importlib.reload(importlib.import_module("app.utils.logger"))
        structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sink),
                            cache_logger_on_first_use=False)

        def broken_redactor():
            raise RuntimeError("fixture-internal-secret")

        monkeypatch.setattr(redaction, "configured_secrets", broken_redactor)
        structlog.get_logger().error("Original payload " + API_KEY, password="must-not-leak")
        output = sink.getvalue()
        assert "REDACTION FAILED" in output
        assert "Original payload" not in output
        assert "must-not-leak" not in output
        assert API_KEY not in output
    finally:
        structlog.configure(**saved)
