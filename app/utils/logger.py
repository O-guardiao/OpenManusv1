import logging
import os
import sys

import structlog

from app.harness.redaction import redact_structlog_event


ENV_MODE = os.getenv("ENV_MODE", "LOCAL")

renderer = [structlog.processors.JSONRenderer()]
if ENV_MODE.lower() == "local".lower():
    renderer = [structlog.dev.ConsoleRenderer(exception_formatter=structlog.dev.plain_traceback)]

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.contextvars.merge_contextvars,
        # Format without local-variable capture, then redact all rendered values.
        structlog.processors.format_exc_info,
        redact_structlog_event,
        *renderer,
    ],
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(level=logging.DEBUG)
