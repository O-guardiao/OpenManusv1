class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class OpenManusError(Exception):
    """Base exception for all OpenManus errors"""


class TokenLimitExceeded(OpenManusError):
    """Exception raised when the token limit is exceeded"""


class ProviderResponseError(ValueError, OpenManusError):
    """An unusable provider response; never eligible for automatic retry."""

    def __init__(self, message: str, code: str = "provider_invalid_response"):
        self.code = code
        super().__init__(f"{code}: {message}")


class StreamInterruptedError(ProviderResponseError):
    """Visible output may already exist, so replay requires an explicit decision."""

    def __init__(self):
        super().__init__(
            "Response interrupted after visible text; automatic replay is disabled.",
            code="stream_interrupted",
        )
