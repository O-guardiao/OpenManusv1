"""Test-environment gates shared by the repository suite."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest


@lru_cache(maxsize=1)
def _docker_daemon_available() -> bool:
    try:
        import docker

        client = docker.from_env(timeout=2)
        try:
            return bool(client.ping())
        finally:
            client.close()
    except Exception:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip true Docker integration tests when the daemon is unavailable.

    A skip is intentionally not a passing integration claim.  Unit, OaK, MCP,
    and policy tests continue to run normally on the same machine.
    """

    if _docker_daemon_available():
        return
    marker = pytest.mark.skip(
        reason="Docker daemon unavailable; integration path not commissioned"
    )
    for item in items:
        parts = {part.casefold() for part in Path(str(item.fspath)).parts}
        if "sandbox" in parts:
            item.add_marker(marker)
