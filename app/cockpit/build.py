"""Build the native and WebAssembly variants of the Go trace verifier."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GO_MODULE_ROOT = PROJECT_ROOT / "cockpit"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, environment: dict[str, str]) -> str:
    process = subprocess.run(
        command,
        cwd=GO_MODULE_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed: "
            f"{process.stderr.strip() or process.stdout.strip()}"
        )
    return process.stdout.strip()


def build_verifiers(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    binary_dir = output / "bin"
    asset_dir = output / "assets"
    cache_dir = output / "go-build-cache"
    binary_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    go = shutil.which("go")
    if go is None:
        raise RuntimeError("Go was not found on PATH")
    environment = os.environ.copy()
    environment.update(
        {
            "GOCACHE": str(cache_dir),
            "GOTELEMETRY": "off",
        }
    )
    go_root = Path(_run([go, "env", "GOROOT"], environment=environment))
    wasm_runtime = go_root / "lib" / "wasm" / "wasm_exec.js"
    if not wasm_runtime.is_file():
        raise RuntimeError(f"Go WASM runtime was not found at {wasm_runtime}")

    native_name = "oaktrace-verify.exe" if os.name == "nt" else "oaktrace-verify"
    native_path = binary_dir / native_name
    wasm_path = asset_dir / "verifier.wasm"
    _run(
        [go, "build", "-trimpath", "-o", str(native_path), "./cmd/verify"],
        environment=environment,
    )
    wasm_environment = environment | {"GOOS": "js", "GOARCH": "wasm"}
    _run(
        [go, "build", "-trimpath", "-o", str(wasm_path), "./cmd/wasm"],
        environment=wasm_environment,
    )
    copied_runtime = asset_dir / "wasm_exec.js"
    shutil.copyfile(wasm_runtime, copied_runtime)

    artifacts = {}
    for name, path in (
        ("native", native_path),
        ("wasm", wasm_path),
        ("wasm_runtime", copied_runtime),
    ):
        artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    report = {
        "schema_version": 1,
        "go_version": _run([go, "version"], environment=environment),
        "artifacts": artifacts,
    }
    (output / "build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
