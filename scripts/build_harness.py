"""Build and exercise a local wheel without network, pip or dependency installs.

Uses an allowlisted staging tree, then runs console entrypoints loaded from the
extracted wheel outside the checkout. Dependencies come from the interpreter
running this script: this is not evidence of a clean-environment installation.
Output must be empty; existing artifacts are never overwritten or removed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("setup.py", "requirements.txt", "README.md", "LICENSE", "main.py", "session_main.py")
DATA_FILES = ("app/cockpit/static/cockpit.css", "app/cockpit/static/verifier.js",
              *(f"app/oak/bundles/openmanus-v1/{name}.json"
                for name in ("schema", "functions", "graph", "kernel.lock")))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_output(path: Path) -> Path:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("Build output must be empty; use a new directory")
    return path


def stage_source(source: Path, destination: Path) -> dict[str, str]:
    """Copy only code and named package resources; never project-local state."""
    files = {source / name for name in ROOT_FILES}
    files.update((source / "app").rglob("*.py"))
    files.update(source / name for name in DATA_FILES)
    manifest = {}
    for path in sorted(files):
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Source symlinks are not accepted in the build staging tree")
        relative = path.relative_to(source)
        content = path.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest[relative.as_posix()] = digest(content)
    return manifest


def inspect_wheel(path: Path) -> list[str]:
    required = {"main.py", "session_main.py", "app/harness/session.py",
                "app/harness/provider.py", "app/harness/context.py", "app/harness/runtime.py",
                "app/harness/control.py", "app/harness/verification.py"}
    required.update(f"app/oak/bundles/openmanus-v1/{name}.json"
                    for name in ("schema", "functions", "graph", "kernel.lock"))
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not required.issubset(names):
            raise ValueError(f"Wheel misses required files: {sorted(required - set(names))}")
        for name in names:
            parts = Path(name).parts
            package_code = name.endswith(".py") and (name.startswith("app/") or name in {"main.py", "session_main.py"})
            metadata = len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] in {
                "METADATA", "WHEEL", "RECORD", "LICENSE", "entry_points.txt", "top_level.txt",
            }
            if (name.startswith(("/", "\\")) or ".." in parts
                    or not (package_code or metadata or name in DATA_FILES)
                    or name.startswith(("config/", "workspace/", "logs/", "tests/"))
                    or name.endswith((".db", ".sqlite", ".jsonl", ".toml"))):
                raise ValueError(f"Unexpected state or unsafe path in wheel: {name}")
        entries = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(entries) != 1:
            raise ValueError("Wheel must contain one console entrypoint manifest")
        entry_text = archive.read(entries[0]).decode()
        if "openmanus = main:cli" not in entry_text or "openmanus-session = session_main:cli" not in entry_text:
            raise ValueError("Wheel console entrypoints must be synchronous CLI wrappers")
    return names


def child_environment(config_path: Path, extracted: Path) -> dict[str, str]:
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT"}
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update({"PYTHONPATH": str(extracted), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                "OPENMANUS_CONFIG": str(config_path), "OPENMANUS_WHEEL_ROOT": str(extracted),
                "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1",
                "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": "127.0.0.1,localhost"})
    return env


ENTRYPOINT_RUNNER = """
import importlib.metadata, inspect, os, pathlib, sys
root = pathlib.Path(os.environ['OPENMANUS_WHEEL_ROOT']).resolve()
distribution = next(d for d in importlib.metadata.distributions(path=[str(root)])
                    if d.metadata['Name'] == 'openmanus')
entry = next(e for e in distribution.entry_points if e.group == 'console_scripts' and e.name == sys.argv[1])
function = entry.load()
assert pathlib.Path(inspect.getfile(function)).resolve().is_relative_to(root)
sys.argv = [entry.name, *sys.argv[2:]]
function()
"""

RECEIPT_CHECK = """
import json, sys
from app.cockpit.store import CockpitStore
store = CockpitStore(sys.argv[1])
tasks = store.list_tasks()
assert len(tasks) == 1 and tasks[0]['status'] == 'failed'
task_id = tasks[0]['id']
verification = store.verify_task(task_id)
assert verification.valid
document = store.export_task(task_id)
receipts = [event['payload']['value'] for event in document['events']
            if event['kind'] == 'evidence.recorded']
assert receipts and receipts[0]['formal_conformance']['valid']
print(json.dumps({'status': 'failed', 'outer_chain_valid': verification.valid,
                  'formal_runtime_valid': receipts[0]['formal_conformance']['valid']}))
"""


def run_step(name: str, argv: list[str], *, cwd: Path, env: dict, output: Path,
             expected: int = 0, timeout: int = 60) -> dict:
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    log = output / f"{name}.log"
    log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    evidence = {"name": name, "argv": argv, "cwd": str(cwd), "exit_code": result.returncode,
                "expected_exit_code": expected, "log": log.name,
                "log_sha256": digest(log.read_bytes())}
    if result.returncode != expected:
        raise RuntimeError(f"{name}: exit {result.returncode}, expected {expected}; inspect {log}")
    return evidence


def build(output_path: Path) -> dict:
    output = prepare_output(output_path)
    report = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "status": "started", "interpreter": sys.executable,
              "setuptools_version": version("setuptools"), "steps": [],
              "scope": "extracted-wheel entrypoints with existing interpreter dependencies",
              "clean_install_verified": False, "remote_provider_verified": False}
    try:
        with tempfile.TemporaryDirectory(prefix="openmanus-wheel-") as temporary:
            temporary_root = Path(temporary).resolve()
            if temporary_root.is_relative_to(ROOT):
                raise ValueError("Wheel smoke must run outside the repository")
            staging, extracted, smoke = (temporary_root / name for name in ("source", "extracted", "smoke"))
            staging.mkdir()
            extracted.mkdir()
            smoke.mkdir()
            report["source_sha256"] = stage_source(ROOT, staging)
            config_path = smoke / "fixture.toml"
            config_path.write_text('[llm]\nmodel="fixture-local"\nbase_url="http://127.0.0.1:1/v1"\n'
                                   'api_key="fixture-not-a-secret"\nmax_tokens=128\n', encoding="utf-8")
            env = child_environment(config_path, extracted)
            report["steps"].append(run_step(
                "build", [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output)],
                cwd=staging, env=env, output=output,
            ))
            wheels = list(output.glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Expected exactly one built wheel")
            wheel = wheels[0]
            report["wheel"] = {"file": wheel.name, "sha256": digest(wheel.read_bytes()),
                               "bytes": wheel.stat().st_size, "contents": inspect_wheel(wheel)}
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(extracted)
            for entrypoint in ("openmanus", "openmanus-session"):
                report["steps"].append(run_step(
                    entrypoint + "-help", [sys.executable, "-c", ENTRYPOINT_RUNNER, entrypoint, "--help"],
                    cwd=smoke, env=env, output=output, timeout=30,
                ))
            db = smoke / "cockpit.db"
            report["steps"].append(run_step(
                "zero-step-smoke", [sys.executable, "-c", ENTRYPOINT_RUNNER, "openmanus",
                                    "--prompt", "local package lifecycle smoke", "--max-steps", "0",
                                    "--session", "new", "--cockpit-db", str(db),
                                    "--audit-log", str(smoke / "audit.jsonl")],
                cwd=smoke, env=env, output=output, expected=1, timeout=30,
            ))
            report["steps"].append(run_step(
                "receipt-verify", [sys.executable, "-c", RECEIPT_CHECK, str(db)],
                cwd=smoke, env=env, output=output, timeout=30,
            ))
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "evidence.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / "build-harness")
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps({"status": report["status"], "wheel": report["wheel"]["file"],
                      "sha256": report["wheel"]["sha256"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
