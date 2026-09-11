#!/usr/bin/env python3
"""CLI and local server entrypoint for the OpenManus evidence cockpit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from app.cockpit.build import build_verifiers
from app.cockpit.intake import RepositoryIntakeService
from app.cockpit.store import CockpitStore, verify_export
from app.cockpit.web import create_app


DEFAULT_DATABASE = Path("workspace/cockpit/openmanus.db")
DEFAULT_BUILD_ROOT = Path("workspace/cockpit/runtime")


def _write_json(value: object, output: Path | None = None) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenManus evidence-first reverse-engineering cockpit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    intake = subparsers.add_parser("intake", help="inspect and persist a repository")
    intake.add_argument("repository", type=Path)
    intake.add_argument("--request-key", required=True)
    intake.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    intake.add_argument("--export", type=Path)

    export = subparsers.add_parser("export", help="export one persisted task")
    export.add_argument("task_id")
    export.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    export.add_argument("--output", type=Path)

    verify = subparsers.add_parser("verify", help="verify a trace with Python")
    verify.add_argument("trace", type=Path)

    build = subparsers.add_parser(
        "build-verifiers", help="build the Go native and browser verifiers"
    )
    build.add_argument("--output-root", type=Path, default=DEFAULT_BUILD_ROOT)

    serve = subparsers.add_parser("serve", help="start the local HTMX cockpit")
    serve.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    serve.add_argument("--runtime-assets", type=Path, default=DEFAULT_BUILD_ROOT / "assets")
    serve.add_argument("--allow-root", action="append", type=Path, default=[])
    serve.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "intake":
        store = CockpitStore(arguments.database)
        task = RepositoryIntakeService(store).run(
            arguments.repository, request_key=arguments.request_key
        )
        document = store.export_task(task["id"])
        document["verification"] = store.verify_task(task["id"]).to_dict()
        _write_json(document, arguments.export)
        return 0 if document["verification"]["valid"] else 1
    if arguments.command == "export":
        store = CockpitStore(arguments.database)
        document = store.export_task(arguments.task_id)
        document["verification"] = store.verify_task(arguments.task_id).to_dict()
        _write_json(document, arguments.output)
        return 0 if document["verification"]["valid"] else 1
    if arguments.command == "verify":
        document = json.loads(arguments.trace.read_text(encoding="utf-8"))
        result = verify_export(document)
        _write_json(result.to_dict())
        return 0 if result.valid else 1
    if arguments.command == "build-verifiers":
        _write_json(build_verifiers(arguments.output_root))
        return 0
    if arguments.command == "serve":
        import uvicorn

        roots = arguments.allow_root or [Path.cwd()]
        app = create_app(
            database_path=arguments.database,
            allowed_roots=roots,
            runtime_asset_dir=arguments.runtime_assets,
        )
        uvicorn.run(app, host=arguments.host, port=arguments.port)
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
