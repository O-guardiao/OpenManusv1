"""Local conversation management. Exports/backups include sensitive content."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Sequence

from app.config import config
from app.harness.control import ControlInbox
from app.harness.session import SessionStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and control local Manus conversations")
    parser.add_argument(
        "--database", type=Path,
        default=config.workspace_root / "cockpit" / "openmanus.db",
        help="Conversation SQLite database; for restore, a new destination database.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List session IDs and recovery status")
    show = commands.add_parser("show", help="Show one session including conversation content")
    show.add_argument("session_id")
    export = commands.add_parser("export", help="Export sensitive content to a new JSON file")
    export.add_argument("session_id")
    export.add_argument("destination", type=Path)
    backup = commands.add_parser("backup", help="Back up the shared SQLite database to a new file")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore", help="Restore a backup into the new --database path")
    restore.add_argument("backup", type=Path)
    steer = commands.add_parser("steer", help="Queue an instruction for the next safe checkpoint")
    steer.add_argument("session_id")
    steer.add_argument("text")
    for action in ("pause", "continue", "cancel"):
        command = commands.add_parser(action, help=f"Queue the {action} control for a session")
        command.add_argument("session_id")
    return parser


def _export_new(destination: Path, payload: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Never initialize a destination before restore's exclusive-create gate.
        if args.command == "restore":
            restored = SessionStore.restore(args.backup, args.database)
            print(f"Restored database: {restored.database_path}")
            return 0
        if not args.database.is_file():
            raise FileNotFoundError("Conversation database does not exist")
        store = SessionStore(args.database)
        if args.command == "list":
            print(json.dumps(store.list_sessions(), indent=2, ensure_ascii=False))
        elif args.command == "show":
            print(json.dumps(store.export_session(args.session_id), indent=2, ensure_ascii=False))
        elif args.command == "export":
            _export_new(args.destination, store.export_session(args.session_id))
            print(f"Exported session: {args.destination.resolve()}")
        elif args.command == "backup":
            print(f"Database backup: {store.backup(args.destination)}")
        else:
            store.get_session(args.session_id)
            request_id = ControlInbox(args.database, args.session_id).send(
                args.command, getattr(args, "text", "")
            )
            print(f"Control queued: {request_id}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as error:
        print(f"Session command failed: {error}", file=sys.stderr)
        return 1


def cli() -> None:
    """Synchronous console-script entrypoint."""
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
