"""Real CLI/SQLite/tool integration against a deterministic loopback HTTP provider.

The provider is a fixture, not a real model: these tests prove orchestration,
stream transport and persistence, not reasoning quality or remote compatibility.
Child environments omit inherited credentials and direct non-loopback HTTP to
an unreachable loopback proxy, also exercising the offline tokenizer fallback.
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading

import pytest

from app.harness.session import SessionStore


ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def provider(replies=None):
    requests = []
    responses = list(replies or [{"role": "assistant", "content": "fixture final answer"}])

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if self.path != "/v1/chat/completions" or not responses:
                self.send_error(400, "Unexpected request")
                return
            message = responses.pop(0)
            finish = "tool_calls" if message.get("tool_calls") else "stop"
            usage = {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38}
            base = {"id": "chatcmpl-local-fixture", "created": 1, "model": "fixture-local"}
            if body.get("stream"):
                chunks = []
                if message.get("tool_calls"):
                    delta = {**message, "tool_calls": [
                        {"index": index, **call} for index, call in enumerate(message["tool_calls"])
                    ]}
                    chunks.append({"index": 0, "delta": delta, "finish_reason": None})
                else:
                    text = message["content"]
                    chunks.extend([
                        {"index": 0, "delta": {"role": "assistant", "content": text[:8]}, "finish_reason": None},
                        {"index": 0, "delta": {"content": text[8:]}, "finish_reason": None},
                    ])
                chunks.append({"index": 0, "delta": {}, "finish_reason": finish})
                frames = [dict(base, object="chat.completion.chunk", choices=[chunk]) for chunk in chunks]
                frames.append(dict(base, object="chat.completion.chunk", choices=[], usage=usage))
                payload = ("".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
                           + "data: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                payload = json.dumps(dict(
                    base, object="chat.completion", usage=usage,
                    choices=[{"index": 0, "message": message, "finish_reason": finish}],
                )).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def cli(root, endpoint, *arguments, prompt="first private prompt", session="new", steps=1):
    config = root / "fixture.toml"
    config.write_text(
        '[llm]\nmodel="fixture-local"\napi_type="openai"\n'
        f'base_url="{endpoint}"\napi_key="fixture-not-a-secret"\n'
        'max_tokens=128\ntemperature=0.0\n', encoding="utf-8",
    )
    keep = {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT"}
    env = {name: value for name, value in os.environ.items() if name.upper() in keep}
    env.update({
        "OPENMANUS_CONFIG": str(config), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
        "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1",
        "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": "127.0.0.1,localhost",
        "TIKTOKEN_CACHE_DIR": str(root / "empty-token-cache"),
    })
    command = [
        sys.executable, str(ROOT / "main.py"), "--prompt", prompt, "--session", session,
        "--max-steps", str(steps), "--cockpit-db", str(root / "cockpit.db"),
        "--audit-log", str(root / "audit.jsonl"), "--events", str(root / "events.jsonl"),
        "--context-tokens", "32768", *map(str, arguments),
    ]
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=30)


def session_id(result):
    match = re.search(r"Session:\s*([0-9a-f-]{36})", result.stdout + result.stderr)
    assert match, result.stdout + result.stderr
    return match.group(1)


def assert_status(root, status):
    with sqlite3.connect(root / "cockpit.db") as db:
        statuses = [row[0] for row in db.execute("SELECT status FROM tasks")]
    assert statuses and all(value == status for value in statuses), statuses


def spec(root):
    path = root / "acceptance.json"
    path.write_text(json.dumps({"version": 1, "checks": [
        {"kind": "text_contains", "path": "result.txt", "value": "verified artifact"},
    ]}), encoding="utf-8")
    return path


@pytest.mark.parametrize("stream", [False, True])
def test_plain_final_finishes_at_first_step_and_receipt_is_completed(tmp_path, stream):
    with provider() as (url, requests):
        result = cli(tmp_path, url, *(["--stream"] if stream else []))
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(requests) == 1
    assert requests[0].get("stream", False) is stream
    assert "fixture final answer" in result.stdout + result.stderr
    assert_status(tmp_path, "completed")
    store = SessionStore(tmp_path / "cockpit.db")
    assert store.get_session(session_id(result))["status"] == "completed"
    assert any(message.content == "fixture final answer" for message in store.load(session_id(result)))
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events


def test_resume_in_another_process_preserves_prompts_without_restoring_grants(tmp_path):
    replies = [{"role": "assistant", "content": "first answer"},
               {"role": "assistant", "content": "second answer"}]
    with provider(replies) as (url, requests):
        first = cli(tmp_path, url, "--allow-tool", "str_replace_editor")
        assert first.returncode == 0, first.stdout + first.stderr
        second = cli(tmp_path, url, prompt="second private prompt", session=session_id(first))
    assert second.returncode == 0, second.stdout + second.stderr
    assert len(requests) == 2
    content = json.dumps(requests[1]["messages"])
    assert "first private prompt" in content and "second private prompt" in content
    names = lambda request: {tool["function"]["name"] for tool in request.get("tools", [])}
    assert "str_replace_editor" in names(requests[0])
    assert "str_replace_editor" not in names(requests[1])
    assert_status(tmp_path, "completed")


@pytest.mark.parametrize("artifact_exists", [False, True])
def test_artifact_acceptance_controls_real_cli_exit_and_receipt(tmp_path, artifact_exists):
    if artifact_exists:
        (tmp_path / "result.txt").write_text("verified artifact", encoding="utf-8")
    with provider() as (url, requests):
        result = cli(tmp_path, url, "--acceptance", spec(tmp_path), "--artifact-root", tmp_path)
    assert len(requests) == 1
    assert result.returncode == (0 if artifact_exists else 1), result.stdout + result.stderr
    assert_status(tmp_path, "completed" if artifact_exists else "failed")


def test_pending_effect_refuses_resume_before_model_request(tmp_path):
    store = SessionStore(tmp_path / "cockpit.db")
    session = store.create()
    with store.lease(session):
        store.begin_run(session, "interrupted-run")
        store.record_effect_intent(session, "pending-op", "str_replace_editor", "a" * 64)
    with provider() as (url, requests):
        result = cli(tmp_path, url, session=session)
    assert result.returncode != 0
    assert requests == []
    assert store.pending_effects(session)[0]["operation_id"] == "pending-op"


def test_real_editor_tool_creates_artifact_then_passes_acceptance(tmp_path):
    target = tmp_path / "result.txt"
    tool = {"id": "edit-1", "type": "function", "function": {
        "name": "str_replace_editor", "arguments": json.dumps({
            "command": "create", "path": str(target), "file_text": "verified artifact",
        }),
    }}
    replies = [{"role": "assistant", "content": None, "tool_calls": [tool]},
               {"role": "assistant", "content": "Artifact created and checked."}]
    with provider(replies) as (url, requests):
        result = cli(tmp_path, url, "--allow-tool", "str_replace_editor", "--stream",
                     "--acceptance", spec(tmp_path), "--artifact-root", tmp_path, steps=2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(requests) == 2
    assert target.read_text(encoding="utf-8") == "verified artifact"
    assert any(message.get("role") == "tool" for message in requests[1]["messages"])
    assert_status(tmp_path, "completed")
    assert SessionStore(tmp_path / "cockpit.db").pending_effects(session_id(result)) == []
