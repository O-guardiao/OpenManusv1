"""Real operator CLI controls against a gated loopback streaming provider.

The HTTP fixture deliberately withholds the terminal event. These tests prove
process orchestration and durable state, not model quality or remote services.
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from app.cockpit.store import CockpitStore
from app.harness.session import SessionStore


ROOT = Path(__file__).resolve().parents[2]
FIRST_ANSWER = "First terminal answer."
SECOND_ANSWER = "Updated answer follows steering."
STEERING = "Include the newly requested detail before finishing."


@contextmanager
def gated_provider():
    requests = []
    first_event_sent = threading.Event()
    release_first_response = threading.Event()
    second_request_received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append(body)
            if self.path != "/v1/chat/completions" or index > 1:
                self.send_error(400, "Unexpected request")
                return
            if index == 1:
                second_request_received.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def emit(delta=None, finish=None, usage=False):
                frame = {
                    "id": f"fixture-control-{index}", "object": "chat.completion.chunk",
                    "created": 1, "model": "fixture-local",
                    "choices": [] if usage else [{
                        "index": 0, "delta": delta or {}, "finish_reason": finish,
                    }],
                }
                if usage:
                    frame["usage"] = {"prompt_tokens": 40, "completion_tokens": 8,
                                      "total_tokens": 48}
                self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
                self.wfile.flush()

            try:
                if index == 0:
                    emit({"role": "assistant", "content": FIRST_ANSWER[:6]})
                    first_event_sent.set()
                    if not release_first_response.wait(timeout=30):
                        return
                    emit({"content": FIRST_ANSWER[6:]})
                else:
                    emit({"role": "assistant", "content": SECOND_ANSWER})
                emit(finish="stop")
                emit(usage=True)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # The cancellation test intentionally disconnects before DONE.
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            f"http://127.0.0.1:{server.server_port}/v1", requests,
            first_event_sent, release_first_response, second_request_received,
        )
    finally:
        release_first_response.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def isolated_environment(root, endpoint):
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
    return env


@contextmanager
def running_cli(root, endpoint):
    env = isolated_environment(root, endpoint)
    command = [
        sys.executable, str(ROOT / "main.py"), "--prompt", "Complete the initial task.",
        "--session", "new", "--stream", "--max-steps", "3",
        "--cockpit-db", str(root / "cockpit.db"),
        "--audit-log", str(root / "audit.jsonl"), "--events", str(root / "events.jsonl"),
        "--context-tokens", "32768",
    ]
    # Files prevent stderr pipe backpressure while the test is issuing controls.
    with (root / "stdout.log").open("w", encoding="utf-8") as stdout:
        with (root / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            try:
                yield process, env
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


def diagnostics(root):
    return "\n".join(
        (root / name).read_text(encoding="utf-8", errors="replace")
        for name in ("stdout.log", "stderr.log")
    )


def events(root):
    path = root / "events.jsonl"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


def wait_event(root, predicate, process, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = next((event for event in events(root) if predicate(event)), None)
        if found:
            return found
        if process.poll() is not None:
            pytest.fail("CLI exited before the expected event:\n" + diagnostics(root))
        time.sleep(.05)
    pytest.fail("CLI did not emit the expected event:\n" + diagnostics(root))


def control(root, env, session_id, action, text=None):
    command = [sys.executable, str(ROOT / "session_main.py"),
               "--database", str(root / "cockpit.db"), action, session_id]
    if text is not None:
        command.append(text)
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cancel_inflight_stream_interrupts_process_session_and_receipt(tmp_path):
    with gated_provider() as (url, requests, ready, release, second):
        with running_cli(tmp_path, url) as (process, env):
            assert ready.wait(timeout=20), diagnostics(tmp_path)
            started = wait_event(tmp_path, lambda event: event["type"] == "model_text_delta", process)
            session_id = started["session_id"]
            control(tmp_path, env, session_id, "cancel")
            assert process.wait(timeout=30) == 130, diagnostics(tmp_path)
            # Completion was withheld: success cannot have come from the provider.
            assert not release.is_set()
            assert not second.is_set()
            assert len(requests) == 1
    store = SessionStore(tmp_path / "cockpit.db")
    assert store.get_session(session_id)["status"] == "interrupted"
    assert store.get_session(session_id)["active_run_id"] is None
    assert not any(message.content == FIRST_ANSWER for message in store.load(session_id))
    cockpit = CockpitStore(tmp_path / "cockpit.db")
    task = cockpit.list_tasks()[0]
    assert task["status"] == "failed"
    receipt = next(event["payload"]["value"] for event in cockpit.export_task(task["id"])["events"]
                   if event["kind"] == "evidence.recorded")
    assert receipt["terminal_status"] == "interrupted"
    assert receipt["formal_conformance"]["valid"] is True
    assert cockpit.verify_task(task["id"]).valid is True
    assert all(event.get("terminal_status") != "completed" for event in events(tmp_path))


@pytest.mark.parametrize("pause_first", [False, True])
def test_steer_during_terminal_reply_starts_another_turn_and_pause_blocks_it(tmp_path, pause_first):
    with gated_provider() as (url, requests, ready, release, second):
        with running_cli(tmp_path, url) as (process, env):
            assert ready.wait(timeout=20), diagnostics(tmp_path)
            started = wait_event(tmp_path, lambda event: event["type"] == "model_text_delta", process)
            session_id = started["session_id"]
            if pause_first:
                control(tmp_path, env, session_id, "pause")
                wait_event(tmp_path, lambda event: event["type"] == "control_received"
                           and event.get("action") == "pause", process)
            control(tmp_path, env, session_id, "steer", STEERING)
            release.set()
            if pause_first:
                wait_event(tmp_path, lambda event: event["type"] == "message_recorded"
                           and event.get("role") == "assistant", process)
                assert not second.wait(timeout=.5), "A second model request started while paused"
                assert process.poll() is None, diagnostics(tmp_path)
                control(tmp_path, env, session_id, "continue")
            assert process.wait(timeout=30) == 0, diagnostics(tmp_path)
            assert second.is_set()
            assert len(requests) == 2
            second_messages = requests[1]["messages"]
            assert any(message.get("role") == "user" and message.get("content") == STEERING
                       for message in second_messages)
    store = SessionStore(tmp_path / "cockpit.db")
    transcript = store.load(session_id)
    assert store.get_session(session_id)["status"] == "completed"
    assert [(message.role, message.content) for message in transcript][-3:] == [
        ("assistant", FIRST_ANSWER), ("user", STEERING), ("assistant", SECOND_ANSWER),
    ]
    assert any(event["type"] == "steering_applied" for event in events(tmp_path))
    assert all(task["status"] == "completed" for task in CockpitStore(tmp_path / "cockpit.db").list_tasks())
