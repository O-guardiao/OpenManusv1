from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.cockpit.web import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_htmx_cockpit_runs_real_intake_and_exports_svg_and_json(tmp_path):
    app = create_app(
        database_path=tmp_path / "cockpit.db",
        allowed_roots=[PROJECT_ROOT],
        runtime_asset_dir=tmp_path / "runtime-assets",
    )
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert 'hx-post="/intakes"' in home.text
    assert "htmx.org@2.0.10" in home.text

    response = client.post(
        "/intakes",
        data={
            "repository_path": str(PROJECT_ROOT),
            "request_key": "web-openmanus-v1",
        },
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 201
    assert 'data-task-id="' in response.text
    task_id = response.headers["X-OpenManus-Task-Id"]

    graph = client.get(f"/tasks/{task_id}/graph.svg")
    assert graph.status_code == 200
    assert graph.headers["content-type"].startswith("image/svg+xml")
    assert "evidence.recorded" in graph.text

    exported = client.get(f"/tasks/{task_id}/export.json")
    assert exported.status_code == 200
    assert exported.json()["task"]["status"] == "completed"
    assert exported.json()["verification"]["valid"] is True


def test_web_intake_rejects_paths_outside_allowed_roots(tmp_path):
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    app = create_app(
        database_path=tmp_path / "cockpit.db",
        allowed_roots=[allowed],
        runtime_asset_dir=tmp_path / "runtime-assets",
    )
    client = TestClient(app)

    response = client.post(
        "/intakes",
        data={"repository_path": str(denied), "request_key": "denied-v1"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 403
    assert "fora das raízes permitidas" in response.text


def test_web_rejects_untrusted_host_headers(tmp_path):
    app = create_app(
        database_path=tmp_path / "cockpit.db",
        allowed_roots=[tmp_path],
        runtime_asset_dir=tmp_path / "runtime-assets",
    )
    client = TestClient(app)

    response = client.get("/", headers={"host": "attacker.example"})

    assert response.status_code == 400
