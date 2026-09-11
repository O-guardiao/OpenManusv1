"""Local-only FastAPI surface for the evidence cockpit."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.cockpit.intake import RepositoryIntakeService
from app.cockpit.render import error_fragment, page, task_card, trace_svg
from app.cockpit.store import CockpitStore, IdempotencyConflict


STATIC_ROOT = Path(__file__).resolve().parent / "static"


def _inside_allowed_root(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def create_app(
    *,
    database_path: str | Path,
    allowed_roots: list[str | Path],
    runtime_asset_dir: str | Path,
) -> FastAPI:
    roots = tuple(Path(root).resolve(strict=True) for root in allowed_roots)
    if not roots:
        raise ValueError("at least one allowed root is required")
    assets = Path(runtime_asset_dir).resolve()
    store = CockpitStore(database_path)
    intake = RepositoryIntakeService(store)
    app = FastAPI(title="OpenManus Evidence Cockpit", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        return HTMLResponse(page(store.list_tasks()))

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/intakes", response_class=HTMLResponse)
    async def create_intake(request: Request) -> HTMLResponse:
        if request.headers.get("HX-Request") != "true":
            return HTMLResponse(
                error_fragment("A mutação exige uma requisição HTMX explícita."),
                status_code=400,
            )
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" not in content_type:
            return HTMLResponse(
                error_fragment("Formato de formulário não aceito."), status_code=415
            )
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        repository_value = form.get("repository_path", [""])[0]
        request_key = form.get("request_key", [""])[0]
        try:
            repository = Path(repository_value).resolve(strict=True)
        except (OSError, RuntimeError):
            return HTMLResponse(
                error_fragment("O repositório informado não existe."), status_code=422
            )
        if not _inside_allowed_root(repository, roots):
            return HTMLResponse(
                error_fragment("O caminho está fora das raízes permitidas."),
                status_code=403,
            )
        try:
            task = intake.run(repository, request_key=request_key)
        except IdempotencyConflict as exc:
            return HTMLResponse(error_fragment(str(exc)), status_code=409)
        except (ValueError, RuntimeError) as exc:
            return HTMLResponse(error_fragment(str(exc)), status_code=422)
        return HTMLResponse(
            task_card(task),
            status_code=201,
            headers={"X-OpenManus-Task-Id": task["id"]},
        )

    @app.get("/tasks/{task_id}", response_class=HTMLResponse)
    async def get_task(task_id: str) -> HTMLResponse:
        try:
            return HTMLResponse(task_card(store.get_task(task_id)))
        except KeyError:
            return HTMLResponse(error_fragment("Tarefa não encontrada."), status_code=404)

    @app.get("/tasks/{task_id}/graph.svg")
    async def graph(task_id: str) -> Response:
        try:
            document = store.export_task(task_id)
        except KeyError:
            return Response(status_code=404)
        return Response(trace_svg(document), media_type="image/svg+xml")

    @app.get("/tasks/{task_id}/export.json")
    async def export(task_id: str) -> JSONResponse:
        try:
            document = store.export_task(task_id)
        except KeyError:
            return JSONResponse({"error": "task not found"}, status_code=404)
        document["verification"] = store.verify_task(task_id).to_dict()
        return JSONResponse(document)

    @app.get("/assets/cockpit.css")
    async def css() -> FileResponse:
        return FileResponse(STATIC_ROOT / "cockpit.css", media_type="text/css")

    @app.get("/assets/verifier.js")
    async def verifier_js() -> FileResponse:
        return FileResponse(
            STATIC_ROOT / "verifier.js", media_type="application/javascript"
        )

    @app.get("/assets/wasm_exec.js")
    async def wasm_runtime() -> Response:
        path = assets / "wasm_exec.js"
        if not path.is_file():
            return Response(status_code=404)
        return FileResponse(path, media_type="application/javascript")

    @app.get("/assets/verifier.wasm")
    async def verifier_wasm() -> Response:
        path = assets / "verifier.wasm"
        if not path.is_file():
            return Response(status_code=404)
        return FileResponse(path, media_type="application/wasm")

    return app
