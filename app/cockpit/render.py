"""Server-rendered HTML and SVG views for the cockpit."""

from __future__ import annotations

from html import escape
import json
from typing import Any, Mapping, Sequence


HTMX_URL = "https://cdn.jsdelivr.net/npm/htmx.org@2.0.10/dist/htmx.min.js"
HTMX_INTEGRITY = "sha384-H5SrcfygHmAuTDZphMHqBJLc3FhssKjG7w/CeCpFReSfwBWDTKpkzPP8c+cLsK+V"


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def task_card(task: Mapping[str, Any]) -> str:
    task_id = _text(task["id"])
    status = _text(task["status"])
    subject = task.get("subject", {})
    if subject.get("kind") == "agent_run":
        subject_label = (
            f"{subject.get('agent', 'Agent')} / request "
            f"{str(subject.get('request_sha256', ''))[:12]}"
        )
    else:
        subject_label = subject.get("path", "unknown")
    subject_path = _text(subject_label)
    verification_url = f"/tasks/{task_id}/export.json"
    return f"""
<article class="task-card" data-task-id="{task_id}">
  <header>
    <div>
      <p class="eyebrow">trace {task_id[:8]}</p>
      <h2>{subject_path}</h2>
    </div>
    <span class="status status-{status}">{status}</span>
  </header>
  <dl>
    <div><dt>Evidências</dt><dd>{int(task.get('evidence_count', 0))}</dd></div>
    <div><dt>Efeitos incertos</dt><dd>{int(task.get('unresolved_effects', 0))}</dd></div>
    <div><dt>Atualizado</dt><dd>{_text(task.get('updated_at', ''))}</dd></div>
  </dl>
  <img class="trace" src="/tasks/{task_id}/graph.svg" alt="Fluxo de eventos desta análise">
  <div class="actions">
    <a href="{verification_url}">Exportar prova JSON</a>
    <button type="button" data-verify-trace data-export-url="{verification_url}">
      Verificar no navegador (WASM)
    </button>
  </div>
  <output class="wasm-result" aria-live="polite"></output>
</article>
""".strip()


def error_fragment(message: str) -> str:
    return f'<section class="error" role="alert">{_text(message)}</section>'


def page(tasks: Sequence[Mapping[str, Any]], *, default_path: str = "") -> str:
    task_markup = "\n".join(task_card(task) for task in tasks)
    if not task_markup:
        task_markup = (
            '<section class="empty">Nenhuma análise registrada. '
            "O primeiro envio cria uma trilha durável.</section>"
        )
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenManus Evidence Cockpit</title>
  <link rel="stylesheet" href="/assets/cockpit.css">
  <script src="{HTMX_URL}" integrity="{HTMX_INTEGRITY}" crossorigin="anonymous" defer></script>
  <script src="/assets/wasm_exec.js" defer></script>
  <script src="/assets/verifier.js" defer></script>
</head>
<body>
  <main>
    <header class="hero">
      <p class="eyebrow">OPENMANUS / EVIDENCE COCKPIT</p>
      <h1>Engenharia reversa que termina em prova, não em opinião.</h1>
      <p>Inspecione um clone real, registre a decisão de reuso e verifique a trilha em Python, Go ou WASM.</p>
    </header>

    <section class="intake-panel" aria-labelledby="intake-title">
      <div>
        <p class="eyebrow">NOVO INTAKE</p>
        <h2 id="intake-title">Analisar repositório local</h2>
      </div>
      <form hx-post="/intakes" hx-target="#task-stream" hx-swap="afterbegin">
        <label>Repositório
          <input name="repository_path" value="{_text(default_path)}" required>
        </label>
        <label>Chave idempotente
          <input name="request_key" placeholder="grok-bot-018-v1" required>
        </label>
        <button type="submit">Executar análise</button>
      </form>
    </section>

    <section id="task-stream" class="task-stream" aria-live="polite">
      {task_markup}
    </section>
  </main>
</body>
</html>
"""


def trace_svg(document: Mapping[str, Any]) -> str:
    events = document.get("events", [])
    task = document.get("task", {})
    width = max(720, 88 + len(events) * 196)
    height = 220
    status_colors = {
        "created": "#7c9cff",
        "accepted": "#66d9c2",
        "running": "#f8c76c",
        "completed": "#73df91",
        "failed": "#ff7b87",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">Trilha {_text(task.get('id', ''))}</title>",
        "<desc id=\"desc\">Eventos encadeados por SHA-256</desc>",
        "<style>text{font-family:ui-monospace,monospace;fill:#e8ecf7}.kind{font-size:13px;font-weight:700}.meta{font-size:11px;fill:#9aa6bd}.line{stroke:#45516b;stroke-width:3}.node{stroke:#151b2a;stroke-width:4}</style>",
        '<rect width="100%" height="100%" rx="18" fill="#0d1220"/>',
    ]
    for index, event in enumerate(events):
        x = 72 + index * 196
        if index:
            parts.append(
                f'<line class="line" x1="{x - 150}" y1="92" x2="{x - 34}" y2="92"/>'
            )
        state = str(event.get("state_after", ""))
        color = status_colors.get(state, "#9aa6bd")
        kind = _text(event.get("kind", ""))
        event_hash = _text(str(event.get("event_hash", ""))[:10])
        parts.extend(
            [
                f'<circle class="node" cx="{x}" cy="92" r="28" fill="{color}"/>',
                f'<text x="{x}" y="97" text-anchor="middle" class="kind" fill="#07100b">{index + 1}</text>',
                f'<text x="{x}" y="143" text-anchor="middle" class="kind">{kind}</text>',
                f'<text x="{x}" y="165" text-anchor="middle" class="meta">{event_hash}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def pretty_json(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
