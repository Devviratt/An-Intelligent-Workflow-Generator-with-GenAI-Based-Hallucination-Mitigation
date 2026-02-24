"""
Landing Page — developer-friendly root endpoint.

Returns JSON by default, or a glassmorphism HTML page when
the client sends ``Accept: text/html``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, ORJSONResponse

router = APIRouter(include_in_schema=False)

_SERVICE_INFO: dict[str, str] = {
    "service": "Intelligent Workflow Generator",
    "version": "1.0.0",
    "description": (
        "Dataset-driven workflow & flowchart engine with "
        "hallucination mitigation"
    ),
    "docs_url": "/docs",
    "health_url": "/api/v1/health",
}

_LANDING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Intelligent Workflow Generator</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

  body{
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
    font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
    background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
    color:#e2e8f0;
    overflow:hidden;
  }

  /* ambient glow blobs */
  body::before,body::after{
    content:'';
    position:fixed;
    border-radius:50%;
    filter:blur(120px);
    opacity:.35;
    pointer-events:none;
    z-index:0;
  }
  body::before{
    width:480px;height:480px;
    background:radial-gradient(circle,#7c3aed,transparent 70%);
    top:-80px;left:-100px;
    animation:drift 18s ease-in-out infinite alternate;
  }
  body::after{
    width:420px;height:420px;
    background:radial-gradient(circle,#06b6d4,transparent 70%);
    bottom:-60px;right:-80px;
    animation:drift 22s ease-in-out infinite alternate-reverse;
  }
  @keyframes drift{
    0%{transform:translate(0,0) scale(1)}
    100%{transform:translate(60px,40px) scale(1.12)}
  }

  /* card entry */
  @keyframes fadeUp{
    from{opacity:0;transform:translateY(32px) scale(.97)}
    to{opacity:1;transform:translateY(0) scale(1)}
  }

  .card{
    position:relative;
    z-index:1;
    max-width:520px;
    width:92%;
    padding:48px 40px 40px;
    border-radius:24px;
    background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.12);
    backdrop-filter:blur(24px) saturate(1.4);
    -webkit-backdrop-filter:blur(24px) saturate(1.4);
    box-shadow:
      0 8px 32px rgba(0,0,0,.35),
      inset 0 1px 0 rgba(255,255,255,.08);
    animation:fadeUp .7s cubic-bezier(.22,1,.36,1) both;
  }

  .badge{
    display:inline-block;
    font-size:.65rem;
    font-weight:700;
    letter-spacing:.12em;
    text-transform:uppercase;
    padding:4px 12px;
    border-radius:999px;
    background:rgba(124,58,237,.25);
    border:1px solid rgba(124,58,237,.4);
    color:#c4b5fd;
    margin-bottom:20px;
  }

  h1{
    font-size:1.65rem;
    font-weight:800;
    line-height:1.25;
    letter-spacing:-.02em;
    background:linear-gradient(135deg,#e2e8f0 0%,#7dd3fc 60%,#c4b5fd 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    margin-bottom:14px;
  }

  .desc{
    font-size:.92rem;
    line-height:1.6;
    color:#94a3b8;
    margin-bottom:32px;
  }

  .actions{
    display:flex;
    gap:14px;
    flex-wrap:wrap;
  }

  .btn{
    flex:1 1 auto;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    gap:8px;
    padding:12px 22px;
    font-size:.88rem;
    font-weight:600;
    border-radius:14px;
    text-decoration:none;
    transition:all .25s cubic-bezier(.22,1,.36,1);
    cursor:pointer;
    border:none;
    white-space:nowrap;
  }

  .btn-primary{
    background:linear-gradient(135deg,#7c3aed,#6d28d9);
    color:#fff;
    box-shadow:0 4px 14px rgba(124,58,237,.35);
  }
  .btn-primary:hover{
    transform:translateY(-2px);
    box-shadow:0 6px 20px rgba(124,58,237,.5);
  }

  .btn-secondary{
    background:rgba(255,255,255,.07);
    color:#cbd5e1;
    border:1px solid rgba(255,255,255,.12);
  }
  .btn-secondary:hover{
    background:rgba(255,255,255,.12);
    transform:translateY(-2px);
  }

  .meta{
    margin-top:28px;
    padding-top:20px;
    border-top:1px solid rgba(255,255,255,.08);
    display:flex;
    gap:24px;
    flex-wrap:wrap;
  }
  .meta-item{
    font-size:.75rem;
    color:#64748b;
  }
  .meta-item span{
    color:#94a3b8;
    font-weight:600;
  }

  /* svg icons inline */
  .icon{width:16px;height:16px;flex-shrink:0}
</style>
</head>
<body>
  <div class="card">
    <div class="badge">v1.0.0</div>
    <h1>Intelligent Workflow Generator</h1>
    <p class="desc">
      Dataset-driven workflow &amp; flowchart engine with
      GenAI-based hallucination mitigation.
      Deterministic output &mdash; no external AI APIs.
    </p>

    <div class="actions">
      <a href="/docs" class="btn btn-primary">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
          <line x1="10" y1="9" x2="8" y2="9"/>
        </svg>
        API Docs
      </a>
      <a href="/api/v1/health" class="btn btn-secondary">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
        Health Check
      </a>
    </div>

    <div class="meta">
      <div class="meta-item">Engine &nbsp;<span>Dataset-Driven</span></div>
      <div class="meta-item">Modes &nbsp;<span>Workflow · Flowchart</span></div>
      <div class="meta-item">Layout &nbsp;<span>Deterministic BFS</span></div>
    </div>
  </div>
</body>
</html>
"""


@router.get("/", response_model=None)
async def root(request: Request) -> HTMLResponse | ORJSONResponse:
    """
    Developer landing page.

    - ``Accept: text/html`` → glassmorphism HTML page
    - Otherwise → JSON service descriptor
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=_LANDING_HTML, status_code=200)
    return ORJSONResponse(content=_SERVICE_INFO, status_code=200)
