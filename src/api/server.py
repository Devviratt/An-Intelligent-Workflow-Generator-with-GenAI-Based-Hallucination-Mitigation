"""
FastAPI Server — production-grade async REST API.

Architecture:
  - All domain routes under ``/api/v1/`` prefix
  - Glassmorphism landing page at ``/``
  - Structured logging middleware
  - Global exception handlers (no stack-trace leakage)
  - Proper CORS (no wildcard)
  - Rich Swagger metadata
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.exceptions import register_exception_handlers
from src.api.landing import router as landing_router
from src.api.middleware import RequestLoggingMiddleware
from src.api.routers.domains import router as domains_router
from src.api.routers.health import router as health_router
from src.api.routers.rag import router as rag_router
from src.api.routers.workflows import router as workflows_router
from src.pipeline import Pipeline

logger = logging.getLogger(__name__)

# =====================================================================
# Pipeline singleton
# =====================================================================

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Return the initialised pipeline — raises if not ready."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised")
    return _pipeline


# =====================================================================
# Lifespan
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    logger.info("Initialising pipeline…")
    _pipeline = Pipeline()
    await _pipeline.initialise()
    logger.info(
        "Pipeline ready — %d domains loaded",
        len(_pipeline.dataset_engine.all_datasets()),
    )
    yield
    logger.info("Shutting down")


# =====================================================================
# Application
# =====================================================================

_API_DESCRIPTION = """\
## Intelligent Workflow Generator

A **production-grade, dataset-driven** workflow and flowchart engine with
built-in **hallucination mitigation** — no external AI API dependencies.

### Generation Modes

| Mode | Description |
|------|-------------|
| **workflow** | Standard process graph with start / end / process / decision nodes |
| **flowchart** | Branch-aware diagram with decision rules and retry constraints |

### Hallucination Mitigation

Every generated node and edge is validated against the domain dataset:

1. **Grounding check** — reject nodes not traceable to a dataset step
2. **Duplicate removal** — deduplicate semantically identical nodes
3. **Transition validation** — enforce allowed/forbidden transitions
4. **Structural integrity** — verify reachability, depth limits, cycles

### Deterministic Layout

A single-pass BFS engine assigns coordinates (depth × branch index)
so the same input always produces the same visual layout — no randomness.

### Observability (Evaluation Mode)

Set `evaluation_mode: true` to collect per-stage profiling, hallucination
metrics, per-node explainability provenance, and a research-grade evaluation report.
"""

app = FastAPI(
    title="Intelligent Workflow Generator",
    description=_API_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "Workflow Generator Team",
        "url": "https://github.com/workflow-generator",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    terms_of_service="https://github.com/workflow-generator/terms",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Middleware (order matters — outermost first) ──
app.add_middleware(RequestLoggingMiddleware)
cors_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]
extra_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if extra_origins:
    cors_origins.extend(
        origin.strip() for origin in extra_origins.split(",") if origin.strip()
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers ──
register_exception_handlers(app)

# ── Routers ──
app.include_router(landing_router)
app.include_router(workflows_router, prefix="/api/v1")
app.include_router(domains_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
app.include_router(rag_router)

# ── Static files (frontend) ──
static_dir = Path(__file__).parent.parent / "api" / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
