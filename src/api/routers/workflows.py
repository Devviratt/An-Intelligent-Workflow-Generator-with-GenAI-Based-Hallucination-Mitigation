"""
Workflow Router — generate, validate, and evaluate workflows.

All routes are under ``/api/v1`` (prefix added by the server).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from src.models.request import (
    GenerateRequest,
    GenerateResponse,
    ValidateRequest,
    ValidateResponse,
)

router = APIRouter(tags=["Workflows"])


# ------------------------------------------------------------------
# Method-not-allowed helpers
# ------------------------------------------------------------------

@router.get(
    "/generate",
    include_in_schema=False,
)
async def generate_method_not_allowed(request: Request) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=405,
        content={
            "error": "Method Not Allowed",
            "expected_method": "POST",
            "endpoint": "/api/v1/generate",
        },
    )


@router.get(
    "/validate",
    include_in_schema=False,
)
async def validate_method_not_allowed(request: Request) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=405,
        content={
            "error": "Method Not Allowed",
            "expected_method": "POST",
            "endpoint": "/api/v1/validate",
        },
    )


@router.get(
    "/evaluate",
    include_in_schema=False,
)
async def evaluate_method_not_allowed(request: Request) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=405,
        content={
            "error": "Method Not Allowed",
            "expected_method": "POST",
            "endpoint": "/api/v1/evaluate",
        },
    )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate a workflow",
    description=(
        "Generate a complete workflow or flowchart from a natural-language "
        "instruction.  The engine selects the best domain dataset, builds a "
        "deterministic node/edge graph, runs hallucination mitigation, "
        "validates the result, and applies BFS-based layout.\n\n"
        "**Modes:**\n"
        "- `workflow` — standard process graph with start/end/process/decision nodes\n"
        "- `flowchart` — branch-aware diagram with decision rules and retry constraints\n\n"
        "Set `evaluation_mode: true` to collect observability data "
        "(stage profiling, hallucination metrics, explainability)."
    ),
)
async def generate_workflow(request: GenerateRequest) -> GenerateResponse:
    from src.api.server import get_pipeline

    pipeline = get_pipeline()
    return await pipeline.generate(request)


@router.post(
    "/validate",
    response_model=ValidateResponse,
    summary="Validate an existing workflow",
    description=(
        "Run the full validation suite on a previously generated workflow.  "
        "Checks include schema conformance, logical consistency, dependency "
        "ordering, cycle detection, depth limits, grounding verification, "
        "and reachability analysis."
    ),
)
async def validate_workflow(request: ValidateRequest) -> ValidateResponse:
    from src.api.server import get_pipeline

    pipeline = get_pipeline()
    return await pipeline.validate(request)


@router.post(
    "/evaluate",
    response_model=GenerateResponse,
    summary="Generate with full evaluation",
    description=(
        "Same as `/generate` but forces `evaluation_mode: true`.  The "
        "response includes a complete `observability` payload with per-stage "
        "profiling, hallucination metrics, per-node explainability, and a "
        "research-grade evaluation report."
    ),
)
async def evaluate_workflow(request: GenerateRequest) -> GenerateResponse:
    from src.api.server import get_pipeline

    request.evaluation_mode = True
    pipeline = get_pipeline()
    return await pipeline.generate(request)
