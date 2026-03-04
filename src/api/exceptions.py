"""
Global Exception Handlers — structured JSON error responses.

Catches all known application exceptions and returns consistent
error payloads.  Stack traces are never exposed in production.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse

from src.engines.flowchart_generator import FlowchartGenerationError
from src.engines.llm_flowchart_generator import LLMFlowchartGenerationError
from src.engines.llm_workflow_generator import LLMWorkflowGenerationError
from src.engines.workflow_generator import WorkflowGenerationError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers to the FastAPI app."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> ORJSONResponse:
        """Pydantic / request-body validation failures."""
        errors = exc.errors()
        details = [
            {
                "field": " → ".join(str(loc) for loc in e.get("loc", ())),
                "message": e.get("msg", ""),
                "type": e.get("type", ""),
            }
            for e in errors
        ]
        return ORJSONResponse(
            status_code=422,
            content={
                "error_type": "ValidationError",
                "message": "Request validation failed",
                "details": details,
            },
        )

    @app.exception_handler(WorkflowGenerationError)
    async def workflow_generation_error_handler(
        request: Request,
        exc: WorkflowGenerationError,
    ) -> ORJSONResponse:
        """Workflow generator raised a known domain error."""
        logger.warning("WorkflowGenerationError: %s", exc)
        return ORJSONResponse(
            status_code=422,
            content={
                "error_type": "WorkflowGenerationError",
                "message": str(exc),
                "details": None,
            },
        )

    @app.exception_handler(FlowchartGenerationError)
    async def flowchart_generation_error_handler(
        request: Request,
        exc: FlowchartGenerationError,
    ) -> ORJSONResponse:
        """Flowchart generator raised a known domain error."""
        logger.warning("FlowchartGenerationError: %s", exc)
        return ORJSONResponse(
            status_code=422,
            content={
                "error_type": "FlowchartGenerationError",
                "message": str(exc),
                "details": None,
            },
        )

    @app.exception_handler(LLMWorkflowGenerationError)
    async def llm_workflow_generation_error_handler(
        request: Request,
        exc: LLMWorkflowGenerationError,
    ) -> ORJSONResponse:
        """LLM workflow generator raised an error."""
        logger.warning("LLMWorkflowGenerationError: %s", exc)
        return ORJSONResponse(
            status_code=422,
            content={
                "error_type": "LLMWorkflowGenerationError",
                "message": str(exc),
                "details": None,
            },
        )

    @app.exception_handler(LLMFlowchartGenerationError)
    async def llm_flowchart_generation_error_handler(
        request: Request,
        exc: LLMFlowchartGenerationError,
    ) -> ORJSONResponse:
        """LLM flowchart generator raised an error."""
        logger.warning("LLMFlowchartGenerationError: %s", exc)
        return ORJSONResponse(
            status_code=422,
            content={
                "error_type": "LLMFlowchartGenerationError",
                "message": str(exc),
                "details": None,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> ORJSONResponse:
        """FastAPI HTTPException — includes 404, 405, etc."""
        return ORJSONResponse(
            status_code=exc.status_code,
            content={
                "error_type": "HTTPException",
                "message": exc.detail,
                "details": None,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> ORJSONResponse:
        """Catch-all for unexpected errors — no stack trace leakage."""
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return ORJSONResponse(
            status_code=500,
            content={
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": None,
            },
        )
