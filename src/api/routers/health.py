"""
Health Router — operational health and readiness probe.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["Operations"])


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str = "healthy"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    dataset_count: int = 0
    modes_supported: list[str] = Field(
        default_factory=lambda: ["workflow", "flowchart"],
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns operational status, UTC timestamp, loaded dataset count, "
        "and supported generation modes."
    ),
)
async def health_check() -> HealthResponse:
    from src.api.server import get_pipeline

    pipeline = get_pipeline()
    dataset_count = len(pipeline.dataset_engine.all_datasets())

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_count=dataset_count,
        modes_supported=["workflow", "flowchart"],
    )
