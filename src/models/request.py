"""API request / response models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.models.validation import ValidationResult
from src.models.workflow import GeneratedWorkflow
from src.observability.models import ObservabilityResult


class GenerationMode(StrEnum):
    """Supported generation modes."""

    WORKFLOW = "workflow"
    FLOWCHART = "flowchart"


class GenerateRequest(BaseModel):
    """Workflow generation request."""

    instruction: str = Field(..., min_length=3, max_length=2000)
    mode: GenerationMode = Field(default=GenerationMode.WORKFLOW)
    domain_hint: str | None = Field(default=None, max_length=64)
    include_optional_steps: bool = Field(default=True)
    custom_steps: list[str] = Field(default_factory=list)
    use_local_model: bool = Field(default=False)
    minimal: bool = Field(default=False)
    evaluation_mode: bool = Field(
        default=False,
        description="Enable research evaluation mode — collects full observability data",
    )


class ValidateRequest(BaseModel):
    """Request to validate an existing workflow."""

    workflow: GeneratedWorkflow
    domain: str | None = None


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str
    message: str
    field: str | None = None


class PipelineMetrics(BaseModel):
    """Timing and stats from the pipeline execution."""

    parse_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    mitigation_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    layout_time_ms: float = 0.0
    total_time_ms: float = 0.0
    domain_selected: str = ""
    domain_confidence: float = 0.0
    nodes_generated: int = 0
    edges_generated: int = 0


class GenerateResponse(BaseModel):
    """Workflow generation response."""

    success: bool
    workflow: GeneratedWorkflow | None = None
    validation: ValidationResult | None = None
    metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
    errors: list[ErrorDetail] = Field(default_factory=list)
    observability: ObservabilityResult | None = Field(
        default=None,
        description="Present when evaluation_mode is enabled",
    )


class DomainInfo(BaseModel):
    """Summary of an available domain."""

    domain: str
    display_name: str
    description: str
    keywords: list[str]
    step_count: int
    transition_count: int


class DomainListResponse(BaseModel):
    """Response listing all available domains."""

    domains: list[DomainInfo] = Field(default_factory=list)
    count: int = 0


class ValidateResponse(BaseModel):
    """Validation-only response."""

    success: bool
    validation: ValidationResult
    errors: list[ErrorDetail] = Field(default_factory=list)
