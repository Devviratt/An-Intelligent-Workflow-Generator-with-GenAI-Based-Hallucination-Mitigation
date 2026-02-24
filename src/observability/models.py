"""
Observability data models — typed schemas for performance monitoring,
hallucination metrics, explainability, and research evaluation.

All models are Pydantic v2, serialisable, and backward-compatible
(every field has a default).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# =====================================================================
# Stage profiling
# =====================================================================


class StageName(StrEnum):
    """Pipeline stage identifiers for profiling."""

    PARSE = "parse"
    GENERATE = "generate"
    MITIGATE = "mitigate"
    VALIDATE = "validate"
    LAYOUT = "layout"
    EXPLAINABILITY = "explainability"
    LOCAL_MODEL = "local_model"
    TOTAL = "total"


class StageMetric(BaseModel):
    """Performance metrics captured for a single pipeline stage."""

    stage: StageName
    duration_ms: float = Field(default=0.0, ge=0.0)
    memory_delta_kb: float = Field(default=0.0, description="Approx RSS delta in KB")
    input_count: int = Field(default=0, ge=0, description="Items entering the stage")
    output_count: int = Field(default=0, ge=0, description="Items leaving the stage")
    metadata: dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# Hallucination metrics
# =====================================================================


class HallucinationMetrics(BaseModel):
    """
    Quantified hallucination statistics extracted from a mitigation pass.

    All fields are passive — computed from the existing ValidationResult
    without modifying core mitigation logic.
    """

    total_nodes_checked: int = Field(default=0, ge=0)
    total_edges_checked: int = Field(default=0, ge=0)
    nodes_removed: int = Field(default=0, ge=0)
    edges_removed: int = Field(default=0, ge=0)
    duplicates_removed: int = Field(default=0, ge=0)
    forbidden_transitions_found: int = Field(default=0, ge=0)
    grounding_violations: int = Field(default=0, ge=0)
    structural_issues: int = Field(default=0, ge=0)
    branch_issues: int = Field(default=0, ge=0)
    retry_issues: int = Field(default=0, ge=0)

    # Computed scores
    node_grounding_rate: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of nodes that passed grounding (1.0 = perfect)",
    )
    edge_grounding_rate: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of edges that passed grounding",
    )
    hallucination_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Composite score: 0.0 = no hallucination, 1.0 = fully hallucinated",
    )


# =====================================================================
# Explainability
# =====================================================================


class NodeProvenance(BaseModel):
    """Provenance record for a single workflow node — where it came from."""

    node_id: str
    domain_step_id: str = ""
    source: str = Field(
        default="dataset",
        description="Origin: 'dataset' | 'custom_step' | 'inferred'",
    )
    dataset_domain: str = ""
    dataset_version: str = ""
    step_required: bool = False
    step_type: str = ""
    grounding_status: str = Field(
        default="grounded",
        description="'grounded' | 'removed' | 'unverified'",
    )
    transition_count: int = Field(
        default=0, ge=0,
        description="Number of outgoing edges from this node",
    )


class ExplainabilityEntry(BaseModel):
    """
    Complete explainability metadata for a generated workflow.

    Provides per-node provenance, generation lineage, and decision
    rationale so the output is fully auditable.
    """

    node_provenance: list[NodeProvenance] = Field(default_factory=list)
    generation_mode: str = ""
    domain_used: str = ""
    dataset_version: str = ""
    instruction_keywords: list[str] = Field(default_factory=list)
    domain_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    total_dataset_steps: int = Field(default=0, ge=0)
    steps_selected: int = Field(default=0, ge=0)
    steps_pruned: int = Field(default=0, ge=0)
    decision_nodes_count: int = Field(default=0, ge=0)
    retry_edges_count: int = Field(default=0, ge=0)


# =====================================================================
# Evaluation report (research mode)
# =====================================================================


class EvaluationReport(BaseModel):
    """
    Comprehensive evaluation report for research/benchmarking.

    Aggregates all observability data into a single serialisable record
    suitable for offline analysis, comparison across runs, and CI metrics.
    """

    # Identity
    run_id: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    instruction: str = ""
    domain: str = ""
    mode: str = ""

    # Profiling
    stage_metrics: list[StageMetric] = Field(default_factory=list)
    total_duration_ms: float = Field(default=0.0, ge=0.0)

    # Hallucination
    hallucination_metrics: HallucinationMetrics = Field(
        default_factory=HallucinationMetrics,
    )

    # Structure
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    decision_node_count: int = Field(default=0, ge=0)
    validation_passed: bool = True
    validation_error_count: int = Field(default=0, ge=0)
    validation_warning_count: int = Field(default=0, ge=0)

    # Explainability
    explainability: ExplainabilityEntry = Field(
        default_factory=ExplainabilityEntry,
    )

    # Quality scores
    completeness_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of required steps present in output",
    )
    grounding_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1.0 - hallucination_score",
    )


# =====================================================================
# Aggregated observability result (attached to response)
# =====================================================================


class ObservabilityResult(BaseModel):
    """
    Top-level observability container attached to a GenerateResponse.

    Contains all monitoring data without altering the core response shape.
    """

    stage_metrics: list[StageMetric] = Field(default_factory=list)
    hallucination_metrics: HallucinationMetrics = Field(
        default_factory=HallucinationMetrics,
    )
    explainability: ExplainabilityEntry = Field(
        default_factory=ExplainabilityEntry,
    )
    evaluation: EvaluationReport | None = Field(
        default=None,
        description="Present only when evaluation_mode is enabled",
    )
