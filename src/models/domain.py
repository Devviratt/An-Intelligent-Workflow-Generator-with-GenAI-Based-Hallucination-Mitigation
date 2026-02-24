"""Domain dataset schema — mirrors the JSON dataset structure exactly."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetryConfig(BaseModel):
    """Retry configuration for a step that supports retries."""

    max_attempts: int = Field(..., ge=1, le=10)
    backoff_seconds: list[int | float] = Field(default_factory=list)
    retry_on: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Flowchart-specific schema extensions
# ---------------------------------------------------------------------------


class DecisionBranch(BaseModel):
    """A single branch in a flowchart decision rule."""

    label: str = Field(..., min_length=1, max_length=32)
    target: str = Field(..., min_length=1, max_length=128)


class DecisionRule(BaseModel):
    """Defines the exact branches for a decision node in a flowchart."""

    branches: list[DecisionBranch] = Field(..., min_length=2)


class FlowchartRetryConstraint(BaseModel):
    """Flowchart-specific retry loop constraint."""

    node: str = Field(..., min_length=1)
    max_attempts: int = Field(..., ge=1, le=10)
    loop_back_to: str = Field(..., min_length=1)


class DomainStep(BaseModel):
    """A single step in a domain dataset."""

    id: str = Field(..., min_length=1, max_length=128)
    label: str = Field(..., min_length=1, max_length=256)
    type: str = Field(..., pattern=r"^(start|end|process|decision)$")
    description: str = Field(default="")
    required: bool = Field(default=False)
    branches: dict[str, str] | None = Field(default=None)
    retry_config: RetryConfig | None = Field(default=None)

    @field_validator("branches")
    @classmethod
    def decision_needs_branches(cls, v: dict[str, str] | None, info: Any) -> dict[str, str] | None:
        step_type = info.data.get("type", "")
        if step_type == "decision" and (v is None or len(v) < 2):
            raise ValueError("Decision nodes must have at least 2 branches")
        return v


class DomainTransition(BaseModel):
    """An allowed transition between steps."""

    model_config = ConfigDict(populate_by_name=True)

    from_step: str = Field(..., alias="from", min_length=1)
    to_step: str = Field(..., alias="to", min_length=1)
    condition: str | None = Field(default=None)


class ForbiddenTransition(BaseModel):
    """A transition that must never appear."""

    model_config = ConfigDict(populate_by_name=True)

    from_step: str = Field(..., alias="from", min_length=1)
    to_step: str = Field(..., alias="to", min_length=1)


class ValidationRules(BaseModel):
    """Domain-level validation constraints."""

    max_depth: int = Field(default=20, ge=1)
    max_retries: int = Field(default=3, ge=0)
    required_steps: list[str] = Field(default_factory=list)
    forbidden_direct_transitions: list[ForbiddenTransition] = Field(default_factory=list)


class DomainMetadata(BaseModel):
    """Domain metadata — SLA, compliance, criticality."""

    avg_steps: int = Field(default=0, ge=0)
    sla_seconds: int = Field(default=0, ge=0)
    compliance: list[str] = Field(default_factory=list)
    criticality: str = Field(default="medium")


class DomainDataset(BaseModel):
    """Complete domain dataset — the single source of truth for workflow generation."""

    domain: str = Field(..., min_length=1, max_length=64)
    display_name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="")
    version: str = Field(default="1.0.0")
    keywords: list[str] = Field(default_factory=list)
    start_node: str = Field(..., min_length=1)
    end_node: str = Field(..., min_length=1)
    error_terminal: str | None = Field(default=None)
    metadata: DomainMetadata = Field(default_factory=DomainMetadata)
    steps: list[DomainStep] = Field(default_factory=list, min_length=2)
    transitions: list[DomainTransition] = Field(default_factory=list, min_length=1)
    validation_rules: ValidationRules = Field(default_factory=ValidationRules)

    # ── Flowchart extensions (optional — backward-compatible) ──
    end_nodes: list[str] = Field(default_factory=list)
    decision_rules: dict[str, DecisionRule] = Field(default_factory=dict)
    flowchart_retry_constraints: list[FlowchartRetryConstraint] = Field(default_factory=list)

    # ----- Computed helpers (not serialised) -----

    @property
    def step_ids(self) -> set[str]:
        return {s.id for s in self.steps}

    @property
    def step_map(self) -> dict[str, DomainStep]:
        return {s.id: s for s in self.steps}

    @property
    def required_step_ids(self) -> set[str]:
        return {s.id for s in self.steps if s.required}

    @property
    def transition_index(self) -> dict[str, list[DomainTransition]]:
        """Index of transitions keyed by source step id."""
        idx: dict[str, list[DomainTransition]] = {}
        for t in self.transitions:
            idx.setdefault(t.from_step, []).append(t)
        return idx

    @property
    def effective_end_nodes(self) -> list[str]:
        """Return explicit end_nodes list, falling back to [end_node] + error_terminal."""
        if self.end_nodes:
            return list(self.end_nodes)
        result = [self.end_node]
        if self.error_terminal:
            result.append(self.error_terminal)
        return result

    @field_validator("steps")
    @classmethod
    def must_have_start_and_end(cls, v: list[DomainStep]) -> list[DomainStep]:
        types = {s.type for s in v}
        if "start" not in types:
            raise ValueError("Dataset must contain at least one start node")
        if "end" not in types:
            raise ValueError("Dataset must contain at least one end node")
        return v
