"""Validation result models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class IssueSeverity(StrEnum):
    """Severity of a validation issue."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueCategory(StrEnum):
    """Category of validation issue."""

    SCHEMA = "schema"
    LOGICAL = "logical"
    DEPENDENCY = "dependency"
    CYCLE = "cycle"
    DEPTH = "depth"
    GROUNDING = "grounding"
    STRUCTURE = "structure"
    TRANSITION = "transition"
    DUPLICATE = "duplicate"
    RETRY = "retry"
    FLOWCHART = "flowchart"
    REACHABILITY = "reachability"


class ValidationIssue(BaseModel):
    """A single validation issue found in a workflow."""

    severity: IssueSeverity
    category: IssueCategory
    message: str
    node_id: str | None = None
    edge_id: str | None = None
    details: dict[str, object] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Aggregated validation result for a workflow."""

    is_valid: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    nodes_validated: int = 0
    edges_validated: int = 0
    checks_performed: list[str] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one (mutating)."""
        self.issues.extend(other.issues)
        self.nodes_validated += other.nodes_validated
        self.edges_validated += other.edges_validated
        self.checks_performed.extend(other.checks_performed)
        if other.error_count > 0:
            self.is_valid = False
