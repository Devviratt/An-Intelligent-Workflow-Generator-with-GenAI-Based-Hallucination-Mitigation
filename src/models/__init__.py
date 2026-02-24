"""Core models package — shared Pydantic schemas for the entire system."""

from src.models.domain import (
    DomainDataset,
    DomainMetadata,
    DomainStep,
    DomainTransition,
    RetryConfig,
    ValidationRules,
    ForbiddenTransition,
    DecisionBranch,
    DecisionRule,
    FlowchartRetryConstraint,
)
from src.models.workflow import (
    Coordinate,
    WorkflowEdge,
    WorkflowNode,
    GeneratedWorkflow,
    LayoutInfo,
    NodeType,
    EdgeCondition,
    EdgeStyle,
)
from src.models.request import (
    GenerateRequest,
    GenerationMode,
    ValidateRequest,
    GenerateResponse,
    ValidateResponse,
    DomainListResponse,
    DomainInfo,
    ErrorDetail,
    PipelineMetrics,
)
from src.models.validation import (
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
    IssueCategory,
)
from src.models.parser import (
    ParsedInstruction,
    DomainMatch,
    ExtractedKeywords,
)
from src.observability.models import (
    EvaluationReport,
    ExplainabilityEntry,
    HallucinationMetrics,
    NodeProvenance,
    ObservabilityResult,
    StageMetric,
    StageName,
)

__all__ = [
    # domain
    "DomainDataset",
    "DomainMetadata",
    "DomainStep",
    "DomainTransition",
    "RetryConfig",
    "ValidationRules",
    "ForbiddenTransition",
    "DecisionBranch",
    "DecisionRule",
    "FlowchartRetryConstraint",
    # workflow
    "Coordinate",
    "WorkflowEdge",
    "WorkflowNode",
    "GeneratedWorkflow",
    "LayoutInfo",
    "NodeType",
    "EdgeCondition",
    "EdgeStyle",
    # request/response
    "GenerateRequest",
    "GenerationMode",
    "ValidateRequest",
    "GenerateResponse",
    "ValidateResponse",
    "DomainListResponse",
    "DomainInfo",
    "ErrorDetail",
    "PipelineMetrics",
    # validation
    "ValidationResult",
    "ValidationIssue",
    "IssueSeverity",
    "IssueCategory",
    # parser
    "ParsedInstruction",
    "DomainMatch",
    "ExtractedKeywords",
    # observability
    "EvaluationReport",
    "ExplainabilityEntry",
    "HallucinationMetrics",
    "NodeProvenance",
    "ObservabilityResult",
    "StageMetric",
    "StageName",
]
