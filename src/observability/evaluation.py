"""
Evaluation Runner — research evaluation mode.

Wraps the standard pipeline response and aggregates all observability
data into a single EvaluationReport.  Does NOT modify any core logic —
purely a post-processing aggregator.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.models.domain import DomainDataset
from src.models.validation import IssueSeverity, ValidationResult
from src.models.workflow import GeneratedWorkflow, NodeType
from src.observability.models import (
    EvaluationReport,
    ExplainabilityEntry,
    HallucinationMetrics,
    StageMetric,
)


class EvaluationRunner:
    """
    Build a complete EvaluationReport from a finished pipeline response.

    Usage::

        runner = EvaluationRunner()
        report = runner.build_report(
            response=response,
            dataset=dataset,
            stage_metrics=profiler.collect(),
            hallucination_metrics=h_metrics,
            explainability=explainability_entry,
            request_instruction=request.instruction,
            request_mode=request.mode.value,
        )
    """

    def build_report(
        self,
        *,
        success: bool,
        workflow: GeneratedWorkflow | None,
        validation: ValidationResult | None,
        dataset: DomainDataset,
        stage_metrics: list[StageMetric],
        hallucination_metrics: HallucinationMetrics,
        explainability: ExplainabilityEntry,
        request_instruction: str,
        request_mode: str,
    ) -> EvaluationReport:
        """Aggregate all observability data into a single report."""

        # Run ID: deterministic from instruction + domain + timestamp
        run_id = self._generate_run_id(
            request_instruction, dataset.domain, request_mode,
        )

        # Total duration
        total_duration = sum(m.duration_ms for m in stage_metrics)

        # Structure counts
        node_count = len(workflow.nodes) if workflow else 0
        edge_count = len(workflow.edges) if workflow else 0
        decision_count = (
            sum(1 for n in workflow.nodes if n.type == NodeType.DECISION)
            if workflow
            else 0
        )

        # Validation
        error_count = validation.error_count if validation else 0
        warning_count = validation.warning_count if validation else 0

        # Completeness: fraction of required steps present in output
        completeness = self._compute_completeness(workflow, dataset)

        # Grounding score: inverse of hallucination score
        grounding_score = round(1.0 - hallucination_metrics.hallucination_score, 4)

        return EvaluationReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            instruction=request_instruction,
            domain=dataset.domain,
            mode=request_mode,
            stage_metrics=stage_metrics,
            total_duration_ms=round(total_duration, 3),
            hallucination_metrics=hallucination_metrics,
            node_count=node_count,
            edge_count=edge_count,
            decision_node_count=decision_count,
            validation_passed=success,
            validation_error_count=error_count,
            validation_warning_count=warning_count,
            explainability=explainability,
            completeness_score=completeness,
            grounding_score=grounding_score,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_run_id(instruction: str, domain: str, mode: str) -> str:
        """Deterministic but unique-ish run identifier."""
        raw = f"eval::{domain}::{mode}::{instruction}::{time.perf_counter_ns()}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"eval_{domain}_{digest}"

    @staticmethod
    def _compute_completeness(
        workflow: GeneratedWorkflow | None,
        dataset: DomainDataset,
    ) -> float:
        """Fraction of dataset required_steps present in the workflow."""
        if not workflow:
            return 0.0
        required = dataset.validation_rules.required_steps
        if not required:
            return 1.0
        present = {n.id for n in workflow.nodes}
        found = sum(1 for r in required if r in present)
        return round(found / len(required), 4)
