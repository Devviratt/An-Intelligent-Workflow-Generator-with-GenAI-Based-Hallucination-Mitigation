"""
Hallucination Metrics Collector — passive extraction from existing mitigation results.

Operates entirely on the ValidationResult produced by HallucinationMitigator.
Does NOT modify core mitigation logic — read-only analysis.
"""

from __future__ import annotations

from src.models.validation import (
    IssueCategory,
    IssueSeverity,
    ValidationResult,
)
from src.models.workflow import GeneratedWorkflow
from src.observability.models import HallucinationMetrics


class HallucinationMetricsCollector:
    """
    Extracts quantified hallucination statistics from a mitigation pass.

    Usage::

        collector = HallucinationMetricsCollector()
        metrics = collector.collect(
            mitigation_result=mitigation_result,
            workflow_before_nodes=original_node_count,
            workflow_before_edges=original_edge_count,
            workflow_after=corrected_workflow,
        )
    """

    def collect(
        self,
        mitigation_result: ValidationResult,
        workflow_before_nodes: int,
        workflow_before_edges: int,
        workflow_after: GeneratedWorkflow,
    ) -> HallucinationMetrics:
        """
        Analyse the mitigation result and produce quantified metrics.

        Args:
            mitigation_result: The ValidationResult from mitigate() / mitigate_flowchart()
            workflow_before_nodes: Node count BEFORE mitigation
            workflow_before_edges: Edge count BEFORE mitigation
            workflow_after: The corrected workflow AFTER mitigation
        """
        issues = mitigation_result.issues

        nodes_removed = workflow_before_nodes - len(workflow_after.nodes)
        edges_removed = workflow_before_edges - len(workflow_after.edges)

        grounding_violations = sum(
            1 for i in issues if i.category == IssueCategory.GROUNDING
        )
        duplicates_removed = sum(
            1 for i in issues if i.category == IssueCategory.DUPLICATE
        )
        structural_issues = sum(
            1 for i in issues if i.category == IssueCategory.STRUCTURE
        )
        branch_issues = sum(
            1 for i in issues
            if i.category in (IssueCategory.FLOWCHART, IssueCategory.LOGICAL)
            and "branch" in i.message.lower()
        )
        retry_issues = sum(
            1 for i in issues if i.category == IssueCategory.RETRY
        )
        forbidden_transitions = sum(
            1 for i in issues if i.category == IssueCategory.TRANSITION
        )

        # Grounding rates
        node_grounding_rate = (
            1.0
            if workflow_before_nodes == 0
            else max(0.0, 1.0 - (nodes_removed / workflow_before_nodes))
        )
        edge_grounding_rate = (
            1.0
            if workflow_before_edges == 0
            else max(0.0, 1.0 - (edges_removed / workflow_before_edges))
        )

        # Composite hallucination score: weighted sum of violation ratios
        total_issues = len(issues)
        total_items = max(workflow_before_nodes + workflow_before_edges, 1)
        hallucination_score = min(1.0, total_issues / total_items)

        return HallucinationMetrics(
            total_nodes_checked=workflow_before_nodes,
            total_edges_checked=workflow_before_edges,
            nodes_removed=max(0, nodes_removed),
            edges_removed=max(0, edges_removed),
            duplicates_removed=duplicates_removed,
            forbidden_transitions_found=forbidden_transitions,
            grounding_violations=grounding_violations,
            structural_issues=structural_issues,
            branch_issues=branch_issues,
            retry_issues=retry_issues,
            node_grounding_rate=round(node_grounding_rate, 4),
            edge_grounding_rate=round(edge_grounding_rate, 4),
            hallucination_score=round(hallucination_score, 4),
        )
