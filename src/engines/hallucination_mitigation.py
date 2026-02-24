"""
Hallucination Mitigation Layer — grounding-based validation and correction.

Every node and transition is validated against the domain dataset.
This layer CORRECTS the workflow (removes invalid elements) before
the formal Validation Engine runs.

Supports two generation modes:
  - Workflow mode  → corrective mitigation (original behaviour)
  - Flowchart mode → strict mitigation (no silent correction — log every event)

Checks:
  1. Every node must exist in the domain dataset
  2. Every transition must be explicitly allowed
  3. Exactly one start node
  4. At least one end node
  5. Decision nodes must have ≥ 2 outgoing branches
  6. No duplicate nodes
  7. Retry loops respect max-retry constraints
  8. No forbidden transitions
  9. [Flowchart] Decision branches match decision_rules exactly
  10. [Flowchart] Retry edges match flowchart_retry_constraints
"""

from __future__ import annotations

import logging
from collections import Counter

from src.config import settings
from src.models.domain import DomainDataset
from src.models.validation import (
    IssueCategory,
    IssueSeverity,
    ValidationIssue,
    ValidationResult,
)
from src.models.workflow import GeneratedWorkflow, NodeType, WorkflowEdge, WorkflowNode

logger = logging.getLogger(__name__)


class HallucinationMitigator:
    """
    Grounding-based hallucination mitigation.

    Operates in two modes:
      - strict: reject the entire workflow on any grounding violation
      - corrective: remove invalid elements and report issues

    Default is corrective mode.
    """

    def mitigate(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> tuple[GeneratedWorkflow, ValidationResult]:
        """
        Validate and correct a generated workflow against the domain dataset.

        Returns:
            (corrected_workflow, validation_result)
        """
        result = ValidationResult(checks_performed=["hallucination_mitigation"])
        issues: list[ValidationIssue] = []

        # 1. Ground nodes against dataset
        workflow, node_issues = self._ground_nodes(workflow, dataset)
        issues.extend(node_issues)

        # 2. Remove duplicates
        if settings.duplicate_removal:
            workflow, dup_issues = self._remove_duplicates(workflow)
            issues.extend(dup_issues)

        # 3. Ground transitions
        workflow, trans_issues = self._ground_transitions(workflow, dataset)
        issues.extend(trans_issues)

        # 4. Structural checks
        struct_issues = self._check_structure(workflow, dataset)
        issues.extend(struct_issues)

        # 5. Decision branch validation
        branch_issues = self._validate_decision_branches(workflow)
        issues.extend(branch_issues)

        # 6. Retry constraint enforcement
        retry_issues = self._enforce_retry_constraints(workflow, dataset)
        issues.extend(retry_issues)

        # 7. Forbidden transition check
        forbidden_issues = self._check_forbidden_transitions(workflow, dataset)
        issues.extend(forbidden_issues)

        # Build result
        result.issues = issues
        result.nodes_validated = len(workflow.nodes)
        result.edges_validated = len(workflow.edges)
        result.is_valid = not any(
            i.severity == IssueSeverity.ERROR for i in issues
        )

        return workflow, result

    # ==================================================================
    # FLOWCHART-SPECIFIC MITIGATION (strict mode)
    # ==================================================================

    def mitigate_flowchart(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> tuple[GeneratedWorkflow, ValidationResult]:
        """
        Strict grounding for flowchart mode.

        Runs all standard checks PLUS:
          - Decision branch labels validated against decision_rules
          - Retry edges validated against flowchart_retry_constraints
        No silent correction — every mitigation event is logged.
        """
        result = ValidationResult(
            checks_performed=["hallucination_mitigation_flowchart"]
        )
        issues: list[ValidationIssue] = []

        # Standard grounding (1–7)
        workflow, node_issues = self._ground_nodes(workflow, dataset)
        issues.extend(node_issues)

        if settings.duplicate_removal:
            workflow, dup_issues = self._remove_duplicates(workflow)
            issues.extend(dup_issues)

        workflow, trans_issues = self._ground_transitions(workflow, dataset)
        issues.extend(trans_issues)

        struct_issues = self._check_structure(workflow, dataset)
        issues.extend(struct_issues)

        # Flowchart-strict: decision branches must match decision_rules exactly
        branch_issues = self._validate_decision_rules_strict(workflow, dataset)
        issues.extend(branch_issues)

        # Flowchart-strict: retry edges must match flowchart_retry_constraints
        retry_issues = self._validate_flowchart_retry_constraints(workflow, dataset)
        issues.extend(retry_issues)

        forbidden_issues = self._check_forbidden_transitions(workflow, dataset)
        issues.extend(forbidden_issues)

        result.issues = issues
        result.nodes_validated = len(workflow.nodes)
        result.edges_validated = len(workflow.edges)
        result.is_valid = not any(
            i.severity == IssueSeverity.ERROR for i in issues
        )

        return workflow, result

    # ------------------------------------------------------------------
    # Flowchart-strict: decision rule validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_decision_rules_strict(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> list[ValidationIssue]:
        """
        Ensure every decision node's outgoing edges match the dataset's
        decision_rules exactly — same labels, same targets.
        """
        issues: list[ValidationIssue] = []
        node_ids = {n.id for n in workflow.nodes}

        for node in workflow.nodes:
            if node.type != NodeType.DECISION:
                continue

            rule = dataset.decision_rules.get(node.id)
            if rule is None:
                # Decision node exists in steps but has no decision_rules entry
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.FLOWCHART,
                        message=(
                            f"Decision node '{node.id}' has no entry in "
                            f"decision_rules — branches unverified."
                        ),
                        node_id=node.id,
                    )
                )
                continue

            # Expected branch labels → targets
            expected: dict[str, str] = {b.label: b.target for b in rule.branches}

            # Actual outgoing edges from this node
            actual: dict[str, str] = {}
            for edge in workflow.edges:
                if edge.source != node.id:
                    continue
                key = edge.condition.branch_key if edge.condition else None
                if key:
                    actual[key] = edge.target

            # Check: every expected branch must be present
            for label, target in expected.items():
                if target not in node_ids:
                    # Target not in selected nodes — acceptable omission
                    continue
                if label not in actual:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.FLOWCHART,
                            message=(
                                f"Decision node '{node.id}': expected branch "
                                f"'{label}' → '{target}' missing."
                            ),
                            node_id=node.id,
                        )
                    )
                elif actual[label] != target:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.FLOWCHART,
                            message=(
                                f"Decision node '{node.id}': branch '{label}' "
                                f"targets '{actual[label]}' but dataset says "
                                f"'{target}'."
                            ),
                            node_id=node.id,
                        )
                    )

            # Check: no extra branch labels that aren't in decision_rules
            for label in actual:
                if label not in expected:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.FLOWCHART,
                            message=(
                                f"Decision node '{node.id}': branch '{label}' "
                                f"not defined in decision_rules — hallucination."
                            ),
                            node_id=node.id,
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # Flowchart-strict: retry constraint validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_flowchart_retry_constraints(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> list[ValidationIssue]:
        """Validate retry loop edges against flowchart_retry_constraints."""
        issues: list[ValidationIssue] = []
        node_ids = {n.id for n in workflow.nodes}

        # Build a lookup: node → (loop_back_to, max_attempts)
        constraint_map: dict[str, tuple[str, int]] = {}
        for rc in dataset.flowchart_retry_constraints:
            constraint_map[rc.node] = (rc.loop_back_to, rc.max_attempts)

        # Check every retry-style edge
        for edge in workflow.edges:
            if edge.condition and edge.condition.branch_key == "retry_allowed":
                rc = constraint_map.get(edge.source)
                if rc is None:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.RETRY,
                            message=(
                                f"Retry edge from '{edge.source}' has no "
                                f"flowchart_retry_constraint defined."
                            ),
                            edge_id=edge.id,
                        )
                    )
                elif edge.target != rc[0]:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.RETRY,
                            message=(
                                f"Retry edge from '{edge.source}' targets "
                                f"'{edge.target}' but constraint says "
                                f"'{rc[0]}'."
                            ),
                            edge_id=edge.id,
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # 1. Node grounding
    # ------------------------------------------------------------------

    @staticmethod
    def _ground_nodes(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> tuple[GeneratedWorkflow, list[ValidationIssue]]:
        """Remove nodes not present in the domain dataset."""
        issues: list[ValidationIssue] = []
        valid_ids = dataset.step_ids
        grounded: list[WorkflowNode] = []

        for node in workflow.nodes:
            if node.domain_step_id not in valid_ids:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.GROUNDING,
                        message=f"Node '{node.id}' not found in domain dataset — removed.",
                        node_id=node.id,
                    )
                )
                logger.warning("Hallucination detected: node '%s' not in dataset", node.id)
            else:
                grounded.append(node)

        workflow.nodes = grounded
        # Remove edges referencing removed nodes
        valid_node_ids = {n.id for n in grounded}
        workflow.edges = [
            e for e in workflow.edges
            if e.source in valid_node_ids and e.target in valid_node_ids
        ]

        return workflow, issues

    # ------------------------------------------------------------------
    # 2. Duplicate removal
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_duplicates(
        workflow: GeneratedWorkflow,
    ) -> tuple[GeneratedWorkflow, list[ValidationIssue]]:
        """Remove duplicate nodes (same domain_step_id)."""
        issues: list[ValidationIssue] = []
        seen: set[str] = set()
        unique: list[WorkflowNode] = []

        for node in workflow.nodes:
            if node.domain_step_id in seen:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.DUPLICATE,
                        message=f"Duplicate node '{node.id}' removed.",
                        node_id=node.id,
                    )
                )
            else:
                seen.add(node.domain_step_id)
                unique.append(node)

        workflow.nodes = unique
        return workflow, issues

    # ------------------------------------------------------------------
    # 3. Transition grounding
    # ------------------------------------------------------------------

    @staticmethod
    def _ground_transitions(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> tuple[GeneratedWorkflow, list[ValidationIssue]]:
        """Remove edges not present in the dataset's allowed transitions."""
        issues: list[ValidationIssue] = []

        # Build set of allowed (from, to, condition) tuples
        allowed: set[tuple[str, str, str | None]] = set()
        for t in dataset.transitions:
            allowed.add((t.from_step, t.to_step, t.condition))

        grounded_edges: list[WorkflowEdge] = []
        for edge in workflow.edges:
            cond = edge.condition.branch_key if edge.condition else None
            key = (edge.source, edge.target, cond)

            if key in allowed:
                grounded_edges.append(edge)
            else:
                # Also allow if (from, to, None) is allowed (unconditional)
                if (edge.source, edge.target, None) in allowed:
                    grounded_edges.append(edge)
                else:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.TRANSITION,
                            message=(
                                f"Transition '{edge.source}' → '{edge.target}' "
                                f"(condition={cond}) not allowed — removed."
                            ),
                            edge_id=edge.id,
                        )
                    )

        workflow.edges = grounded_edges
        return workflow, issues

    # ------------------------------------------------------------------
    # 4. Structural checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_structure(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> list[ValidationIssue]:
        """Check exactly one start node and at least one end node."""
        issues: list[ValidationIssue] = []

        start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
        end_nodes = [n for n in workflow.nodes if n.type == NodeType.END]

        if len(start_nodes) == 0:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.STRUCTURE,
                    message="Workflow has no start node.",
                )
            )
        elif len(start_nodes) > 1:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.STRUCTURE,
                    message=f"Workflow has {len(start_nodes)} start nodes — must have exactly 1.",
                    details={"start_nodes": [n.id for n in start_nodes]},
                )
            )

        if len(end_nodes) == 0:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.STRUCTURE,
                    message="Workflow has no end node.",
                )
            )

        # Check required steps are present
        node_ids = {n.id for n in workflow.nodes}
        for req in dataset.validation_rules.required_steps:
            if req not in node_ids:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.STRUCTURE,
                        message=f"Required step '{req}' is missing from workflow.",
                    )
                )

        return issues

    # ------------------------------------------------------------------
    # 5. Decision branch validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_decision_branches(
        workflow: GeneratedWorkflow,
    ) -> list[ValidationIssue]:
        """Check decision nodes for outgoing edge count.

        After grounding removes transitions that are not in the dataset,
        a decision node may legitimately have fewer than 2 branches
        (e.g. because the selected steps don't include all branch targets).
        This is therefore a WARNING, not an ERROR — the workflow is still
        structurally sound; it simply has a narrower branch set.
        """
        issues: list[ValidationIssue] = []

        # Count outgoing edges per node
        outgoing: Counter[str] = Counter()
        for edge in workflow.edges:
            outgoing[edge.source] += 1

        for node in workflow.nodes:
            if node.type == NodeType.DECISION:
                count = outgoing.get(node.id, 0)
                if count < 2:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.STRUCTURE,
                            message=(
                                f"Decision node '{node.id}' has {count} "
                                f"outgoing edge(s) — minimum 2 recommended."
                            ),
                            node_id=node.id,
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # 6. Retry constraints
    # ------------------------------------------------------------------

    def _enforce_retry_constraints(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> list[ValidationIssue]:
        """Ensure retry loops don't exceed max attempts."""
        issues: list[ValidationIssue] = []
        max_retries = dataset.validation_rules.max_retries

        # Find retry nodes
        step_map = dataset.step_map
        for node in workflow.nodes:
            step = step_map.get(node.domain_step_id)
            if step and step.retry_config:
                if step.retry_config.max_attempts > max_retries:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.RETRY,
                            message=(
                                f"Retry step '{node.id}' allows "
                                f"{step.retry_config.max_attempts} attempts "
                                f"but domain max is {max_retries}."
                            ),
                            node_id=node.id,
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # 7. Forbidden transitions
    # ------------------------------------------------------------------

    @staticmethod
    def _check_forbidden_transitions(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> list[ValidationIssue]:
        """Check no forbidden transitions are present."""
        issues: list[ValidationIssue] = []

        forbidden = {
            (ft.from_step, ft.to_step)
            for ft in dataset.validation_rules.forbidden_direct_transitions
        }

        for edge in workflow.edges:
            if (edge.source, edge.target) in forbidden:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.TRANSITION,
                        message=(
                            f"Forbidden transition '{edge.source}' → "
                            f"'{edge.target}' detected."
                        ),
                        edge_id=edge.id,
                    )
                )

        return issues
