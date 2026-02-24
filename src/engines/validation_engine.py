"""
Validation Engine — comprehensive, multi-pass validation of generated workflows.

Validation passes (workflow mode):
  1. Schema validation      — structural correctness of the workflow model
  2. Logical validation     — semantic consistency (node types, edge targets)
  3. Dependency validation  — all required predecessors are present
  4. Cycle detection        — detect and report graph cycles (except allowed loops)
  5. Depth verification     — ensure workflow doesn't exceed max depth

Additional passes (flowchart mode):
  6. Reachability           — all nodes reachable from start
  7. Orphan detection       — no orphan nodes (no edges at all)
  8. Decision branch ≥ 2    — strict (ERROR level for flowcharts)
  9. Duplication guard      — no exponential node duplication
"""

from __future__ import annotations

import logging
from collections import deque

from src.config import settings
from src.models.domain import DomainDataset
from src.models.validation import (
    IssueCategory,
    IssueSeverity,
    ValidationIssue,
    ValidationResult,
)
from src.models.workflow import GeneratedWorkflow, NodeType

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Multi-pass workflow validator.

    Each pass is independent and can be run in isolation.
    Results are merged into a single ValidationResult.
    """

    def validate(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset | None = None,
    ) -> ValidationResult:
        """Run all validation passes and return merged result."""
        result = ValidationResult()

        # Pass 1: Schema
        schema_result = self._validate_schema(workflow)
        result.merge(schema_result)

        # Pass 2: Logical
        logical_result = self._validate_logical(workflow)
        result.merge(logical_result)

        # Pass 3: Dependency
        if dataset:
            dep_result = self._validate_dependencies(workflow, dataset)
            result.merge(dep_result)

        # Pass 4: Cycle detection
        cycle_result = self._detect_cycles(workflow, dataset)
        result.merge(cycle_result)

        # Pass 5: Depth verification
        depth_result = self._verify_depth(workflow, dataset)
        result.merge(depth_result)

        return result

    def validate_flowchart(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> ValidationResult:
        """Run all validation passes including flowchart-specific ones."""
        # Standard passes
        result = self.validate(workflow, dataset)

        # Flowchart pass 6: reachability
        reach_result = self._validate_reachability(workflow)
        result.merge(reach_result)

        # Flowchart pass 7: orphan detection
        orphan_result = self._detect_orphans(workflow)
        result.merge(orphan_result)

        # Flowchart pass 8: strict decision branch count
        branch_result = self._validate_decision_branch_count_strict(workflow)
        result.merge(branch_result)

        # Flowchart pass 9: duplication guard
        dup_result = self._detect_exponential_duplication(workflow)
        result.merge(dup_result)

        return result

    # ------------------------------------------------------------------
    # Pass 1: Schema validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_schema(workflow: GeneratedWorkflow) -> ValidationResult:
        """Validate structural correctness."""
        result = ValidationResult(checks_performed=["schema_validation"])
        issues: list[ValidationIssue] = []

        # Check workflow has nodes and edges
        if not workflow.nodes:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.SCHEMA,
                    message="Workflow has no nodes.",
                )
            )

        if not workflow.edges:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.SCHEMA,
                    message="Workflow has no edges.",
                )
            )

        # Check node IDs are unique
        node_ids = [n.id for n in workflow.nodes]
        seen: set[str] = set()
        for nid in node_ids:
            if nid in seen:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.SCHEMA,
                        message=f"Duplicate node ID: '{nid}'.",
                        node_id=nid,
                    )
                )
            seen.add(nid)

        # Check edge IDs are unique
        edge_ids = [e.id for e in workflow.edges]
        seen_edges: set[str] = set()
        for eid in edge_ids:
            if eid in seen_edges:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.SCHEMA,
                        message=f"Duplicate edge ID: '{eid}'.",
                        edge_id=eid,
                    )
                )
            seen_edges.add(eid)

        # Check edges reference existing nodes
        valid_node_ids = set(node_ids)
        for edge in workflow.edges:
            if edge.source not in valid_node_ids:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.SCHEMA,
                        message=f"Edge '{edge.id}' references unknown source '{edge.source}'.",
                        edge_id=edge.id,
                    )
                )
            if edge.target not in valid_node_ids:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.SCHEMA,
                        message=f"Edge '{edge.id}' references unknown target '{edge.target}'.",
                        edge_id=edge.id,
                    )
                )

        # Check max nodes constraint
        if len(workflow.nodes) > settings.max_workflow_nodes:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.SCHEMA,
                    message=(
                        f"Workflow has {len(workflow.nodes)} nodes — "
                        f"max allowed is {settings.max_workflow_nodes}."
                    ),
                )
            )

        result.issues = issues
        result.nodes_validated = len(workflow.nodes)
        result.edges_validated = len(workflow.edges)
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result

    # ------------------------------------------------------------------
    # Pass 2: Logical validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_logical(workflow: GeneratedWorkflow) -> ValidationResult:
        """Validate semantic consistency."""
        result = ValidationResult(checks_performed=["logical_validation"])
        issues: list[ValidationIssue] = []

        node_map = workflow.node_map

        # Start nodes should have no incoming edges
        incoming: dict[str, int] = {n.id: 0 for n in workflow.nodes}
        outgoing: dict[str, int] = {n.id: 0 for n in workflow.nodes}
        for edge in workflow.edges:
            if edge.target in incoming:
                incoming[edge.target] += 1
            if edge.source in outgoing:
                outgoing[edge.source] += 1

        for node in workflow.nodes:
            if node.type == NodeType.START:
                if incoming.get(node.id, 0) > 0:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.LOGICAL,
                            message=f"Start node '{node.id}' has incoming edges.",
                            node_id=node.id,
                        )
                    )
                if outgoing.get(node.id, 0) == 0:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.LOGICAL,
                            message=f"Start node '{node.id}' has no outgoing edges.",
                            node_id=node.id,
                        )
                    )

            elif node.type == NodeType.END:
                if outgoing.get(node.id, 0) > 0:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.LOGICAL,
                            message=f"End node '{node.id}' has outgoing edges.",
                            node_id=node.id,
                        )
                    )

            elif node.type == NodeType.PROCESS:
                if incoming.get(node.id, 0) == 0:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.LOGICAL,
                            message=f"Process node '{node.id}' has no incoming edges.",
                            node_id=node.id,
                        )
                    )

        # Self-loops
        for edge in workflow.edges:
            if edge.source == edge.target:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.LOGICAL,
                        message=f"Self-loop detected on node '{edge.source}'.",
                        edge_id=edge.id,
                    )
                )

        result.issues = issues
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result

    # ------------------------------------------------------------------
    # Pass 3: Dependency validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dependencies(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> ValidationResult:
        """Check all required predecessor steps are present."""
        result = ValidationResult(checks_performed=["dependency_validation"])
        issues: list[ValidationIssue] = []

        node_ids = {n.id for n in workflow.nodes}

        # Build predecessor map from dataset transitions
        predecessors: dict[str, set[str]] = {}
        for t in dataset.transitions:
            predecessors.setdefault(t.to_step, set()).add(t.from_step)

        # For each node, check that at least one predecessor exists (if expected)
        for node in workflow.nodes:
            if node.type == NodeType.START:
                continue
            expected_preds = predecessors.get(node.id, set())
            if expected_preds and not expected_preds & node_ids:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.DEPENDENCY,
                        message=(
                            f"Node '{node.id}' has no reachable predecessors "
                            f"in the workflow (expected one of: "
                            f"{sorted(expected_preds)})."
                        ),
                        node_id=node.id,
                    )
                )

        result.issues = issues
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result

    # ------------------------------------------------------------------
    # Pass 4: Cycle detection (DFS-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_cycles(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset | None = None,
    ) -> ValidationResult:
        """Detect cycles using iterative DFS with color-marking."""
        result = ValidationResult(checks_performed=["cycle_detection"])
        issues: list[ValidationIssue] = []

        # Build adjacency
        adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for e in workflow.edges:
            if e.source in adj:
                adj[e.source].append(e.target)

        # Allowed retry edges (not counted as cycles)
        allowed_back_edges: set[tuple[str, str]] = set()
        if dataset:
            # Whitelist ALL transitions explicitly defined in the dataset.
            # These are legitimate domain back-edges (e.g. escalate → investigate
            # in incident_response) and should never be flagged as cycles.
            node_ids = {n.id for n in workflow.nodes}
            for t in dataset.transitions:
                if t.from_step in node_ids and t.to_step in node_ids:
                    allowed_back_edges.add((t.from_step, t.to_step))

        # Iterative DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n.id: WHITE for n in workflow.nodes}
        parent: dict[str, str | None] = {n.id: None for n in workflow.nodes}

        def _dfs_from(start: str) -> None:
            stack: list[tuple[str, int]] = [(start, 0)]
            while stack:
                node, idx = stack.pop()
                if idx == 0:
                    if color[node] != WHITE:
                        continue
                    color[node] = GRAY

                neighbours = adj.get(node, [])
                if idx < len(neighbours):
                    stack.append((node, idx + 1))
                    nb = neighbours[idx]
                    if color.get(nb) == GRAY:
                        if (node, nb) not in allowed_back_edges:
                            issues.append(
                                ValidationIssue(
                                    severity=IssueSeverity.ERROR,
                                    category=IssueCategory.CYCLE,
                                    message=f"Cycle detected involving edge '{node}' → '{nb}'.",
                                    details={"from": node, "to": nb},
                                )
                            )
                    elif color.get(nb) == WHITE:
                        parent[nb] = node
                        stack.append((nb, 0))
                else:
                    color[node] = BLACK

        for n in workflow.nodes:
            if color[n.id] == WHITE:
                _dfs_from(n.id)

        result.issues = issues
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result

    # ------------------------------------------------------------------
    # Pass 5: Depth verification
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_depth(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset | None = None,
    ) -> ValidationResult:
        """Verify workflow depth doesn't exceed limits."""
        result = ValidationResult(checks_performed=["depth_verification"])
        issues: list[ValidationIssue] = []

        max_depth = settings.max_workflow_depth
        if dataset and dataset.validation_rules.max_depth:
            max_depth = min(max_depth, dataset.validation_rules.max_depth)

        # BFS to compute depth
        start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
        if not start_nodes:
            return result

        adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for e in workflow.edges:
            if e.source in adj:
                adj[e.source].append(e.target)

        depths: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()
        for sn in start_nodes:
            queue.append((sn.id, 0))
            depths[sn.id] = 0

        while queue:
            nid, depth = queue.popleft()
            for nb in adj.get(nid, []):
                if nb not in depths:
                    depths[nb] = depth + 1
                    queue.append((nb, depth + 1))

        actual_max = max(depths.values()) if depths else 0

        if actual_max > max_depth:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.DEPTH,
                    message=(
                        f"Workflow depth is {actual_max} — "
                        f"max allowed is {max_depth}."
                    ),
                    details={"actual_depth": actual_max, "max_depth": max_depth},
                )
            )

        result.issues = issues
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result

    # ==================================================================
    # Flowchart-specific passes
    # ==================================================================

    # ------------------------------------------------------------------
    # Pass 6: Reachability — all nodes reachable from start
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_reachability(workflow: GeneratedWorkflow) -> ValidationResult:
        """Ensure every node is reachable from the start node via BFS."""
        result = ValidationResult(checks_performed=["flowchart_reachability"])
        issues: list[ValidationIssue] = []

        start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
        if not start_nodes:
            return result

        adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for e in workflow.edges:
            if e.source in adj:
                adj[e.source].append(e.target)

        reachable: set[str] = set()
        queue: deque[str] = deque(s.id for s in start_nodes)
        while queue:
            nid = queue.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for nb in adj.get(nid, []):
                if nb not in reachable:
                    queue.append(nb)

        for node in workflow.nodes:
            if node.id not in reachable:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.REACHABILITY,
                        message=f"Node '{node.id}' is not reachable from start.",
                        node_id=node.id,
                    )
                )

        result.issues = issues
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result

    # ------------------------------------------------------------------
    # Pass 7: Orphan detection — no edges at all
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_orphans(workflow: GeneratedWorkflow) -> ValidationResult:
        """Detect nodes that have zero edges (neither incoming nor outgoing)."""
        result = ValidationResult(checks_performed=["flowchart_orphan_detection"])
        issues: list[ValidationIssue] = []

        connected: set[str] = set()
        for e in workflow.edges:
            connected.add(e.source)
            connected.add(e.target)

        for node in workflow.nodes:
            if node.id not in connected:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.REACHABILITY,
                        message=f"Orphan node '{node.id}' has no edges.",
                        node_id=node.id,
                    )
                )

        result.issues = issues
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result

    # ------------------------------------------------------------------
    # Pass 8: Strict decision branch count (flowchart — ERROR level)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_decision_branch_count_strict(
        workflow: GeneratedWorkflow,
    ) -> ValidationResult:
        """Every decision node MUST have ≥ 2 outgoing edges (ERROR)."""
        result = ValidationResult(
            checks_performed=["flowchart_decision_branch_count"]
        )
        issues: list[ValidationIssue] = []

        from collections import Counter

        outgoing: Counter[str] = Counter()
        for e in workflow.edges:
            outgoing[e.source] += 1

        for node in workflow.nodes:
            if node.type == NodeType.DECISION:
                count = outgoing.get(node.id, 0)
                if count < 2:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.FLOWCHART,
                            message=(
                                f"Decision node '{node.id}' has {count} "
                                f"outgoing edge(s) — minimum 2 required."
                            ),
                            node_id=node.id,
                        )
                    )

        result.issues = issues
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result

    # ------------------------------------------------------------------
    # Pass 9: Exponential duplication guard
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_exponential_duplication(
        workflow: GeneratedWorkflow,
    ) -> ValidationResult:
        """Guard against exponential node duplication (same domain_step_id)."""
        result = ValidationResult(
            checks_performed=["flowchart_duplication_guard"]
        )
        issues: list[ValidationIssue] = []

        from collections import Counter

        id_counts = Counter(n.domain_step_id for n in workflow.nodes)
        for step_id, count in id_counts.items():
            if count > 1:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.DUPLICATE,
                        message=(
                            f"domain_step_id '{step_id}' appears {count} times "
                            f"— exponential duplication detected."
                        ),
                    )
                )

        result.issues = issues
        result.is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return result
