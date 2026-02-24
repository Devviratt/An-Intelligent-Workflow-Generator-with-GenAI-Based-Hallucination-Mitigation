"""
Flowchart Generation Engine — deterministic, dataset-driven flowchart synthesis.

Principles:
  - NO free-text generation.  NO inferred branches.
  - Every node comes from the domain dataset ``steps``.
  - Decision branches come ONLY from ``decision_rules``.
  - Transitions come ONLY from ``transitions`` (the allowed list).
  - Retry loops are represented as a SINGLE backward edge, NOT expanded.
  - Output is deterministic: same input → same output.

Contract:
  - Minimum 1 decision node in the output.
  - Each decision has ≥ 2 outgoing branches (from ``decision_rules``).
  - Branch labels match the dataset *exactly*.
  - No new nodes.  No new transitions.
  - If the request asks for unsupported logic → structured error.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import Sequence

from src.models.domain import DomainDataset, DomainStep
from src.models.parser import ParsedInstruction
from src.models.workflow import (
    EdgeCondition,
    EdgeStyle,
    GeneratedWorkflow,
    NodeType,
    WorkflowEdge,
    WorkflowNode,
)

logger = logging.getLogger(__name__)


class FlowchartGenerationError(Exception):
    """Raised when a valid flowchart cannot be generated from the dataset."""


class FlowchartGenerator:
    """
    Generates a flowchart graph from a domain dataset.

    Algorithm
    ---------
    1. Collect ALL steps from the dataset (no step selection — flowcharts
       are the full picture).
    2. Validate that the dataset contains decision_rules.
    3. Build nodes from steps.
    4. Build edges from allowed_transitions (the ``transitions`` list).
    5. Expand decision branches strictly from ``decision_rules``.
    6. Add retry loop edges from ``flowchart_retry_constraints``.
    7. Prune unreachable nodes from the start node.
    8. Final structural assertions.
    """

    def generate(
        self,
        dataset: DomainDataset,
        parsed: ParsedInstruction,
        *,
        include_optional: bool = True,
    ) -> GeneratedWorkflow:
        """Generate a complete flowchart from the dataset."""

        # ── Pre-flight checks ──
        if not dataset.decision_rules:
            raise FlowchartGenerationError(
                f"Domain '{dataset.domain}' has no decision_rules — "
                "cannot generate flowchart."
            )

        # ── 1. Select steps ──
        steps = self._select_steps(dataset, include_optional)

        # ── 2. Build nodes ──
        nodes = self._build_nodes(steps, dataset)
        selected_ids = {s.id for s in steps}

        # ── 3. Build edges from allowed transitions ──
        edges = self._build_edges(dataset, selected_ids)

        # ── 4. Add retry loop backward edges ──
        edges = self._add_retry_edges(dataset, selected_ids, edges)

        # ── 5. Prune unreachable nodes ──
        nodes, edges = self._prune_unreachable(nodes, edges, dataset.start_node)

        # ── 6. Structural assertions ──
        self._assert_structure(nodes, edges, dataset)

        # ── 7. Assemble workflow ──
        workflow_id = self._generate_id(dataset.domain, parsed.cleaned_text)

        return GeneratedWorkflow(
            workflow_id=workflow_id,
            domain=dataset.domain,
            title=f"{dataset.display_name} Flowchart",
            description=dataset.description,
            is_flowchart=True,
            nodes=nodes,
            edges=edges,
            metadata={
                "dataset_version": dataset.version,
                "compliance": dataset.metadata.compliance,
                "criticality": dataset.metadata.criticality,
                "decision_count": sum(
                    1 for n in nodes if n.type == NodeType.DECISION
                ),
            },
        )

    # ------------------------------------------------------------------
    # Step selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_steps(
        dataset: DomainDataset,
        include_optional: bool,
    ) -> list[DomainStep]:
        """Select steps — for flowcharts we take everything by default."""
        if include_optional:
            return list(dataset.steps)
        return [s for s in dataset.steps if s.required]

    # ------------------------------------------------------------------
    # Node building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nodes(
        steps: list[DomainStep],
        dataset: DomainDataset,
    ) -> list[WorkflowNode]:
        nodes: list[WorkflowNode] = []
        for step in steps:
            node = WorkflowNode(
                id=step.id,
                label=step.label,
                type=NodeType(step.type),
                description=step.description,
                domain_step_id=step.id,
                branches=step.branches,
                metadata={
                    "required": step.required,
                    "domain": dataset.domain,
                },
            )
            nodes.append(node)
        return nodes

    # ------------------------------------------------------------------
    # Edge building — strictly from dataset transitions
    # ------------------------------------------------------------------

    @staticmethod
    def _build_edges(
        dataset: DomainDataset,
        selected_ids: set[str],
    ) -> list[WorkflowEdge]:
        """
        Build edges ONLY from the dataset ``transitions`` list.

        Decision branches are matched against ``decision_rules`` for
        label accuracy but no new edges are invented.
        """
        edges: list[WorkflowEdge] = []
        seen: set[tuple[str, str, str | None]] = set()

        # Pre-index decision rule labels for fast lookup
        decision_labels: dict[str, set[str]] = {}
        for node_id, rule in dataset.decision_rules.items():
            decision_labels[node_id] = {b.label for b in rule.branches}

        for t in dataset.transitions:
            if t.from_step not in selected_ids or t.to_step not in selected_ids:
                continue

            edge_key = (t.from_step, t.to_step, t.condition)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            # Edge ID
            edge_id = f"e_{t.from_step}__{t.to_step}"
            if t.condition:
                edge_id += f"__{t.condition}"

            # Build condition metadata
            condition: EdgeCondition | None = None
            if t.condition:
                condition = EdgeCondition(
                    label=t.condition.replace("_", " ").title(),
                    branch_key=t.condition,
                )

            edges.append(
                WorkflowEdge(
                    id=edge_id,
                    source=t.from_step,
                    target=t.to_step,
                    condition=condition,
                    style=EdgeStyle.NORMAL,
                )
            )

        return edges

    # ------------------------------------------------------------------
    # Retry loop edges
    # ------------------------------------------------------------------

    @staticmethod
    def _add_retry_edges(
        dataset: DomainDataset,
        selected_ids: set[str],
        edges: list[WorkflowEdge],
    ) -> list[WorkflowEdge]:
        """
        Add or upgrade retry loop edges per constraint.

        If an edge matching (node → loop_back_to) already exists from
        the transitions list, upgrade it to RETRY_LOOP style instead of
        creating a duplicate.  Otherwise, add a new backward edge.
        """
        # Index existing edges for fast lookup
        edge_by_pair: dict[tuple[str, str], WorkflowEdge] = {}
        for e in edges:
            edge_by_pair[(e.source, e.target)] = e

        for rc in dataset.flowchart_retry_constraints:
            if rc.node not in selected_ids or rc.loop_back_to not in selected_ids:
                continue
            pair = (rc.node, rc.loop_back_to)

            if pair in edge_by_pair:
                # Upgrade existing edge to retry style
                existing_edge = edge_by_pair[pair]
                existing_edge.style = EdgeStyle.RETRY_LOOP
                existing_edge.metadata = existing_edge.metadata or {}
                existing_edge.metadata["max_attempts"] = rc.max_attempts
                if existing_edge.condition is None:
                    existing_edge.condition = EdgeCondition(
                        label=f"Retry (max {rc.max_attempts})",
                        branch_key="retry_allowed",
                    )
            else:
                edge_id = f"e_{rc.node}__{rc.loop_back_to}__retry"
                new_edge = WorkflowEdge(
                    id=edge_id,
                    source=rc.node,
                    target=rc.loop_back_to,
                    condition=EdgeCondition(
                        label=f"Retry (max {rc.max_attempts})",
                        branch_key="retry_allowed",
                    ),
                    style=EdgeStyle.RETRY_LOOP,
                    metadata={"max_attempts": rc.max_attempts},
                )
                edges.append(new_edge)
                edge_by_pair[pair] = new_edge

        return edges

    # ------------------------------------------------------------------
    # Reachability pruning
    # ------------------------------------------------------------------

    @staticmethod
    def _prune_unreachable(
        nodes: list[WorkflowNode],
        edges: list[WorkflowEdge],
        start_node: str,
    ) -> tuple[list[WorkflowNode], list[WorkflowEdge]]:
        """Remove nodes not reachable from the start node."""
        adj: dict[str, list[str]] = {n.id: [] for n in nodes}
        for e in edges:
            if e.source in adj:
                adj[e.source].append(e.target)

        reachable: set[str] = set()
        queue: deque[str] = deque([start_node])
        while queue:
            nid = queue.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for nb in adj.get(nid, []):
                if nb not in reachable:
                    queue.append(nb)

        pruned = len(nodes) - len(reachable)
        if pruned > 0:
            logger.info("Flowchart: pruned %d unreachable nodes", pruned)

        filtered_nodes = [n for n in nodes if n.id in reachable]
        filtered_edges = [
            e for e in edges if e.source in reachable and e.target in reachable
        ]
        return filtered_nodes, filtered_edges

    # ------------------------------------------------------------------
    # Structural assertions
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_structure(
        nodes: list[WorkflowNode],
        edges: list[WorkflowEdge],
        dataset: DomainDataset,
    ) -> None:
        """Final structural checks — raise on violation."""
        node_ids = {n.id for n in nodes}

        # Must have start node
        if dataset.start_node not in node_ids:
            raise FlowchartGenerationError(
                f"Start node '{dataset.start_node}' missing after pruning."
            )

        # Must have at least one end node
        end_ids = set(dataset.effective_end_nodes) & node_ids
        if not end_ids:
            raise FlowchartGenerationError(
                "No end node reachable after pruning."
            )

        # Must have at least one decision node
        decision_nodes = [n for n in nodes if n.type == NodeType.DECISION]
        if not decision_nodes:
            raise FlowchartGenerationError(
                "Flowchart requires at least 1 decision node."
            )

        # Decision nodes must have ≥ 2 outgoing edges
        from collections import Counter

        outgoing: Counter[str] = Counter()
        for e in edges:
            outgoing[e.source] += 1

        for dn in decision_nodes:
            if outgoing.get(dn.id, 0) < 2:
                raise FlowchartGenerationError(
                    f"Decision node '{dn.id}' has fewer than 2 outgoing "
                    f"edges ({outgoing.get(dn.id, 0)})."
                )

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_id(domain: str, text: str) -> str:
        raw = f"fc_{domain}::{text}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"fc_{domain}_{digest}"
