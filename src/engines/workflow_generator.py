"""
Workflow Generation Engine — deterministic, dataset-driven workflow synthesis.

Principles:
  - NO free-text generation
  - Every node comes from the domain dataset
  - Transitions come only from the dataset's allowed transitions
  - Custom steps are validated against the domain and rejected if unsupported
  - Output is deterministic: same input → same output
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import Sequence

from src.models.domain import DomainDataset, DomainStep, DomainTransition
from src.models.parser import ParsedInstruction
from src.models.workflow import (
    EdgeCondition,
    GeneratedWorkflow,
    NodeType,
    WorkflowEdge,
    WorkflowNode,
)

logger = logging.getLogger(__name__)


class WorkflowGenerationError(Exception):
    """Raised when workflow cannot be generated from the dataset."""


class WorkflowGenerator:
    """
    Generates a workflow graph from a domain dataset + parsed instruction.

    Algorithm:
      1. Select steps from dataset (required + optional based on flags)
      2. Validate any custom steps against the domain
      3. Build node list from selected steps
      4. Build edge list from dataset transitions (only between selected nodes)
      5. Ensure connectivity: prune unreachable nodes
      6. Generate deterministic IDs
    """

    def generate(
        self,
        dataset: DomainDataset,
        parsed: ParsedInstruction,
        include_optional: bool = True,
        custom_steps: Sequence[str] | None = None,
    ) -> GeneratedWorkflow:
        """Generate a complete workflow from the dataset."""

        # 1. Select steps
        selected_steps = self._select_steps(dataset, parsed, include_optional)

        # 2. Validate & incorporate custom steps
        if custom_steps:
            selected_steps = self._validate_custom_steps(
                dataset, selected_steps, custom_steps
            )

        # 3. Build nodes
        nodes = self._build_nodes(selected_steps, dataset)

        # 4. Build edges (only for transitions between selected nodes)
        selected_ids = {s.id for s in selected_steps}
        edges = self._build_edges(dataset, selected_ids)

        # 5. Prune unreachable nodes
        nodes, edges = self._prune_unreachable(
            nodes, edges, dataset.start_node
        )

        # 6. Ensure we still have start and end
        node_ids = {n.id for n in nodes}
        if dataset.start_node not in node_ids:
            raise WorkflowGenerationError(
                f"Start node '{dataset.start_node}' unreachable after pruning"
            )
        if dataset.end_node not in node_ids:
            # Try to find any end node
            end_nodes = [n for n in nodes if n.type == NodeType.END]
            if not end_nodes:
                raise WorkflowGenerationError(
                    "No reachable end node after pruning"
                )

        # 7. Build workflow
        workflow_id = self._generate_workflow_id(dataset.domain, parsed.cleaned_text)

        return GeneratedWorkflow(
            workflow_id=workflow_id,
            domain=dataset.domain,
            title=f"{dataset.display_name} Workflow",
            description=dataset.description,
            nodes=nodes,
            edges=edges,
            metadata={
                "dataset_version": dataset.version,
                "compliance": dataset.metadata.compliance,
                "criticality": dataset.metadata.criticality,
                "include_optional": include_optional,
            },
        )

    # ------------------------------------------------------------------
    # Step selection
    # ------------------------------------------------------------------

    def _select_steps(
        self,
        dataset: DomainDataset,
        parsed: ParsedInstruction,
        include_optional: bool,
    ) -> list[DomainStep]:
        """Select steps based on required flag and intent."""
        selected: list[DomainStep] = []
        intent = parsed.intent_flags

        for step in dataset.steps:
            if step.required:
                selected.append(step)
                continue

            if not include_optional:
                continue

            # Intent-based inclusion
            if step.retry_config and not intent.get("include_retry", True):
                continue

            if intent.get("minimal", False):
                # Skip non-required in minimal mode
                continue

            selected.append(step)

        # Ensure path connectivity: add optional steps that are the only
        # bridge between two required steps in the transition graph.
        selected = self._ensure_path_connectivity(dataset, selected)

        return selected

    # ------------------------------------------------------------------
    # Path connectivity — bridge optional steps
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_path_connectivity(
        dataset: DomainDataset,
        selected: list[DomainStep],
    ) -> list[DomainStep]:
        """
        Ensure selected steps form a connected path from start to end.

        If only required steps are selected, there may be gaps where
        transitions only go through optional steps. This method finds
        the shortest bridge through optional steps and adds them.
        """
        selected_ids = {s.id for s in selected}
        step_map = dataset.step_map

        # Build full adjacency from dataset transitions
        full_adj: dict[str, list[tuple[str, str | None]]] = {}
        for t in dataset.transitions:
            full_adj.setdefault(t.from_step, []).append((t.to_step, t.condition))

        # BFS from start_node through ALL dataset steps to find which
        # intermediate (non-selected) steps are needed to bridge gaps
        start = dataset.start_node
        end_nodes = {dataset.end_node}
        if dataset.error_terminal:
            end_nodes.add(dataset.error_terminal)

        # For each selected node, check if it can reach another selected node
        # directly via transitions. If not, find bridge steps.
        added: set[str] = set()
        queue: deque[tuple[str, list[str]]] = deque()

        for sid in selected_ids:
            # Check outgoing transitions
            outgoing = full_adj.get(sid, [])
            has_direct_selected_target = any(
                t[0] in selected_ids for t in outgoing
            )
            if has_direct_selected_target:
                continue

            # BFS to find shortest path to any selected node
            visited: set[str] = set()
            queue.clear()
            queue.append((sid, []))

            while queue:
                current, path = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                for next_id, _ in full_adj.get(current, []):
                    if next_id == sid:
                        continue
                    new_path = path + [next_id]
                    if next_id in selected_ids:
                        # Found bridge — add intermediate steps
                        for bridge_id in new_path[:-1]:
                            if bridge_id not in selected_ids and bridge_id not in added:
                                bridge_step = step_map.get(bridge_id)
                                if bridge_step:
                                    selected.append(bridge_step)
                                    added.add(bridge_id)
                        break
                    if next_id not in visited:
                        queue.append((next_id, new_path))

        return selected

    # ------------------------------------------------------------------
    # Custom step validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_custom_steps(
        dataset: DomainDataset,
        selected: list[DomainStep],
        custom_steps: Sequence[str],
    ) -> list[DomainStep]:
        """
        Validate custom steps against the domain.
        Only steps that exist in the dataset are allowed.
        """
        selected_ids = {s.id for s in selected}
        step_map = dataset.step_map

        for custom_id in custom_steps:
            normalised = custom_id.strip().lower().replace(" ", "_")
            if normalised in selected_ids:
                continue  # already selected
            if normalised not in step_map:
                logger.warning(
                    "Custom step '%s' rejected — not in domain '%s'",
                    custom_id,
                    dataset.domain,
                )
                continue  # reject unsupported
            selected.append(step_map[normalised])
            selected_ids.add(normalised)

        return selected

    # ------------------------------------------------------------------
    # Node / edge building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nodes(steps: list[DomainStep], dataset: DomainDataset) -> list[WorkflowNode]:
        """Convert domain steps to workflow nodes."""
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

    @staticmethod
    def _build_edges(
        dataset: DomainDataset,
        selected_ids: set[str],
    ) -> list[WorkflowEdge]:
        """Build edges from dataset transitions, limited to selected nodes."""
        edges: list[WorkflowEdge] = []
        seen_edges: set[tuple[str, str, str | None]] = set()

        for t in dataset.transitions:
            if t.from_step not in selected_ids or t.to_step not in selected_ids:
                continue

            edge_key = (t.from_step, t.to_step, t.condition)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edge_id = f"e_{t.from_step}__{t.to_step}"
            if t.condition:
                edge_id += f"__{t.condition}"

            condition = None
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
                )
            )

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
        """Remove nodes not reachable from the start node via BFS."""
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
            for neighbour in adj.get(nid, []):
                if neighbour not in reachable:
                    queue.append(neighbour)

        pruned_count = len(nodes) - len(reachable)
        if pruned_count > 0:
            logger.info("Pruned %d unreachable nodes", pruned_count)

        filtered_nodes = [n for n in nodes if n.id in reachable]
        filtered_edges = [
            e for e in edges if e.source in reachable and e.target in reachable
        ]
        return filtered_nodes, filtered_edges

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_workflow_id(domain: str, text: str) -> str:
        """Deterministic workflow ID from domain + instruction text."""
        raw = f"{domain}::{text}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"wf_{domain}_{digest}"
