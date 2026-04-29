"""
LLM Post-Processor — dataset-constraint enforcement for LLM-generated workflows.

Applies minimal corrections to LLM output so it aligns with deterministic
dataset-driven generation:
  1. Node grounding (semantic match → canonical dataset step)
  2. Edge validation (keep only allowed transitions)
  3. Required-step injection
  4. Prune unreachable nodes
  5. Connectivity repair (single start, reachable end)

Principle: only fix what is broken; do not rebuild the entire graph.
"""

from __future__ import annotations

import logging
from collections import deque

from src.models.domain import DomainDataset, DomainStep
from src.models.workflow import GeneratedWorkflow, NodeType, WorkflowEdge, WorkflowNode

logger = logging.getLogger(__name__)


class LLMPostProcessor:
    """
    Post-processes an LLM-generated workflow to enforce dataset constraints.
    """

    def process(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """Apply minimal dataset constraints to an LLM-generated workflow."""
        # 1. Ground nodes to dataset
        workflow = self._ground_nodes(workflow, dataset)

        # 2. Validate & filter edges
        workflow = self._validate_edges(workflow, dataset)

        # 3. Inject any missing required steps
        workflow = self._inject_required_steps(workflow, dataset)

        # 4. Prune unreachable nodes
        workflow = self._prune_unreachable(workflow, dataset)

        # 5. Repair connectivity (single start, reachable end)
        workflow = self._repair_connectivity(workflow, dataset)

        return workflow

    # ------------------------------------------------------------------
    # 1. Node grounding
    # ------------------------------------------------------------------

    @staticmethod
    def _ground_nodes(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """Map LLM nodes to canonical dataset steps via semantic match."""
        step_map = dataset.step_map
        step_ids = set(step_map.keys())

        grounded: list[WorkflowNode] = []
        seen_ids: set[str] = set()

        for node in workflow.nodes:
            # Try exact ID match first
            match_id = node.domain_step_id or node.id
            canonical = step_map.get(match_id)

            # Try label-based fuzzy match if no exact ID match
            if canonical is None:
                canonical = LLMPostProcessor._find_best_step_match(
                    node.label or node.id, step_map
                )

            if canonical is None:
                logger.warning(
                    "LLM node '%s' ('%s') not found in dataset — removed.",
                    node.id,
                    node.label,
                )
                continue

            # Prevent duplicates
            if canonical.id in seen_ids:
                continue
            seen_ids.add(canonical.id)

            grounded.append(
                WorkflowNode(
                    id=canonical.id,
                    label=canonical.label,
                    type=NodeType(canonical.type),
                    description=canonical.description,
                    domain_step_id=canonical.id,
                    branches=canonical.branches,
                    metadata={**node.metadata, "grounded": True},
                )
            )

        workflow.nodes = grounded
        return workflow

    @staticmethod
    def _find_best_step_match(
        label: str,
        step_map: dict[str, DomainStep],
    ) -> DomainStep | None:
        """Find the dataset step whose label is closest to the given label."""
        needle = label.lower().strip().replace("_", " ")
        best: DomainStep | None = None
        best_score = 0.0

        for step in step_map.values():
            haystack = step.label.lower().strip().replace("_", " ")
            # Exact match
            if needle == haystack:
                return step

            # Containment
            score = 0.0
            if needle in haystack or haystack in needle:
                score = 0.7

            # Word overlap
            needle_words = set(needle.split())
            haystack_words = set(haystack.split())
            if needle_words and haystack_words:
                overlap = len(needle_words & haystack_words) / max(
                    len(needle_words), len(haystack_words)
                )
                score = max(score, overlap)

            if score > best_score:
                best_score = score
                best = step

        # Threshold: require at least 50% word overlap or containment
        return best if best_score >= 0.5 else None

    # ------------------------------------------------------------------
    # 2. Edge validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_edges(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """Keep only edges that match allowed transitions in the dataset."""
        valid_node_ids = {n.id for n in workflow.nodes}

        # Build set of allowed transitions
        allowed: set[tuple[str, str, str | None]] = set()
        for t in dataset.transitions:
            allowed.add((t.from_step, t.to_step, t.condition))

        valid_edges: list[WorkflowEdge] = []
        seen: set[tuple[str, str, str | None]] = set()

        for edge in workflow.edges:
            source = edge.source
            target = edge.target
            cond = edge.condition.branch_key if edge.condition else None

            # Both nodes must exist
            if source not in valid_node_ids or target not in valid_node_ids:
                continue

            # Transition must be allowed
            key = (source, target, cond)
            alt_key = (source, target, None)

            if key not in allowed and alt_key not in allowed:
                logger.warning(
                    "LLM edge %s -> %s (%s) not in dataset — removed.",
                    source,
                    target,
                    cond,
                )
                continue

            # Deduplicate
            if key in seen:
                continue
            seen.add(key)

            valid_edges.append(edge)

        workflow.edges = valid_edges
        return workflow

    # ------------------------------------------------------------------
    # 3. Required-step injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_required_steps(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """Add any required dataset steps that the LLM omitted."""
        existing_ids = {n.id for n in workflow.nodes}

        for step in dataset.steps:
            if step.required and step.id not in existing_ids:
                logger.info(
                    "Injecting required step '%s' (%s) missing from LLM output.",
                    step.id,
                    step.label,
                )
                workflow.nodes.append(
                    WorkflowNode(
                        id=step.id,
                        label=step.label,
                        type=NodeType(step.type),
                        description=step.description,
                        domain_step_id=step.id,
                        branches=step.branches,
                        metadata={"injected": True},
                    )
                )

        return workflow

    # ------------------------------------------------------------------
    # 4. Prune unreachable nodes
    # ------------------------------------------------------------------

    @staticmethod
    def _prune_unreachable(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """Remove nodes not reachable from the start node via BFS."""
        if not workflow.nodes:
            return workflow

        adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for edge in workflow.edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        reachable: set[str] = set()
        queue: deque[str] = deque()

        # Start BFS from dataset start node if present, else from any start node
        start_candidates = [n.id for n in workflow.nodes if n.type == NodeType.START]
        start = dataset.start_node if dataset.start_node in adj else None
        if start is None and start_candidates:
            start = start_candidates[0]

        if start is None:
            # No start node — can't prune meaningfully
            return workflow

        queue.append(start)
        while queue:
            nid = queue.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for neighbour in adj.get(nid, []):
                if neighbour not in reachable:
                    queue.append(neighbour)

        pruned = [n for n in workflow.nodes if n.id in reachable]
        pruned_edges = [
            e for e in workflow.edges
            if e.source in reachable and e.target in reachable
        ]

        if len(pruned) < len(workflow.nodes):
            logger.info(
                "Pruned %d unreachable LLM nodes",
                len(workflow.nodes) - len(pruned),
            )

        workflow.nodes = pruned
        workflow.edges = pruned_edges
        return workflow

    # ------------------------------------------------------------------
    # 5. Connectivity repair
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_connectivity(
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """Ensure single start node and at least one reachable end node."""
        node_ids = {n.id for n in workflow.nodes}

        # --- Single start node ---
        start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
        if len(start_nodes) > 1:
            # Keep only the one matching dataset.start_node, else first
            keep_id = (
                dataset.start_node
                if dataset.start_node in {n.id for n in start_nodes}
                else start_nodes[0].id
            )
            workflow.nodes = [
                n for n in workflow.nodes
                if n.type != NodeType.START or n.id == keep_id
            ]
            # Remove edges from removed start nodes
            workflow.edges = [
                e for e in workflow.edges
                if e.source == keep_id or e.source in node_ids
            ]
            node_ids = {n.id for n in workflow.nodes}

        if not any(n.type == NodeType.START for n in workflow.nodes):
            # Inject dataset start node if missing
            if dataset.start_node in dataset.step_map:
                step = dataset.step_map[dataset.start_node]
                workflow.nodes.append(
                    WorkflowNode(
                        id=step.id,
                        label=step.label,
                        type=NodeType(step.type),
                        description=step.description,
                        domain_step_id=step.id,
                        metadata={"injected": True},
                    )
                )
                node_ids.add(step.id)

        # --- Reachable end node ---
        end_nodes = [n for n in workflow.nodes if n.type == NodeType.END]
        if not end_nodes:
            # Inject dataset end node if missing
            if dataset.end_node in dataset.step_map:
                step = dataset.step_map[dataset.end_node]
                workflow.nodes.append(
                    WorkflowNode(
                        id=step.id,
                        label=step.label,
                        type=NodeType(step.type),
                        description=step.description,
                        domain_step_id=step.id,
                        metadata={"injected": True},
                    )
                )
                node_ids.add(step.id)

        # Re-check reachability after repairs
        workflow = LLMPostProcessor._prune_unreachable(workflow, dataset)
        return workflow

