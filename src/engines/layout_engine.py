"""
Deterministic Layout Engine — BFS-based, single-pass coordinate assignment.

Supports two modes:
  - Workflow layout: standard BFS depth/branch layout
  - Flowchart layout: decision-aware branch positioning with:
    * YES branch fixed at +Y (right / downward)
    * NO branch fixed at -Y (left / upward)
    * Fixed branch angle (±45 degrees from parent)
    * Retry loops rendered as backward edges (style metadata only)
    * 3-second timeout guard

Shared algorithm:
  1. BFS from start node to compute depth for every node
  2. Assign X coordinate = depth × horizontal_spacing
  3. Assign Y coordinate = branch_index × vertical_spacing (per depth level)
  4. One pass — no recursive repositioning
  5. Runs once only
  6. O(V + E) time complexity
  7. Deterministic: same graph → same layout
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque

from src.config import settings
from src.models.workflow import (
    Coordinate,
    EdgeStyle,
    GeneratedWorkflow,
    LayoutInfo,
    NodeType,
)

logger = logging.getLogger(__name__)

# Timeout guard (seconds)
_LAYOUT_TIMEOUT: float = 3.0


class LayoutEngine:
    """Single-pass, BFS-based deterministic layout engine."""

    def __init__(
        self,
        h_spacing: float | None = None,
        v_spacing: float | None = None,
        padding: float | None = None,
    ) -> None:
        self._h_spacing = h_spacing or settings.node_horizontal_spacing
        self._v_spacing = v_spacing or settings.node_vertical_spacing
        self._padding = padding or settings.layout_padding

    def compute_layout(self, workflow: GeneratedWorkflow) -> GeneratedWorkflow:
        """
        Compute and assign layout positions to all nodes in-place.

        Returns the same workflow object with updated layout info.
        """
        if not workflow.nodes:
            return workflow

        # 1. Build adjacency list
        adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for edge in workflow.edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        # Sort adjacency lists for determinism
        for nid in adj:
            adj[nid].sort()

        # 2. BFS to compute depth
        start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
        if not start_nodes:
            # Fallback: use first node
            start_nodes = [workflow.nodes[0]]

        depths = self._bfs_depth(adj, [s.id for s in start_nodes], workflow)

        # 3. Group nodes by depth
        depth_groups: dict[int, list[str]] = {}
        for nid, depth in depths.items():
            depth_groups.setdefault(depth, []).append(nid)

        # Sort within each depth level for determinism
        for depth in depth_groups:
            depth_groups[depth].sort()

        # 4. Assign coordinates — single pass
        node_map = workflow.node_map
        for depth, node_ids in sorted(depth_groups.items()):
            for branch_idx, nid in enumerate(node_ids):
                if nid not in node_map:
                    continue

                x = self._padding + depth * self._h_spacing
                y = self._padding + branch_idx * self._v_spacing

                node_map[nid].layout = LayoutInfo(
                    depth=depth,
                    branch_index=branch_idx,
                    position=Coordinate(x=round(x, 2), y=round(y, 2)),
                )

        # Handle any nodes not reached by BFS (isolated nodes)
        max_depth = max(depths.values()) if depths else 0
        unvisited = [n for n in workflow.nodes if n.id not in depths]
        for i, node in enumerate(unvisited):
            d = max_depth + 1
            x = self._padding + d * self._h_spacing
            y = self._padding + i * self._v_spacing
            node.layout = LayoutInfo(
                depth=d,
                branch_index=i,
                position=Coordinate(x=round(x, 2), y=round(y, 2)),
            )

        logger.info(
            "Layout computed: %d nodes, max depth %d",
            len(workflow.nodes),
            max_depth,
        )
        return workflow

    # ------------------------------------------------------------------
    # BFS depth calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _bfs_depth(
        adj: dict[str, list[str]],
        start_ids: list[str],
        workflow: GeneratedWorkflow,
    ) -> dict[str, int]:
        """
        Compute depth of each node via BFS from start nodes.

        Returns dict of node_id → depth (0-indexed).
        """
        depths: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()

        for sid in start_ids:
            if sid not in depths:
                queue.append((sid, 0))
                depths[sid] = 0

        while queue:
            nid, depth = queue.popleft()
            for neighbour in adj.get(nid, []):
                if neighbour not in depths:
                    depths[neighbour] = depth + 1
                    queue.append((neighbour, depth + 1))

        return depths

    # ==================================================================
    # Flowchart-specific layout
    # ==================================================================

    def compute_flowchart_layout(
        self,
        workflow: GeneratedWorkflow,
    ) -> GeneratedWorkflow:
        """
        Flowchart-aware layout with decision-branch positioning.

        Rules:
          - X = depth × h_spacing  (same as workflow)
          - For decision nodes:  first branch at +Y, second at −Y (±45°)
          - Retry loops are annotated with ``EdgeStyle.RETRY_LOOP`` but
            not repositioned — they are backward edges.
          - Single pass.  Timeout guard at ``_LAYOUT_TIMEOUT`` seconds.
        """
        if not workflow.nodes:
            return workflow

        t_start = time.monotonic()

        # 1. Build adjacency
        adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for edge in workflow.edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        # Sort for determinism
        for nid in adj:
            adj[nid].sort()

        # 2. BFS depth
        start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
        if not start_nodes:
            start_nodes = [workflow.nodes[0]]

        depths = self._bfs_depth(adj, [s.id for s in start_nodes], workflow)

        # 3. Build decision-aware branch index assignment
        #    We do a BFS again, but this time we track Y-offset
        #    Decision branches get ±offset from the parent's Y.
        node_map = workflow.node_map
        positions: dict[str, tuple[float, float]] = {}
        branch_indices: dict[str, int] = {}

        # Mapping edge → condition-ordering for decision node children
        decision_edges: dict[str, list[tuple[str, str | None]]] = {}
        for edge in workflow.edges:
            if edge.style == EdgeStyle.RETRY_LOOP:
                continue
            cond_key = edge.condition.branch_key if edge.condition else None
            decision_edges.setdefault(edge.source, []).append(
                (edge.target, cond_key)
            )

        # Sort decision children deterministically
        for nid in decision_edges:
            decision_edges[nid].sort(key=lambda x: (x[1] or "", x[0]))

        # BFS for position assignment
        y_at_depth: dict[int, float] = {}  # next available Y at each depth
        queue2: deque[tuple[str, float]] = deque()
        visited: set[str] = set()

        for sn in start_nodes:
            d = depths.get(sn.id, 0)
            x = self._padding + d * self._h_spacing
            y = self._padding
            positions[sn.id] = (x, y)
            branch_indices[sn.id] = 0
            y_at_depth[d] = y + self._v_spacing
            visited.add(sn.id)
            queue2.append((sn.id, y))

        while queue2:
            # Timeout guard
            if time.monotonic() - t_start > _LAYOUT_TIMEOUT:
                logger.warning("Flowchart layout timeout — using partial layout")
                break

            nid, parent_y = queue2.popleft()
            parent_node = node_map.get(nid)
            children = decision_edges.get(nid, [])

            if parent_node and parent_node.type == NodeType.DECISION and len(children) >= 2:
                # Decision branches: spread ±Y from parent centre
                branch_offset = self._v_spacing * 0.8
                half = (len(children) - 1) / 2.0
                for bi, (child_id, _cond) in enumerate(children):
                    if child_id in visited:
                        continue
                    d = depths.get(child_id, 0)
                    x = self._padding + d * self._h_spacing
                    # ±45° spread: first branch below (+Y), subsequent branches above
                    y = parent_y + (bi - half) * branch_offset
                    positions[child_id] = (x, y)
                    branch_indices[child_id] = bi
                    visited.add(child_id)
                    queue2.append((child_id, y))
            else:
                for child_id, _ in children:
                    if child_id in visited:
                        continue
                    d = depths.get(child_id, 0)
                    x = self._padding + d * self._h_spacing
                    y = y_at_depth.get(d, self._padding)
                    y_at_depth[d] = y + self._v_spacing
                    positions[child_id] = (x, y)
                    branch_indices[child_id] = 0
                    visited.add(child_id)
                    queue2.append((child_id, y))

        # 4. Assign layout info to nodes
        for node in workflow.nodes:
            pos = positions.get(node.id)
            if pos:
                node.layout = LayoutInfo(
                    depth=depths.get(node.id, 0),
                    branch_index=branch_indices.get(node.id, 0),
                    position=Coordinate(x=round(pos[0], 2), y=round(pos[1], 2)),
                )
            else:
                # Unvisited fallback
                max_depth = max(depths.values()) if depths else 0
                d = max_depth + 1
                x = self._padding + d * self._h_spacing
                y = self._padding
                node.layout = LayoutInfo(
                    depth=d,
                    branch_index=0,
                    position=Coordinate(x=round(x, 2), y=round(y, 2)),
                )

        elapsed = time.monotonic() - t_start
        logger.info(
            "Flowchart layout computed: %d nodes, %.1f ms",
            len(workflow.nodes),
            elapsed * 1000,
        )
        return workflow
