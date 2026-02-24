"""Workflow output models — nodes, edges, and the complete generated workflow."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    """Allowed node types in a rendered workflow."""

    START = "start"
    END = "end"
    PROCESS = "process"
    DECISION = "decision"


class Coordinate(BaseModel):
    """2-D coordinate for layout."""

    x: float = 0.0
    y: float = 0.0


class LayoutInfo(BaseModel):
    """Deterministic layout metadata for a node."""

    depth: int = Field(default=0, ge=0)
    branch_index: int = Field(default=0, ge=0)
    position: Coordinate = Field(default_factory=Coordinate)


class WorkflowNode(BaseModel):
    """A single node in the generated workflow."""

    id: str
    label: str
    type: NodeType
    description: str = ""
    domain_step_id: str = ""
    branches: dict[str, str] | None = None
    layout: LayoutInfo = Field(default_factory=LayoutInfo)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EdgeCondition(BaseModel):
    """Condition on an edge (optional)."""

    label: str = ""
    branch_key: str | None = None


class EdgeStyle(StrEnum):
    """Visual style for edge rendering."""

    NORMAL = "normal"
    RETRY_LOOP = "retry_loop"


class WorkflowEdge(BaseModel):
    """A directed edge between two workflow nodes."""

    id: str
    source: str
    target: str
    condition: EdgeCondition | None = None
    style: EdgeStyle = EdgeStyle.NORMAL
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedWorkflow(BaseModel):
    """Complete generated workflow — the final output of the pipeline."""

    workflow_id: str
    domain: str
    title: str
    description: str = ""
    is_flowchart: bool = False
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def node_map(self) -> dict[str, WorkflowNode]:
        return {n.id: n for n in self.nodes}

    @property
    def adjacency(self) -> dict[str, list[str]]:
        """Adjacency list: node-id → [target-ids]."""
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            adj.setdefault(e.source, []).append(e.target)
        return adj
