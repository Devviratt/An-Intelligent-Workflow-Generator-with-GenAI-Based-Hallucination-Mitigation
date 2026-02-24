"""Tests for the Layout Engine."""

from __future__ import annotations

from src.engines.domain_engine import DomainDatasetEngine
from src.engines.instruction_parser import InstructionParser
from src.engines.layout_engine import LayoutEngine
from src.engines.workflow_generator import WorkflowGenerator
from src.models.workflow import GeneratedWorkflow, NodeType


class TestLayoutEngine:
    """Tests for BFS-based deterministic layout."""

    def _make_workflow(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> GeneratedWorkflow:
        parsed = parser.parse("payment processing")
        ds = dataset_engine.get_strict("online_payment")
        return generator.generate(ds, parsed)

    def test_all_nodes_have_positions(
        self,
        layout_engine: LayoutEngine,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._make_workflow(generator, parser, dataset_engine)
        workflow = layout_engine.compute_layout(workflow)

        for node in workflow.nodes:
            assert node.layout.position.x >= 0
            assert node.layout.position.y >= 0

    def test_start_node_at_depth_zero(
        self,
        layout_engine: LayoutEngine,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._make_workflow(generator, parser, dataset_engine)
        workflow = layout_engine.compute_layout(workflow)

        start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
        for sn in start_nodes:
            assert sn.layout.depth == 0

    def test_deterministic_layout(
        self,
        layout_engine: LayoutEngine,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        """Same workflow → same layout."""
        workflow1 = self._make_workflow(generator, parser, dataset_engine)
        workflow2 = self._make_workflow(generator, parser, dataset_engine)

        workflow1 = layout_engine.compute_layout(workflow1)
        workflow2 = layout_engine.compute_layout(workflow2)

        for n1 in workflow1.nodes:
            n2 = next((n for n in workflow2.nodes if n.id == n1.id), None)
            assert n2 is not None
            assert n1.layout.position.x == n2.layout.position.x
            assert n1.layout.position.y == n2.layout.position.y
            assert n1.layout.depth == n2.layout.depth

    def test_depth_increases_with_distance(
        self,
        layout_engine: LayoutEngine,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._make_workflow(generator, parser, dataset_engine)
        workflow = layout_engine.compute_layout(workflow)

        node_map = workflow.node_map

        # End node should have greater depth than start node
        start = node_map.get("initiate_payment")
        end = node_map.get("transaction_complete")
        if start and end:
            assert end.layout.depth > start.layout.depth

    def test_no_overlapping_positions(
        self,
        layout_engine: LayoutEngine,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._make_workflow(generator, parser, dataset_engine)
        workflow = layout_engine.compute_layout(workflow)

        positions = set()
        for node in workflow.nodes:
            pos = (node.layout.position.x, node.layout.position.y)
            assert pos not in positions, f"Overlapping position: {pos}"
            positions.add(pos)
