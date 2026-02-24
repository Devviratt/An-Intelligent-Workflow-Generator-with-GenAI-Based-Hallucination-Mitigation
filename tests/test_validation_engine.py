"""Tests for the Validation Engine."""

from __future__ import annotations

from src.engines.domain_engine import DomainDatasetEngine
from src.engines.instruction_parser import InstructionParser
from src.engines.validation_engine import ValidationEngine
from src.engines.workflow_generator import WorkflowGenerator
from src.models.validation import IssueCategory, IssueSeverity
from src.models.workflow import (
    GeneratedWorkflow,
    NodeType,
    WorkflowEdge,
    WorkflowNode,
)


class TestValidationEngine:
    """Tests for multi-pass workflow validation."""

    def _make_workflow(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> GeneratedWorkflow:
        parsed = parser.parse("payment processing")
        ds = dataset_engine.get_strict("online_payment")
        return generator.generate(ds, parsed)

    def test_valid_workflow_passes(
        self,
        validator: ValidationEngine,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._make_workflow(generator, parser, dataset_engine)
        ds = dataset_engine.get_strict("online_payment")
        result = validator.validate(workflow, ds)

        assert result.nodes_validated > 0
        assert result.edges_validated > 0
        assert len(result.checks_performed) >= 4

    def test_detects_empty_workflow(self, validator: ValidationEngine) -> None:
        workflow = GeneratedWorkflow(
            workflow_id="empty",
            domain="test",
            title="Empty",
            nodes=[],
            edges=[],
        )
        result = validator.validate(workflow)
        assert not result.is_valid

        schema_errors = [
            i for i in result.issues
            if i.category == IssueCategory.SCHEMA
            and i.severity == IssueSeverity.ERROR
        ]
        assert len(schema_errors) >= 1

    def test_detects_duplicate_node_ids(self, validator: ValidationEngine) -> None:
        workflow = GeneratedWorkflow(
            workflow_id="dup",
            domain="test",
            title="Dup",
            nodes=[
                WorkflowNode(id="a", label="A", type=NodeType.START, domain_step_id="a"),
                WorkflowNode(id="a", label="A2", type=NodeType.PROCESS, domain_step_id="a"),
                WorkflowNode(id="b", label="B", type=NodeType.END, domain_step_id="b"),
            ],
            edges=[
                WorkflowEdge(id="e1", source="a", target="b"),
            ],
        )
        result = validator.validate(workflow)
        assert not result.is_valid

    def test_detects_self_loop(self, validator: ValidationEngine) -> None:
        workflow = GeneratedWorkflow(
            workflow_id="loop",
            domain="test",
            title="Self Loop",
            nodes=[
                WorkflowNode(id="a", label="A", type=NodeType.START, domain_step_id="a"),
                WorkflowNode(id="b", label="B", type=NodeType.END, domain_step_id="b"),
            ],
            edges=[
                WorkflowEdge(id="e1", source="a", target="a"),
                WorkflowEdge(id="e2", source="a", target="b"),
            ],
        )
        result = validator.validate(workflow)
        assert not result.is_valid

        logical_errors = [
            i for i in result.issues if i.category == IssueCategory.LOGICAL
        ]
        assert any("self-loop" in i.message.lower() for i in logical_errors)

    def test_detects_dangling_edge(self, validator: ValidationEngine) -> None:
        workflow = GeneratedWorkflow(
            workflow_id="dangling",
            domain="test",
            title="Dangling",
            nodes=[
                WorkflowNode(id="a", label="A", type=NodeType.START, domain_step_id="a"),
            ],
            edges=[
                WorkflowEdge(id="e1", source="a", target="nonexistent"),
            ],
        )
        result = validator.validate(workflow)
        assert not result.is_valid

    def test_depth_verification(
        self,
        validator: ValidationEngine,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._make_workflow(generator, parser, dataset_engine)
        ds = dataset_engine.get_strict("online_payment")
        result = validator.validate(workflow, ds)

        assert "depth_verification" in result.checks_performed

    def test_all_passes_run(
        self,
        validator: ValidationEngine,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._make_workflow(generator, parser, dataset_engine)
        ds = dataset_engine.get_strict("online_payment")
        result = validator.validate(workflow, ds)

        expected_passes = [
            "schema_validation",
            "logical_validation",
            "dependency_validation",
            "cycle_detection",
            "depth_verification",
        ]
        for p in expected_passes:
            assert p in result.checks_performed, f"Pass '{p}' not executed"
