"""Tests for the Hallucination Mitigation Layer."""

from __future__ import annotations

from src.engines.domain_engine import DomainDatasetEngine
from src.engines.hallucination_mitigation import HallucinationMitigator
from src.engines.instruction_parser import InstructionParser
from src.engines.workflow_generator import WorkflowGenerator
from src.models.validation import IssueCategory, IssueSeverity
from src.models.workflow import (
    EdgeCondition,
    GeneratedWorkflow,
    NodeType,
    WorkflowEdge,
    WorkflowNode,
)


class TestHallucinationMitigation:
    """Tests for grounding-based hallucination mitigation."""

    def _generate_workflow(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
        domain: str = "online_payment",
    ) -> GeneratedWorkflow:
        parsed = parser.parse("payment processing")
        ds = dataset_engine.get_strict(domain)
        return generator.generate(ds, parsed)

    def test_valid_workflow_passes(
        self,
        mitigator: HallucinationMitigator,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._generate_workflow(generator, parser, dataset_engine)
        ds = dataset_engine.get_strict("online_payment")
        corrected, result = mitigator.mitigate(workflow, ds)

        assert result.nodes_validated > 0
        assert result.edges_validated > 0

    def test_detects_hallucinated_node(
        self,
        mitigator: HallucinationMitigator,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._generate_workflow(generator, parser, dataset_engine)
        ds = dataset_engine.get_strict("online_payment")

        # Inject a hallucinated node
        workflow.nodes.append(
            WorkflowNode(
                id="hallucinated_step",
                label="Do Something Fake",
                type=NodeType.PROCESS,
                domain_step_id="hallucinated_step",
            )
        )

        corrected, result = mitigator.mitigate(workflow, ds)

        # Hallucinated node should be removed
        node_ids = {n.id for n in corrected.nodes}
        assert "hallucinated_step" not in node_ids

        # Issue should be reported
        grounding_issues = [
            i for i in result.issues if i.category == IssueCategory.GROUNDING
        ]
        assert len(grounding_issues) >= 1

    def test_detects_invalid_transition(
        self,
        mitigator: HallucinationMitigator,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._generate_workflow(generator, parser, dataset_engine)
        ds = dataset_engine.get_strict("online_payment")

        # Inject a transition that shouldn't exist
        workflow.edges.append(
            WorkflowEdge(
                id="e_bad",
                source="initiate_payment",
                target="transaction_complete",
                condition=EdgeCondition(label="Skip Everything", branch_key="skip"),
            )
        )

        corrected, result = mitigator.mitigate(workflow, ds)

        # Bad edge should be removed
        edge_ids = {e.id for e in corrected.edges}
        assert "e_bad" not in edge_ids

    def test_detects_duplicate_nodes(
        self,
        mitigator: HallucinationMitigator,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        workflow = self._generate_workflow(generator, parser, dataset_engine)
        ds = dataset_engine.get_strict("online_payment")

        # Duplicate a node
        first_node = workflow.nodes[0]
        dup = first_node.model_copy(update={"id": "duplicate_of_first"})
        workflow.nodes.append(dup)

        corrected, result = mitigator.mitigate(workflow, ds)

        dup_issues = [
            i for i in result.issues if i.category == IssueCategory.DUPLICATE
        ]
        assert len(dup_issues) >= 1

    def test_checks_decision_branches(
        self,
        mitigator: HallucinationMitigator,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        ds = dataset_engine.get_strict("online_payment")

        # Build a workflow with a decision node that has only 1 outgoing edge
        workflow = GeneratedWorkflow(
            workflow_id="wf_test",
            domain="online_payment",
            title="Test",
            nodes=[
                WorkflowNode(
                    id="initiate_payment",
                    label="Start",
                    type=NodeType.START,
                    domain_step_id="initiate_payment",
                ),
                WorkflowNode(
                    id="fraud_check",
                    label="Fraud Check",
                    type=NodeType.DECISION,
                    domain_step_id="fraud_check",
                    branches={"passed": "ok", "blocked": "no"},
                ),
                WorkflowNode(
                    id="transaction_complete",
                    label="Done",
                    type=NodeType.END,
                    domain_step_id="transaction_complete",
                ),
            ],
            edges=[
                WorkflowEdge(id="e1", source="initiate_payment", target="fraud_check"),
                WorkflowEdge(
                    id="e2",
                    source="fraud_check",
                    target="transaction_complete",
                    condition=EdgeCondition(label="Passed", branch_key="passed"),
                ),
            ],
        )

        _, result = mitigator.mitigate(workflow, ds)

        branch_issues = [
            i for i in result.issues
            if i.category == IssueCategory.STRUCTURE
            and "decision" in i.message.lower()
        ]
        assert len(branch_issues) >= 1
