"""Tests for the Flowchart Generator engine."""

from __future__ import annotations

import pytest

from src.engines.domain_engine import DomainDatasetEngine
from src.engines.flowchart_generator import (
    FlowchartGenerationError,
    FlowchartGenerator,
)
from src.engines.hallucination_mitigation import HallucinationMitigator
from src.engines.instruction_parser import InstructionParser
from src.engines.layout_engine import LayoutEngine
from src.engines.validation_engine import ValidationEngine
from src.models.validation import IssueCategory, IssueSeverity
from src.models.workflow import EdgeStyle, GeneratedWorkflow, NodeType


# ── helpers ──────────────────────────────────────────────────────────
def _gen_flowchart(
    flowchart_generator: FlowchartGenerator,
    parser: InstructionParser,
    dataset_engine: DomainDatasetEngine,
    domain: str,
    instruction: str = "generate flowchart",
    *,
    include_optional: bool = True,
) -> GeneratedWorkflow:
    parsed = parser.parse(instruction)
    ds = dataset_engine.get_strict(domain)
    return flowchart_generator.generate(
        ds, parsed, include_optional=include_optional
    )


# =====================================================================
# Flowchart Generation — core behaviour
# =====================================================================
class TestFlowchartGeneratorCore:
    """Core flowchart generation behaviour."""

    def test_generates_flowchart_online_payment(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        assert wf.is_flowchart is True
        assert wf.workflow_id.startswith("fc_online_payment_")
        assert wf.domain == "online_payment"
        assert len(wf.nodes) > 0
        assert len(wf.edges) > 0

    def test_has_start_node(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        start_nodes = [n for n in wf.nodes if n.type == NodeType.START]
        assert len(start_nodes) == 1

    def test_has_end_node(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        end_nodes = [n for n in wf.nodes if n.type == NodeType.END]
        assert len(end_nodes) >= 1

    def test_has_decision_nodes(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        decisions = [n for n in wf.nodes if n.type == NodeType.DECISION]
        assert len(decisions) >= 1

    def test_decision_has_min_two_outgoing_edges(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        decisions = [n for n in wf.nodes if n.type == NodeType.DECISION]
        for dn in decisions:
            outgoing = [e for e in wf.edges if e.source == dn.id]
            assert len(outgoing) >= 2, (
                f"Decision '{dn.id}' has {len(outgoing)} outgoing edges"
            )

    def test_deterministic_output(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        """Same input produces identical output."""
        wf1 = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        wf2 = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )

        assert wf1.workflow_id == wf2.workflow_id
        assert len(wf1.nodes) == len(wf2.nodes)
        assert len(wf1.edges) == len(wf2.edges)
        assert sorted(n.id for n in wf1.nodes) == sorted(n.id for n in wf2.nodes)
        assert sorted(e.id for e in wf1.edges) == sorted(e.id for e in wf2.edges)

    def test_is_flowchart_flag(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        assert wf.is_flowchart is True

    def test_metadata_includes_decision_count(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        assert "decision_count" in wf.metadata
        assert wf.metadata["decision_count"] >= 1


# =====================================================================
# Retry edges
# =====================================================================
class TestFlowchartRetryEdges:
    """Retry loop backward edges."""

    def test_retry_edge_present(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        """online_payment has retry_payment→bank_authorization retry."""
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        retry_edges = [e for e in wf.edges if e.style == EdgeStyle.RETRY_LOOP]
        assert len(retry_edges) >= 1

    def test_retry_edge_source_target(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        retry_edges = [e for e in wf.edges if e.style == EdgeStyle.RETRY_LOOP]
        # Should have retry_payment → bank_authorization
        src_tgt = {(e.source, e.target) for e in retry_edges}
        assert ("retry_payment", "bank_authorization") in src_tgt

    def test_retry_edge_has_condition_label(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        retry_edges = [e for e in wf.edges if e.style == EdgeStyle.RETRY_LOOP]
        for re_ in retry_edges:
            assert re_.condition is not None
            assert "Retry" in re_.condition.label


# =====================================================================
# All domains
# =====================================================================
class TestFlowchartAllDomains:
    """Flowchart generation succeeds for every domain."""

    _DOMAINS = [
        "online_payment",
        "user_registration",
        "order_fulfillment",
        "incident_response",
        "data_pipeline",
        "ci_cd_deployment",
    ]

    @pytest.mark.parametrize("domain", _DOMAINS)
    def test_generates_for_domain(
        self,
        domain: str,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, domain
        )
        assert wf.is_flowchart is True
        assert wf.domain == domain
        assert len(wf.nodes) > 0
        assert len(wf.edges) > 0

    @pytest.mark.parametrize("domain", _DOMAINS)
    def test_has_decision_node_per_domain(
        self,
        domain: str,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, domain
        )
        decisions = [n for n in wf.nodes if n.type == NodeType.DECISION]
        assert len(decisions) >= 1, f"No decision nodes for {domain}"

    @pytest.mark.parametrize("domain", _DOMAINS)
    def test_retry_edge_per_domain(
        self,
        domain: str,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        """Each domain has at least one retry constraint."""
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, domain
        )
        retry_edges = [e for e in wf.edges if e.style == EdgeStyle.RETRY_LOOP]
        assert len(retry_edges) >= 1, f"No retry edges for {domain}"


# =====================================================================
# Strict grounding — no hallucinated content
# =====================================================================
class TestFlowchartGrounding:
    """All flowchart nodes/edges trace back to the dataset."""

    def test_all_nodes_from_dataset(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        ds = dataset_engine.get_strict("online_payment")
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        step_ids = {s.id for s in ds.steps}
        for node in wf.nodes:
            assert node.id in step_ids, (
                f"Node '{node.id}' not from dataset"
            )

    def test_all_edges_from_dataset_transitions(
        self,
        flowchart_generator: FlowchartGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        ds = dataset_engine.get_strict("online_payment")
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        allowed = {(t.from_step, t.to_step) for t in ds.transitions}
        retry_allowed = {
            (rc.node, rc.loop_back_to)
            for rc in ds.flowchart_retry_constraints
        }
        all_allowed = allowed | retry_allowed

        for edge in wf.edges:
            assert (edge.source, edge.target) in all_allowed, (
                f"Edge {edge.source}→{edge.target} not in dataset"
            )


# =====================================================================
# Flowchart hallucination mitigation (strict)
# =====================================================================
class TestFlowchartHallucinationMitigation:
    """mitigate_flowchart() validates decision rules and retry constraints."""

    def test_valid_flowchart_passes_mitigation(
        self,
        flowchart_generator: FlowchartGenerator,
        mitigator: HallucinationMitigator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        ds = dataset_engine.get_strict("online_payment")
        corrected, result = mitigator.mitigate_flowchart(wf, ds)
        assert result.nodes_validated > 0
        assert result.edges_validated > 0

    def test_mitigation_no_errors_on_clean_flowchart(
        self,
        flowchart_generator: FlowchartGenerator,
        mitigator: HallucinationMitigator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        ds = dataset_engine.get_strict("online_payment")
        _, result = mitigator.mitigate_flowchart(wf, ds)
        error_issues = [
            i for i in result.issues if i.severity == IssueSeverity.ERROR
        ]
        assert len(error_issues) == 0


# =====================================================================
# Flowchart validation engine
# =====================================================================
class TestFlowchartValidation:
    """validate_flowchart() runs standard + flowchart-specific passes."""

    def test_validate_flowchart_passes(
        self,
        flowchart_generator: FlowchartGenerator,
        validator: ValidationEngine,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        ds = dataset_engine.get_strict("online_payment")
        result = validator.validate_flowchart(wf, ds)
        assert result.is_valid
        assert len(result.checks_performed) > 0

    def test_validate_flowchart_checks_reachability(
        self,
        flowchart_generator: FlowchartGenerator,
        validator: ValidationEngine,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        ds = dataset_engine.get_strict("online_payment")
        result = validator.validate_flowchart(wf, ds)
        assert "flowchart_reachability" in result.checks_performed

    def test_validate_flowchart_all_domains(
        self,
        flowchart_generator: FlowchartGenerator,
        validator: ValidationEngine,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        domains = [
            "online_payment",
            "user_registration",
            "order_fulfillment",
            "incident_response",
            "data_pipeline",
            "ci_cd_deployment",
        ]
        for domain in domains:
            wf = _gen_flowchart(
                flowchart_generator, parser, dataset_engine, domain
            )
            ds = dataset_engine.get_strict(domain)
            result = validator.validate_flowchart(wf, ds)
            assert result.is_valid, (
                f"Flowchart validation failed for {domain}: "
                f"{[i.message for i in result.issues if i.severity == IssueSeverity.ERROR]}"
            )


# =====================================================================
# Flowchart layout
# =====================================================================
class TestFlowchartLayout:
    """compute_flowchart_layout() assigns positions deterministically."""

    def test_layout_assigns_positions(
        self,
        flowchart_generator: FlowchartGenerator,
        layout_engine: LayoutEngine,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        wf = layout_engine.compute_flowchart_layout(wf)
        for node in wf.nodes:
            assert node.layout.position.x >= 0
            assert node.layout.position.y >= 0

    def test_layout_deterministic(
        self,
        flowchart_generator: FlowchartGenerator,
        layout_engine: LayoutEngine,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        wf1 = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        wf2 = _gen_flowchart(
            flowchart_generator, parser, dataset_engine, "online_payment"
        )
        wf1 = layout_engine.compute_flowchart_layout(wf1)
        wf2 = layout_engine.compute_flowchart_layout(wf2)

        positions1 = {n.id: (n.layout.position.x, n.layout.position.y) for n in wf1.nodes}
        positions2 = {n.id: (n.layout.position.x, n.layout.position.y) for n in wf2.nodes}
        assert positions1 == positions2
