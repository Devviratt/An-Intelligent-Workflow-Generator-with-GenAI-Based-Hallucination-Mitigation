"""Tests for the Workflow Generator."""

from __future__ import annotations

import pytest

from src.engines.domain_engine import DomainDatasetEngine
from src.engines.instruction_parser import InstructionParser
from src.engines.workflow_generator import WorkflowGenerator
from src.models.workflow import NodeType


class TestWorkflowGenerator:
    """Tests for deterministic workflow generation."""

    def test_generates_workflow(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        parsed = parser.parse("Create a payment processing flow")
        ds = dataset_engine.get_strict("online_payment")
        workflow = generator.generate(ds, parsed)

        assert workflow.workflow_id.startswith("wf_online_payment_")
        assert workflow.domain == "online_payment"
        assert len(workflow.nodes) > 0
        assert len(workflow.edges) > 0

    def test_has_start_and_end_nodes(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        parsed = parser.parse("payment flow")
        ds = dataset_engine.get_strict("online_payment")
        workflow = generator.generate(ds, parsed)

        start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
        end_nodes = [n for n in workflow.nodes if n.type == NodeType.END]
        assert len(start_nodes) == 1
        assert len(end_nodes) >= 1

    def test_deterministic_output(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        """Same input must produce same output."""
        parsed = parser.parse("payment processing workflow")
        ds = dataset_engine.get_strict("online_payment")

        wf1 = generator.generate(ds, parsed)
        wf2 = generator.generate(ds, parsed)

        assert wf1.workflow_id == wf2.workflow_id
        assert len(wf1.nodes) == len(wf2.nodes)
        assert len(wf1.edges) == len(wf2.edges)

        ids1 = sorted(n.id for n in wf1.nodes)
        ids2 = sorted(n.id for n in wf2.nodes)
        assert ids1 == ids2

    def test_required_steps_included(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        parsed = parser.parse("payment flow")
        ds = dataset_engine.get_strict("online_payment")
        workflow = generator.generate(ds, parsed)

        node_ids = {n.id for n in workflow.nodes}
        for req in ds.validation_rules.required_steps:
            assert req in node_ids, f"Required step '{req}' missing"

    def test_minimal_mode(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        parsed_full = parser.parse("complete payment processing")
        parsed_minimal = parser.parse("simple minimal payment flow")
        ds = dataset_engine.get_strict("online_payment")

        wf_full = generator.generate(ds, parsed_full)
        wf_minimal = generator.generate(ds, parsed_minimal, include_optional=False)

        assert len(wf_minimal.nodes) <= len(wf_full.nodes)

    def test_custom_step_valid(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        parsed = parser.parse("payment flow")
        ds = dataset_engine.get_strict("online_payment")
        workflow = generator.generate(ds, parsed, custom_steps=["retry_payment"])

        node_ids = {n.id for n in workflow.nodes}
        assert "retry_payment" in node_ids

    def test_custom_step_invalid_rejected(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        parsed = parser.parse("payment flow")
        ds = dataset_engine.get_strict("online_payment")
        workflow = generator.generate(ds, parsed, custom_steps=["fly_to_moon"])

        node_ids = {n.id for n in workflow.nodes}
        assert "fly_to_moon" not in node_ids

    def test_all_domains_generate_successfully(
        self,
        generator: WorkflowGenerator,
        parser: InstructionParser,
        dataset_engine: DomainDatasetEngine,
    ) -> None:
        instructions = {
            "online_payment": "process a credit card payment",
            "user_registration": "register a new user account",
            "order_fulfillment": "fulfill and ship an order",
            "incident_response": "respond to a critical incident",
            "data_pipeline": "run a data ETL pipeline",
            "ci_cd_deployment": "deploy code to production",
        }
        for domain, instruction in instructions.items():
            parsed = parser.parse(instruction)
            ds = dataset_engine.get_strict(domain)
            workflow = generator.generate(ds, parsed)
            assert len(workflow.nodes) > 0, f"No nodes for {domain}"
            assert len(workflow.edges) > 0, f"No edges for {domain}"
