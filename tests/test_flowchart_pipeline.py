"""Integration tests for the flowchart pipeline (mode=flowchart)."""

from __future__ import annotations

import pytest

from src.engines.llm_flowchart_generator import LLMFlowchartGenerationError
from src.models.request import GenerateRequest, GenerationMode
from src.models.workflow import EdgeStyle, NodeType
from src.pipeline import Pipeline


class TestFlowchartPipeline:
    """End-to-end tests for flowchart generation through the pipeline."""

    @pytest.mark.asyncio
    async def test_generate_flowchart_online_payment(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction="Create a payment processing flowchart",
            mode=GenerationMode.FLOWCHART,
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.is_flowchart is True
        assert response.workflow.domain == "online_payment"
        assert len(response.workflow.nodes) > 5
        assert len(response.workflow.edges) > 5

    @pytest.mark.asyncio
    async def test_generate_flowchart_with_domain_hint(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction="some flowchart",
            domain_hint="ci_cd_deployment",
            mode=GenerationMode.FLOWCHART,
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.domain == "ci_cd_deployment"
        assert response.workflow.is_flowchart is True

    @pytest.mark.asyncio
    async def test_flowchart_has_decision_nodes(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction="payment flowchart",
            domain_hint="online_payment",
            mode=GenerationMode.FLOWCHART,
        )
        response = await pipeline.generate(request)

        assert response.workflow is not None
        decisions = [
            n for n in response.workflow.nodes if n.type == NodeType.DECISION
        ]
        assert len(decisions) >= 1

    @pytest.mark.asyncio
    async def test_flowchart_has_retry_edges(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction="payment flowchart",
            domain_hint="online_payment",
            mode=GenerationMode.FLOWCHART,
        )
        response = await pipeline.generate(request)

        assert response.workflow is not None
        retry = [
            e
            for e in response.workflow.edges
            if e.style == EdgeStyle.RETRY_LOOP
        ]
        assert len(retry) >= 1

    @pytest.mark.asyncio
    async def test_flowchart_has_layout(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="payment flowchart",
            domain_hint="online_payment",
            mode=GenerationMode.FLOWCHART,
        )
        response = await pipeline.generate(request)

        assert response.workflow is not None
        for node in response.workflow.nodes:
            assert node.layout.position.x >= 0
            assert node.layout.position.y >= 0

    @pytest.mark.asyncio
    async def test_flowchart_metrics_populated(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction="payment flowchart",
            domain_hint="online_payment",
            mode=GenerationMode.FLOWCHART,
        )
        response = await pipeline.generate(request)

        assert response.metrics.parse_time_ms >= 0
        assert response.metrics.generation_time_ms >= 0
        assert response.metrics.mitigation_time_ms >= 0
        assert response.metrics.validation_time_ms >= 0
        assert response.metrics.layout_time_ms >= 0
        assert response.metrics.total_time_ms > 0
        assert response.metrics.nodes_generated > 0
        assert response.metrics.edges_generated > 0

    @pytest.mark.asyncio
    async def test_flowchart_validation_included(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction="payment flowchart",
            domain_hint="online_payment",
            mode=GenerationMode.FLOWCHART,
        )
        response = await pipeline.generate(request)

        assert response.validation is not None
        assert len(response.validation.checks_performed) > 0

    @pytest.mark.asyncio
    async def test_flowchart_all_domains(self, pipeline: Pipeline) -> None:
        domains = [
            ("online_payment", "payment card transaction"),
            ("user_registration", "user signup registration"),
            ("order_fulfillment", "shipping order delivery"),
            ("incident_response", "incident alert triage"),
            ("data_pipeline", "data ETL pipeline"),
            ("ci_cd_deployment", "CI/CD deploy release"),
            ("loan_approval", "loan application underwriting approval"),
            ("insurance_claim_processing", "insurance claim settlement review"),
        ]
        for domain, instruction in domains:
            request = GenerateRequest(
                instruction=instruction,
                domain_hint=domain,
                mode=GenerationMode.FLOWCHART,
            )
            response = await pipeline.generate(request)
            assert response.success, (
                f"Flowchart failed for {domain}: {response.errors}"
            )
            assert response.workflow is not None
            assert response.workflow.is_flowchart is True

    @pytest.mark.asyncio
    async def test_flowchart_invalid_domain(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="something",
            domain_hint="nonexistent",
            mode=GenerationMode.FLOWCHART,
        )
        response = await pipeline.generate(request)
        assert not response.success
        assert len(response.errors) > 0

    @pytest.mark.asyncio
    async def test_workflow_mode_still_works(self, pipeline: Pipeline) -> None:
        """Ensure default mode=workflow is unaffected."""
        request = GenerateRequest(
            instruction="Create an online payment processing workflow"
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.is_flowchart is False
        assert response.workflow.domain == "online_payment"

    @pytest.mark.asyncio
    async def test_explicit_workflow_mode(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="payment flow",
            domain_hint="online_payment",
            mode=GenerationMode.WORKFLOW,
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.is_flowchart is False

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_deterministic_flowchart(
        self, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fail_llm(*args: object, **kwargs: object) -> None:
            raise LLMFlowchartGenerationError("simulated ollama memory failure")

        monkeypatch.setattr(pipeline._llm_flowchart_generator, "generate", fail_llm)

        request = GenerateRequest(
            instruction="payment flowchart",
            domain_hint="online_payment",
            mode=GenerationMode.FLOWCHART,
            prefer_llm_generation=True,
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.is_flowchart is True
        assert response.workflow.metadata["generation_engine"] == "deterministic_fallback"
        assert "simulated ollama memory failure" in response.workflow.metadata["fallback_reason"]
        assert response.validation is not None
        assert "skipped_llm" not in response.validation.checks_performed

    @pytest.mark.asyncio
    async def test_flowchart_mode_can_be_embedded_in_instruction(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction=(
                "domain: online_payment\n"
                "mode: flowchart\n"
                "- payment request received\n"
                "- validate request\n"
                "- payment authorized\n"
                "- notify customer\n"
            )
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.is_flowchart is True
        assert response.workflow.domain == "online_payment"
        assert response.workflow.metadata["input_format"] == "step_list"
