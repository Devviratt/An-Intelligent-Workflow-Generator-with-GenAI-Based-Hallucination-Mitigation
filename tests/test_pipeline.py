"""Tests for the full async pipeline."""

from __future__ import annotations

import pytest

from src.engines.llm_workflow_generator import LLMWorkflowGenerationError
from src.models.request import GenerateRequest
from src.pipeline import Pipeline


class TestPipeline:
    """Integration tests for the complete pipeline."""

    @pytest.mark.asyncio
    async def test_generate_payment_workflow(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="Create an online payment processing workflow"
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.domain == "online_payment"
        assert len(response.workflow.nodes) > 5
        assert len(response.workflow.edges) > 5
        assert response.metrics.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_generate_registration_workflow(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="Build a user registration signup flow"
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.domain == "user_registration"
        assert response.workflow.metadata["generation_engine"] == "deterministic"

    @pytest.mark.asyncio
    async def test_generate_with_domain_hint(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="do something",
            domain_hint="incident_response",
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.domain == "incident_response"

    @pytest.mark.asyncio
    async def test_generate_invalid_domain_hint(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="do something",
            domain_hint="nonexistent_domain",
        )
        response = await pipeline.generate(request)

        assert not response.success
        assert len(response.errors) > 0

    @pytest.mark.asyncio
    async def test_workflow_has_layout(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="payment processing workflow"
        )
        response = await pipeline.generate(request)

        assert response.workflow is not None
        for node in response.workflow.nodes:
            assert node.layout.position.x >= 0
            assert node.layout.position.y >= 0

    @pytest.mark.asyncio
    async def test_metrics_populated(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="order fulfillment shipping workflow"
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
    async def test_all_domains_generate(self, pipeline: Pipeline) -> None:
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
            )
            response = await pipeline.generate(request)
            assert response.success, f"Failed for {domain}: {response.errors}"
            assert response.workflow is not None

    @pytest.mark.asyncio
    async def test_validation_result_included(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="payment processing workflow"
        )
        response = await pipeline.generate(request)

        assert response.validation is not None
        assert len(response.validation.checks_performed) > 0

    @pytest.mark.asyncio
    async def test_custom_steps_handled(self, pipeline: Pipeline) -> None:
        request = GenerateRequest(
            instruction="payment flow",
            domain_hint="online_payment",
            custom_steps=["retry_payment", "fly_to_moon"],
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        node_ids = {n.id for n in response.workflow.nodes}
        assert "retry_payment" in node_ids
        assert "fly_to_moon" not in node_ids  # rejected

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_deterministic_workflow(
        self, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fail_llm(*args: object, **kwargs: object) -> None:
            raise LLMWorkflowGenerationError("simulated ollama memory failure")

        monkeypatch.setattr(pipeline._llm_generator, "generate", fail_llm)

        request = GenerateRequest(
            instruction="payment flow",
            domain_hint="online_payment",
            custom_steps=["retry_payment"],
            prefer_llm_generation=True,
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.metadata["generation_engine"] == "deterministic_fallback"
        assert "simulated ollama memory failure" in response.workflow.metadata["fallback_reason"]
        assert response.validation is not None
        assert "skipped_llm" not in response.validation.checks_performed

    @pytest.mark.asyncio
    async def test_generate_from_bullet_list_instruction(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction=(
                "domain: ci_cd_deployment\n"
                "- code push detected\n"
                "- run linting\n"
                "- run unit tests\n"
                "- deploy to staging\n"
                "- manual approval gate\n"
                "- notify team\n"
            )
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.domain == "ci_cd_deployment"
        assert response.workflow.metadata["input_format"] == "step_list"
        node_ids = {n.id for n in response.workflow.nodes}
        assert "approval_gate" in node_ids
        assert "notify_team" in node_ids

    @pytest.mark.asyncio
    async def test_generate_from_json_instruction_payload(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction=(
                '{'
                '"domain":"ci_cd_deployment",'
                '"instruction":"Build a release workflow",'
                '"steps":["Code Push Detected","Run Linting","Retry Build","Notify Team"]'
                '}'
            )
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.domain == "ci_cd_deployment"
        assert response.workflow.metadata["input_format"] == "json"
        node_ids = {n.id for n in response.workflow.nodes}
        assert "retry_build" in node_ids
        assert "notify_team" in node_ids

    @pytest.mark.asyncio
    async def test_generate_from_arrow_flow_instruction(
        self, pipeline: Pipeline
    ) -> None:
        request = GenerateRequest(
            instruction=(
                "domain: ci_cd_deployment\n"
                "Code Push Detected -> Run Linting -> Run Unit Tests -> "
                "Build Artifact -> Security Scan -> Notify Team"
            )
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.workflow is not None
        assert response.workflow.domain == "ci_cd_deployment"
        assert response.workflow.metadata["input_format"] == "arrow_flow"
        assert "notify_team" in {n.id for n in response.workflow.nodes}
