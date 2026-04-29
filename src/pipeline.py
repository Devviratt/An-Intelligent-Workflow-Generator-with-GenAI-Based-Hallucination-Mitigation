"""
Pipeline orchestrator connecting all engine stages.

Pipeline:
  Parse -> Generate -> Mitigate -> Validate -> Layout -> Optional Model -> Output

Supports two modes:
  - workflow
  - flowchart

Deterministic dataset-based generators are used by default for speed. If a
request explicitly opts into LLM generation, Ollama is attempted first and the
pipeline falls back to deterministic generation if Ollama fails.
"""

from __future__ import annotations

import logging
import time

from src.config import settings
from src.engines.domain_engine import DomainDatasetEngine
from src.engines.flowchart_generator import FlowchartGenerationError, FlowchartGenerator
from src.engines.hallucination_mitigation import HallucinationMitigator
from src.engines.input_adapter import InputAdapter
from src.engines.instruction_parser import InstructionParser
from src.engines.layout_engine import LayoutEngine
from src.engines.llm_flowchart_generator import (
    LLMFlowchartGenerationError,
    LLMFlowchartGenerator,
)
from src.engines.llm_workflow_generator import (
    LLMWorkflowGenerationError,
    LLMWorkflowGenerator,
)
from src.engines.local_model import LocalModelIntegration
from src.engines.validation_engine import ValidationEngine
from src.engines.workflow_generator import WorkflowGenerationError, WorkflowGenerator
from src.models.request import (
    ErrorDetail,
    GenerateRequest,
    GenerateResponse,
    GenerationMode,
    PipelineMetrics,
    ValidateRequest,
    ValidateResponse,
)
from src.models.validation import ValidationResult
from src.observability.evaluation import EvaluationRunner
from src.observability.explainability import ExplainabilityEngine
from src.observability.hallucination_metrics import HallucinationMetricsCollector
from src.observability.models import ObservabilityResult, StageName
from src.observability.profiler import StageProfiler

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Async orchestrator for the complete workflow generation pipeline.

    Lifecycle:
        pipeline = Pipeline()
        await pipeline.initialise()
        response = await pipeline.generate(request)
    """

    def __init__(self) -> None:
        self._dataset_engine = DomainDatasetEngine()
        self._parser = InstructionParser()
        self._input_adapter = InputAdapter()
        self._llm_generator = LLMWorkflowGenerator()
        self._llm_flowchart_generator = LLMFlowchartGenerator()
        self._generator = WorkflowGenerator()
        self._flowchart_generator = FlowchartGenerator()
        self._mitigator = HallucinationMitigator()
        self._validator = ValidationEngine()
        self._layout = LayoutEngine()
        self._local_model = LocalModelIntegration()
        self._hallucination_collector = HallucinationMetricsCollector()
        self._explainability = ExplainabilityEngine()
        self._evaluation_runner = EvaluationRunner()
        self._initialised = False

    async def initialise(self) -> None:
        """Load datasets and fit the parser."""
        if self._initialised:
            return

        await self._dataset_engine.load_all()
        datasets = self._dataset_engine.all_datasets()
        self._parser.fit(datasets)
        self._initialised = True
        logger.info("Pipeline initialised with %d domains", len(datasets))

    def _ensure_initialised(self) -> None:
        if not self._initialised:
            raise RuntimeError("Pipeline.initialise() must be called first")

    @property
    def dataset_engine(self) -> DomainDatasetEngine:
        return self._dataset_engine

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Execute the full generation pipeline."""
        self._ensure_initialised()
        metrics = PipelineMetrics()
        total_start = time.perf_counter()
        profiler = StageProfiler()
        profiler.start_total()

        try:
            adapted = self._input_adapter.adapt(request.instruction)

            with profiler.stage(StageName.PARSE, input_count=1):
                parsed = self._parser.parse(adapted.normalized_instruction)
                parsed = parsed.model_copy(
                    update={
                        "original_text": request.instruction,
                        "custom_steps_requested": adapted.step_hints,
                    }
                )
            metrics.parse_time_ms = profiler.get(StageName.PARSE).duration_ms
            profiler.set_output_count(StageName.PARSE, 1)

            domain = request.domain_hint or adapted.domain_hint or parsed.selected_domain
            if not domain:
                return GenerateResponse(
                    success=False,
                    errors=[
                        ErrorDetail(
                            code="NO_DOMAIN_MATCH",
                            message=(
                                "Could not determine domain from instruction. "
                                f"Available domains: {self._dataset_engine.domains}"
                            ),
                        )
                    ],
                    metrics=metrics,
                )

            dataset = self._dataset_engine.get(domain)
            if dataset is None:
                return GenerateResponse(
                    success=False,
                    errors=[
                        ErrorDetail(
                            code="DOMAIN_NOT_FOUND",
                            message=f"Domain '{domain}' not found.",
                        )
                    ],
                    metrics=metrics,
                )

            metrics.domain_selected = domain
            metrics.domain_confidence = (
                parsed.best_match.confidence if parsed.best_match else 0.0
            )

            request_mode = adapted.mode_hint or request.mode
            is_flowchart = request_mode == GenerationMode.FLOWCHART
            generation_engine = "deterministic"
            fallback_reason: str | None = None
            resolved_custom_steps = self._input_adapter.resolve_custom_steps(
                dataset,
                adapted.step_hints,
                request.custom_steps,
            )

            with profiler.stage(
                StageName.GENERATE,
                input_count=len(dataset.steps),
                metadata={"mode": request_mode.value, "engine": generation_engine},
            ):
                if request.prefer_llm_generation:
                    generation_engine = "llm"
                    profiler.add_stage_metadata(
                        StageName.GENERATE, "engine", generation_engine
                    )
                    try:
                        if is_flowchart:
                            workflow = await self._llm_flowchart_generator.generate(
                                dataset=dataset,
                                parsed=parsed,
                                include_optional=request.include_optional_steps,
                            )
                        else:
                            workflow = await self._llm_generator.generate(
                                dataset=dataset,
                                parsed=parsed,
                                include_optional=request.include_optional_steps,
                            )
                    except (LLMWorkflowGenerationError, LLMFlowchartGenerationError) as exc:
                        fallback_reason = str(exc)
                        generation_engine = "deterministic_fallback"
                        profiler.add_stage_metadata(
                            StageName.GENERATE, "engine", generation_engine
                        )
                        profiler.add_stage_metadata(
                            StageName.GENERATE, "fallback_reason", fallback_reason
                        )
                        logger.warning(
                            "LLM generation failed, using deterministic fallback: %s",
                            exc,
                        )
                        if is_flowchart:
                            workflow = self._flowchart_generator.generate(
                                dataset=dataset,
                                parsed=parsed,
                                include_optional=request.include_optional_steps,
                            )
                        else:
                            workflow = self._generator.generate(
                                dataset=dataset,
                                parsed=parsed,
                                include_optional=request.include_optional_steps,
                                custom_steps=resolved_custom_steps,
                            )
                elif is_flowchart:
                    workflow = self._flowchart_generator.generate(
                        dataset=dataset,
                        parsed=parsed,
                        include_optional=request.include_optional_steps,
                    )
                else:
                    workflow = self._generator.generate(
                        dataset=dataset,
                        parsed=parsed,
                        include_optional=request.include_optional_steps,
                        custom_steps=resolved_custom_steps,
                    )

            metrics.generation_time_ms = profiler.get(StageName.GENERATE).duration_ms
            profiler.set_output_count(StageName.GENERATE, len(workflow.nodes))

            pre_mitigation_nodes = len(workflow.nodes)
            pre_mitigation_edges = len(workflow.edges)

            # Mitigation: run for ALL generation engines (LLM output is post-processed
            # by LLMPostProcessor, but mitigation provides an additional validation layer)
            with profiler.stage(StageName.MITIGATE, input_count=len(workflow.nodes)):
                if is_flowchart:
                    workflow, mitigation_result = self._mitigator.mitigate_flowchart(
                        workflow, dataset
                    )
                else:
                    workflow, mitigation_result = self._mitigator.mitigate(
                        workflow, dataset
                    )
            metrics.mitigation_time_ms = profiler.get(StageName.MITIGATE).duration_ms
            profiler.set_output_count(StageName.MITIGATE, len(workflow.nodes))

            # Validation: ALWAYS run for grounding and safety (dataset as validation layer)
            with profiler.stage(StageName.VALIDATE, input_count=len(workflow.nodes)):
                if is_flowchart:
                    validation_result = self._validator.validate_flowchart(
                        workflow, dataset
                    )
                else:
                    validation_result = self._validator.validate(workflow, dataset)
            metrics.validation_time_ms = profiler.get(StageName.VALIDATE).duration_ms
            profiler.set_output_count(StageName.VALIDATE, len(workflow.nodes))

            with profiler.stage(StageName.LAYOUT, input_count=len(workflow.nodes)):
                if is_flowchart:
                    workflow = self._layout.compute_flowchart_layout(workflow)
                else:
                    workflow = self._layout.compute_layout(workflow)
            metrics.layout_time_ms = profiler.get(StageName.LAYOUT).duration_ms
            profiler.set_output_count(StageName.LAYOUT, len(workflow.nodes))

            if request.use_local_model and settings.use_local_model:
                try:
                    with profiler.stage(StageName.LOCAL_MODEL):
                        workflow = await self._local_model.refine_descriptions(
                            workflow, dataset
                        )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("Local model refinement skipped: %s", exc)

            with profiler.stage(StageName.EXPLAINABILITY, input_count=len(workflow.nodes)):
                workflow = self._explainability.annotate_nodes(workflow, dataset)
            profiler.set_output_count(StageName.EXPLAINABILITY, len(workflow.nodes))

            profiler.end_total()
            metrics.nodes_generated = len(workflow.nodes)
            metrics.edges_generated = len(workflow.edges)
            metrics.total_time_ms = round(
                (time.perf_counter() - total_start) * 1000, 2
            )

            workflow.metadata["generation_engine"] = generation_engine
            workflow.metadata["input_format"] = adapted.detected_format
            workflow.metadata["adapted_instruction"] = adapted.normalized_instruction
            workflow.metadata["resolved_custom_steps"] = resolved_custom_steps
            if adapted.domain_hint:
                workflow.metadata["input_domain_hint"] = adapted.domain_hint
            if fallback_reason:
                workflow.metadata["fallback_reason"] = fallback_reason

            observability: ObservabilityResult | None = None
            if request.evaluation_mode:
                h_metrics = self._hallucination_collector.collect(
                    mitigation_result=mitigation_result,
                    workflow_before_nodes=pre_mitigation_nodes,
                    workflow_before_edges=pre_mitigation_edges,
                    workflow_after=workflow,
                )
                explain_entry = self._explainability.build(
                    workflow,
                    dataset,
                    parsed,
                    mode=request_mode.value,
                )
                stage_metrics = profiler.collect()
                eval_report = self._evaluation_runner.build_report(
                    success=validation_result.is_valid,
                    workflow=workflow,
                    validation=validation_result,
                    dataset=dataset,
                    stage_metrics=stage_metrics,
                    hallucination_metrics=h_metrics,
                    explainability=explain_entry,
                    request_instruction=request.instruction,
                    request_mode=request_mode.value,
                )
                observability = ObservabilityResult(
                    stage_metrics=stage_metrics,
                    hallucination_metrics=h_metrics,
                    explainability=explain_entry,
                    evaluation=eval_report,
                )

            return GenerateResponse(
                success=validation_result.is_valid,
                workflow=workflow,
                validation=validation_result,
                metrics=metrics,
                errors=[
                    ErrorDetail(code="VALIDATION_ERROR", message=issue.message)
                    for issue in validation_result.issues
                    if issue.severity.value == "error"
                ],
                observability=observability,
            )

        except (WorkflowGenerationError, FlowchartGenerationError) as exc:
            profiler.end_total()
            metrics.total_time_ms = round(
                (time.perf_counter() - total_start) * 1000, 2
            )
            return GenerateResponse(
                success=False,
                errors=[ErrorDetail(code="GENERATION_ERROR", message=str(exc))],
                metrics=metrics,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected pipeline error")
            profiler.end_total()
            metrics.total_time_ms = round(
                (time.perf_counter() - total_start) * 1000, 2
            )
            return GenerateResponse(
                success=False,
                errors=[ErrorDetail(code="INTERNAL_ERROR", message=str(exc))],
                metrics=metrics,
            )

    async def validate(self, request: ValidateRequest) -> ValidateResponse:
        """Validate an existing workflow without generating."""
        self._ensure_initialised()

        dataset = None
        if request.domain:
            dataset = self._dataset_engine.get(request.domain)
        elif request.workflow.domain:
            dataset = self._dataset_engine.get(request.workflow.domain)

        result = self._validator.validate(request.workflow, dataset)

        if dataset:
            _, mitigation_result = self._mitigator.mitigate(request.workflow, dataset)
            result.merge(mitigation_result)

        return ValidateResponse(
            success=result.is_valid,
            validation=result,
            errors=[
                ErrorDetail(code="VALIDATION_ERROR", message=issue.message)
                for issue in result.issues
                if issue.severity.value == "error"
            ],
        )
