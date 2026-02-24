"""
Benchmark & Observability Test Suite
=====================================

Tests for the observability layer:
  1. StageProfiler — per-stage timing / memory capture
  2. HallucinationMetricsCollector — passive extraction from mitigation
  3. ExplainabilityEngine — provenance annotation per node
  4. EvaluationRunner — research-evaluation aggregation
  5. Pipeline integration — evaluation_mode end-to-end

All tests are additive — core generation logic is NOT tested here.
"""

from __future__ import annotations

import time

import pytest
import pytest_asyncio

from src.models.domain import (
    DomainDataset,
    DomainStep,
    DomainTransition,
    ValidationRules,
)
from src.models.parser import DomainMatch, ExtractedKeywords, ParsedInstruction
from src.models.request import GenerateRequest, GenerateResponse, PipelineMetrics
from src.models.validation import (
    IssueCategory,
    IssueSeverity,
    ValidationIssue,
    ValidationResult,
)
from src.models.workflow import (
    EdgeStyle,
    GeneratedWorkflow,
    NodeType,
    WorkflowEdge,
    WorkflowNode,
)
from src.observability.explainability import ExplainabilityEngine
from src.observability.evaluation import EvaluationRunner
from src.observability.hallucination_metrics import HallucinationMetricsCollector
from src.observability.models import (
    EvaluationReport,
    ExplainabilityEntry,
    HallucinationMetrics,
    NodeProvenance,
    ObservabilityResult,
    StageMetric,
    StageName,
)
from src.observability.profiler import StageProfiler
from src.pipeline import Pipeline


# =====================================================================
# Fixtures — synthetic domain artefacts for unit-level tests
# =====================================================================


@pytest.fixture
def _mini_dataset() -> DomainDataset:
    """Minimal valid dataset for unit tests."""
    return DomainDataset(
        domain="test_domain",
        display_name="Test Domain",
        version="1.0.0",
        keywords=["test", "benchmark"],
        start_node="start",
        end_node="end",
        steps=[
            DomainStep(id="start", label="Start", type="start", required=True),
            DomainStep(id="step_a", label="Step A", type="process", required=True),
            DomainStep(id="decide", label="Decide", type="decision", required=False, branches={"yes": "step_a", "no": "end"}),
            DomainStep(id="end", label="End", type="end", required=True),
        ],
        transitions=[
            DomainTransition(**{"from": "start", "to": "step_a"}),
            DomainTransition(**{"from": "step_a", "to": "decide"}),
            DomainTransition(**{"from": "decide", "to": "step_a", "condition": "yes"}),
            DomainTransition(**{"from": "decide", "to": "end", "condition": "no"}),
        ],
        validation_rules=ValidationRules(required_steps=["start", "step_a", "end"]),
    )


@pytest.fixture
def _mini_workflow() -> GeneratedWorkflow:
    """Workflow that matches _mini_dataset."""
    return GeneratedWorkflow(
        workflow_id="wf_test_001",
        domain="test_domain",
        title="Test Workflow",
        nodes=[
            WorkflowNode(id="start", label="Start", type=NodeType.START, domain_step_id="start"),
            WorkflowNode(id="step_a", label="Step A", type=NodeType.PROCESS, domain_step_id="step_a"),
            WorkflowNode(id="decide", label="Decide", type=NodeType.DECISION, domain_step_id="decide"),
            WorkflowNode(id="end", label="End", type=NodeType.END, domain_step_id="end"),
        ],
        edges=[
            WorkflowEdge(id="e1", source="start", target="step_a"),
            WorkflowEdge(id="e2", source="step_a", target="decide"),
            WorkflowEdge(id="e3", source="decide", target="step_a", style=EdgeStyle.RETRY_LOOP),
            WorkflowEdge(id="e4", source="decide", target="end"),
        ],
    )


@pytest.fixture
def _parsed_instruction() -> ParsedInstruction:
    return ParsedInstruction(
        original_text="run the test benchmark flow",
        cleaned_text="run test benchmark flow",
        keywords=ExtractedKeywords(
            raw_tokens=["run", "test", "benchmark", "flow"],
            cleaned_tokens=["run", "test", "benchmark", "flow"],
        ),
        domain_matches=[DomainMatch(domain="test_domain", confidence=0.85, matched_keywords=["test"])],
        selected_domain="test_domain",
    )


@pytest.fixture
def _clean_mitigation_result() -> ValidationResult:
    """Mitigation result with no issues (perfect grounding)."""
    return ValidationResult(
        is_valid=True,
        issues=[],
        nodes_validated=4,
        edges_validated=4,
        checks_performed=["grounding_check", "duplicate_check"],
    )


@pytest.fixture
def _noisy_mitigation_result() -> ValidationResult:
    """Mitigation result with several hallucination issues."""
    return ValidationResult(
        is_valid=True,
        issues=[
            ValidationIssue(severity=IssueSeverity.WARNING, category=IssueCategory.GROUNDING, message="Node X not grounded", node_id="x"),
            ValidationIssue(severity=IssueSeverity.WARNING, category=IssueCategory.GROUNDING, message="Node Y not grounded", node_id="y"),
            ValidationIssue(severity=IssueSeverity.WARNING, category=IssueCategory.DUPLICATE, message="Duplicate node Z", node_id="z"),
            ValidationIssue(severity=IssueSeverity.INFO, category=IssueCategory.STRUCTURE, message="Dangling edge"),
            ValidationIssue(severity=IssueSeverity.WARNING, category=IssueCategory.TRANSITION, message="Forbidden transition A→B"),
            ValidationIssue(severity=IssueSeverity.WARNING, category=IssueCategory.RETRY, message="Retry limit exceeded"),
        ],
        nodes_validated=6,
        edges_validated=5,
        checks_performed=["grounding_check", "duplicate_check", "structure_check"],
    )


# =====================================================================
# 1. StageProfiler tests
# =====================================================================


class TestStageProfiler:
    """Unit tests for the per-stage performance profiler."""

    def test_stage_captures_timing(self) -> None:
        """A stage context manager should capture positive timing."""
        profiler = StageProfiler()
        profiler.start_total()

        with profiler.stage(StageName.PARSE, input_count=1):
            time.sleep(0.01)  # ~10ms
        profiler.set_output_count(StageName.PARSE, 1)

        profiler.end_total()

        metric = profiler.get(StageName.PARSE)
        assert metric is not None
        assert metric.stage == StageName.PARSE
        assert metric.duration_ms >= 5  # at least a few ms
        assert metric.input_count == 1
        assert metric.output_count == 1

    def test_multiple_stages(self) -> None:
        """Multiple stages should be independently captured."""
        profiler = StageProfiler()
        profiler.start_total()

        with profiler.stage(StageName.PARSE, input_count=1):
            time.sleep(0.005)
        profiler.set_output_count(StageName.PARSE, 3)

        with profiler.stage(StageName.GENERATE, input_count=3):
            time.sleep(0.005)
        profiler.set_output_count(StageName.GENERATE, 10)

        with profiler.stage(StageName.VALIDATE, input_count=10):
            time.sleep(0.005)
        profiler.set_output_count(StageName.VALIDATE, 10)

        profiler.end_total()

        metrics = profiler.collect()
        assert len(metrics) == 3
        stages = {m.stage for m in metrics}
        assert stages == {StageName.PARSE, StageName.GENERATE, StageName.VALIDATE}

    def test_total_metric_from_marks(self) -> None:
        """total_metric() should use start_total / end_total marks."""
        profiler = StageProfiler()
        profiler.start_total()
        time.sleep(0.015)
        profiler.end_total()

        total = profiler.total_metric()
        assert total.stage == StageName.TOTAL
        assert total.duration_ms >= 10

    def test_total_metric_fallback_sum(self) -> None:
        """Without marks, total_metric() sums individual stages."""
        profiler = StageProfiler()

        with profiler.stage(StageName.PARSE):
            time.sleep(0.005)
        with profiler.stage(StageName.GENERATE):
            time.sleep(0.005)

        total = profiler.total_metric()
        individual = sum(m.duration_ms for m in profiler.collect())
        assert abs(total.duration_ms - individual) < 1.0  # within 1ms

    def test_collect_excludes_total(self) -> None:
        """collect() should not include the TOTAL stage."""
        profiler = StageProfiler()
        profiler.start_total()
        with profiler.stage(StageName.PARSE):
            pass
        profiler.end_total()

        for m in profiler.collect():
            assert m.stage != StageName.TOTAL

    def test_stage_metadata(self) -> None:
        """Extra metadata can be attached to a stage."""
        profiler = StageProfiler()
        with profiler.stage(StageName.GENERATE, metadata={"domain": "test"}):
            pass
        profiler.add_stage_metadata(StageName.GENERATE, "extra", 42)

        metric = profiler.get(StageName.GENERATE)
        assert metric is not None
        assert metric.metadata["domain"] == "test"
        assert metric.metadata["extra"] == 42

    def test_memory_delta_is_numeric(self) -> None:
        """memory_delta_kb should be a number (may be zero for tiny ops)."""
        profiler = StageProfiler()
        with profiler.stage(StageName.PARSE):
            _ = [0] * 1000
        metric = profiler.get(StageName.PARSE)
        assert metric is not None
        assert isinstance(metric.memory_delta_kb, float)

    def test_get_nonexistent_stage(self) -> None:
        """get() for an uncaptured stage returns None."""
        profiler = StageProfiler()
        assert profiler.get(StageName.LAYOUT) is None

    def test_set_output_count_ignored_for_missing(self) -> None:
        """set_output_count for uncaptured stage should not raise."""
        profiler = StageProfiler()
        profiler.set_output_count(StageName.LAYOUT, 99)  # no-op, no error


# =====================================================================
# 2. HallucinationMetricsCollector tests
# =====================================================================


class TestHallucinationMetricsCollector:
    """Unit tests for passive hallucination metrics extraction."""

    def test_clean_result_perfect_grounding(
        self,
        _clean_mitigation_result: ValidationResult,
        _mini_workflow: GeneratedWorkflow,
    ) -> None:
        """A clean mitigation result should yield perfect scores."""
        collector = HallucinationMetricsCollector()
        metrics = collector.collect(
            mitigation_result=_clean_mitigation_result,
            workflow_before_nodes=4,
            workflow_before_edges=4,
            workflow_after=_mini_workflow,
        )

        assert isinstance(metrics, HallucinationMetrics)
        assert metrics.nodes_removed == 0
        assert metrics.edges_removed == 0
        assert metrics.grounding_violations == 0
        assert metrics.duplicates_removed == 0
        assert metrics.node_grounding_rate == 1.0
        assert metrics.edge_grounding_rate == 1.0
        assert metrics.hallucination_score == 0.0

    def test_noisy_result_detects_issues(
        self,
        _noisy_mitigation_result: ValidationResult,
        _mini_workflow: GeneratedWorkflow,
    ) -> None:
        """A noisy mitigation result should report violations."""
        collector = HallucinationMetricsCollector()
        # Simulate that we started with 6 nodes / 5 edges and ended with 4/4
        metrics = collector.collect(
            mitigation_result=_noisy_mitigation_result,
            workflow_before_nodes=6,
            workflow_before_edges=5,
            workflow_after=_mini_workflow,
        )

        assert metrics.grounding_violations == 2
        assert metrics.duplicates_removed == 1
        assert metrics.structural_issues == 1
        assert metrics.forbidden_transitions_found == 1
        assert metrics.retry_issues == 1
        assert metrics.nodes_removed == 2
        assert metrics.edges_removed == 1

        # Rates should reflect removals
        assert 0.0 < metrics.node_grounding_rate < 1.0
        assert 0.0 < metrics.edge_grounding_rate < 1.0
        assert metrics.hallucination_score > 0.0

    def test_grounding_rates_bounded(
        self,
        _noisy_mitigation_result: ValidationResult,
        _mini_workflow: GeneratedWorkflow,
    ) -> None:
        """Grounding rates and hallucination score stay in [0, 1]."""
        collector = HallucinationMetricsCollector()
        metrics = collector.collect(
            mitigation_result=_noisy_mitigation_result,
            workflow_before_nodes=6,
            workflow_before_edges=5,
            workflow_after=_mini_workflow,
        )

        assert 0.0 <= metrics.node_grounding_rate <= 1.0
        assert 0.0 <= metrics.edge_grounding_rate <= 1.0
        assert 0.0 <= metrics.hallucination_score <= 1.0

    def test_zero_input_yields_perfect_rate(self) -> None:
        """Zero starting nodes/edges should not divide by zero."""
        collector = HallucinationMetricsCollector()
        metrics = collector.collect(
            mitigation_result=ValidationResult(is_valid=True),
            workflow_before_nodes=0,
            workflow_before_edges=0,
            workflow_after=GeneratedWorkflow(
                workflow_id="empty", domain="x", title="Empty",
            ),
        )

        assert metrics.node_grounding_rate == 1.0
        assert metrics.edge_grounding_rate == 1.0
        assert metrics.hallucination_score == 0.0

    def test_total_counts_match_input(
        self,
        _clean_mitigation_result: ValidationResult,
        _mini_workflow: GeneratedWorkflow,
    ) -> None:
        """total_nodes/edges_checked should reflect the before counts."""
        collector = HallucinationMetricsCollector()
        metrics = collector.collect(
            mitigation_result=_clean_mitigation_result,
            workflow_before_nodes=10,
            workflow_before_edges=8,
            workflow_after=_mini_workflow,
        )

        assert metrics.total_nodes_checked == 10
        assert metrics.total_edges_checked == 8


# =====================================================================
# 3. ExplainabilityEngine tests
# =====================================================================


class TestExplainabilityEngine:
    """Unit tests for the explainability / provenance engine."""

    def test_build_returns_entry(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """build() should return a well-formed ExplainabilityEntry."""
        engine = ExplainabilityEngine()
        entry = engine.build(_mini_workflow, _mini_dataset, _parsed_instruction, mode="workflow")

        assert isinstance(entry, ExplainabilityEntry)
        assert entry.generation_mode == "workflow"
        assert entry.domain_used == "test_domain"
        assert entry.dataset_version == "1.0.0"
        assert entry.total_dataset_steps == 4
        assert entry.steps_selected == 4

    def test_provenance_per_node(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """Each node should get a provenance record."""
        engine = ExplainabilityEngine()
        entry = engine.build(_mini_workflow, _mini_dataset, _parsed_instruction)

        assert len(entry.node_provenance) == len(_mini_workflow.nodes)
        node_ids = {p.node_id for p in entry.node_provenance}
        assert node_ids == {n.id for n in _mini_workflow.nodes}

    def test_grounded_nodes_marked_correctly(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """Nodes with valid domain_step_id should be grounded."""
        engine = ExplainabilityEngine()
        entry = engine.build(_mini_workflow, _mini_dataset, _parsed_instruction)

        for prov in entry.node_provenance:
            assert prov.grounding_status == "grounded"
            assert prov.source == "dataset"

    def test_ungrounded_node_detected(
        self,
        _mini_dataset: DomainDataset,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """A node whose domain_step_id is NOT in the dataset should be unverified."""
        wf = GeneratedWorkflow(
            workflow_id="wf_ungrounded",
            domain="test_domain",
            title="Ungrounded",
            nodes=[
                WorkflowNode(id="start", label="Start", type=NodeType.START, domain_step_id="start"),
                WorkflowNode(id="alien", label="Alien", type=NodeType.PROCESS, domain_step_id="alien_step"),
                WorkflowNode(id="end", label="End", type=NodeType.END, domain_step_id="end"),
            ],
            edges=[
                WorkflowEdge(id="e1", source="start", target="alien"),
                WorkflowEdge(id="e2", source="alien", target="end"),
            ],
        )
        engine = ExplainabilityEngine()
        entry = engine.build(wf, _mini_dataset, _parsed_instruction)

        by_id = {p.node_id: p for p in entry.node_provenance}
        assert by_id["alien"].grounding_status == "unverified"
        assert by_id["alien"].source == "inferred"
        assert by_id["start"].grounding_status == "grounded"

    def test_decision_and_retry_counts(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """Decision and retry edge counts should be populated."""
        engine = ExplainabilityEngine()
        entry = engine.build(_mini_workflow, _mini_dataset, _parsed_instruction)

        assert entry.decision_nodes_count == 1  # "decide" node
        assert entry.retry_edges_count == 1  # one RETRY_LOOP edge

    def test_annotate_nodes_adds_metadata(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
    ) -> None:
        """annotate_nodes() should inject _explainability into each node.metadata."""
        engine = ExplainabilityEngine()
        annotated = engine.annotate_nodes(_mini_workflow, _mini_dataset)

        for node in annotated.nodes:
            assert "_explainability" in node.metadata
            exp = node.metadata["_explainability"]
            assert "source" in exp
            assert "grounding_status" in exp
            assert "dataset_domain" in exp
            assert exp["dataset_domain"] == "test_domain"

    def test_instruction_keywords_captured(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """Instruction keywords should be included in the entry."""
        engine = ExplainabilityEngine()
        entry = engine.build(_mini_workflow, _mini_dataset, _parsed_instruction)

        assert len(entry.instruction_keywords) > 0
        assert "test" in entry.instruction_keywords

    def test_domain_confidence_captured(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """Domain confidence from parsed instruction should propagate."""
        engine = ExplainabilityEngine()
        entry = engine.build(_mini_workflow, _mini_dataset, _parsed_instruction)

        assert entry.domain_confidence == pytest.approx(0.85, abs=0.01)

    def test_transition_counts_per_node(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """Each node provenance should have the correct outgoing edge count."""
        engine = ExplainabilityEngine()
        entry = engine.build(_mini_workflow, _mini_dataset, _parsed_instruction)

        by_id = {p.node_id: p for p in entry.node_provenance}
        # "decide" has 2 outgoing edges (e3 → step_a, e4 → end)
        assert by_id["decide"].transition_count == 2
        assert by_id["end"].transition_count == 0


# =====================================================================
# 4. EvaluationRunner tests
# =====================================================================


class TestEvaluationRunner:
    """Unit tests for research evaluation report builder."""

    def _make_stage_metrics(self) -> list[StageMetric]:
        return [
            StageMetric(stage=StageName.PARSE, duration_ms=5.0),
            StageMetric(stage=StageName.GENERATE, duration_ms=20.0),
            StageMetric(stage=StageName.MITIGATE, duration_ms=10.0),
            StageMetric(stage=StageName.VALIDATE, duration_ms=8.0),
            StageMetric(stage=StageName.LAYOUT, duration_ms=3.0),
        ]

    def test_build_report_returns_report(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _clean_mitigation_result: ValidationResult,
        _parsed_instruction: ParsedInstruction,
    ) -> None:
        """build_report() should return a valid EvaluationReport."""
        runner = EvaluationRunner()
        h_metrics = HallucinationMetrics()
        explain = ExplainabilityEntry()
        stage_metrics = self._make_stage_metrics()

        report = runner.build_report(
            success=True,
            workflow=_mini_workflow,
            validation=_clean_mitigation_result,
            dataset=_mini_dataset,
            stage_metrics=stage_metrics,
            hallucination_metrics=h_metrics,
            explainability=explain,
            request_instruction="test benchmark flow",
            request_mode="workflow",
        )

        assert isinstance(report, EvaluationReport)
        assert report.domain == "test_domain"
        assert report.mode == "workflow"
        assert report.validation_passed is True
        assert report.node_count == 4
        assert report.edge_count == 4

    def test_run_id_is_deterministic_prefix(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
    ) -> None:
        """run_id should start with eval_{domain}_."""
        runner = EvaluationRunner()
        report = runner.build_report(
            success=True,
            workflow=_mini_workflow,
            validation=ValidationResult(is_valid=True),
            dataset=_mini_dataset,
            stage_metrics=[],
            hallucination_metrics=HallucinationMetrics(),
            explainability=ExplainabilityEntry(),
            request_instruction="test",
            request_mode="workflow",
        )

        assert report.run_id.startswith("eval_test_domain_")

    def test_total_duration_summed(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
    ) -> None:
        """total_duration_ms should be the sum of stage durations."""
        runner = EvaluationRunner()
        stage_metrics = self._make_stage_metrics()
        expected = sum(m.duration_ms for m in stage_metrics)

        report = runner.build_report(
            success=True,
            workflow=_mini_workflow,
            validation=ValidationResult(is_valid=True),
            dataset=_mini_dataset,
            stage_metrics=stage_metrics,
            hallucination_metrics=HallucinationMetrics(),
            explainability=ExplainabilityEntry(),
            request_instruction="t",
            request_mode="workflow",
        )

        assert report.total_duration_ms == pytest.approx(expected, abs=0.01)

    def test_grounding_score_inverse(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
    ) -> None:
        """grounding_score = 1.0 - hallucination_score."""
        runner = EvaluationRunner()
        h_metrics = HallucinationMetrics(hallucination_score=0.3)

        report = runner.build_report(
            success=True,
            workflow=_mini_workflow,
            validation=ValidationResult(is_valid=True),
            dataset=_mini_dataset,
            stage_metrics=[],
            hallucination_metrics=h_metrics,
            explainability=ExplainabilityEntry(),
            request_instruction="t",
            request_mode="workflow",
        )

        assert report.grounding_score == pytest.approx(0.7, abs=0.001)

    def test_completeness_all_required_present(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
    ) -> None:
        """When all required steps are present, completeness = 1.0."""
        runner = EvaluationRunner()

        report = runner.build_report(
            success=True,
            workflow=_mini_workflow,
            validation=ValidationResult(is_valid=True),
            dataset=_mini_dataset,
            stage_metrics=[],
            hallucination_metrics=HallucinationMetrics(),
            explainability=ExplainabilityEntry(),
            request_instruction="t",
            request_mode="workflow",
        )

        # _mini_dataset requires ["start", "step_a", "end"] — all present
        assert report.completeness_score == pytest.approx(1.0, abs=0.001)

    def test_completeness_partial(
        self,
        _mini_dataset: DomainDataset,
    ) -> None:
        """Missing required steps reduce completeness."""
        partial_wf = GeneratedWorkflow(
            workflow_id="partial",
            domain="test_domain",
            title="Partial",
            nodes=[
                WorkflowNode(id="start", label="Start", type=NodeType.START),
                WorkflowNode(id="end", label="End", type=NodeType.END),
            ],
            edges=[WorkflowEdge(id="e1", source="start", target="end")],
        )
        runner = EvaluationRunner()

        report = runner.build_report(
            success=True,
            workflow=partial_wf,
            validation=ValidationResult(is_valid=True),
            dataset=_mini_dataset,
            stage_metrics=[],
            hallucination_metrics=HallucinationMetrics(),
            explainability=ExplainabilityEntry(),
            request_instruction="t",
            request_mode="workflow",
        )

        # Required: start, step_a, end → only start and end present = 2/3
        assert report.completeness_score == pytest.approx(2 / 3, abs=0.01)

    def test_null_workflow_handled(
        self,
        _mini_dataset: DomainDataset,
    ) -> None:
        """None workflow should yield zero counts and zero completeness."""
        runner = EvaluationRunner()

        report = runner.build_report(
            success=False,
            workflow=None,
            validation=None,
            dataset=_mini_dataset,
            stage_metrics=[],
            hallucination_metrics=HallucinationMetrics(),
            explainability=ExplainabilityEntry(),
            request_instruction="t",
            request_mode="workflow",
        )

        assert report.node_count == 0
        assert report.edge_count == 0
        assert report.completeness_score == 0.0
        assert report.validation_passed is False

    def test_decision_node_count(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
    ) -> None:
        """decision_node_count should reflect actual decision nodes."""
        runner = EvaluationRunner()

        report = runner.build_report(
            success=True,
            workflow=_mini_workflow,
            validation=ValidationResult(is_valid=True),
            dataset=_mini_dataset,
            stage_metrics=[],
            hallucination_metrics=HallucinationMetrics(),
            explainability=ExplainabilityEntry(),
            request_instruction="t",
            request_mode="workflow",
        )

        assert report.decision_node_count == 1

    def test_validation_error_counts(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
        _noisy_mitigation_result: ValidationResult,
    ) -> None:
        """Validation error/warning counts should be extracted."""
        runner = EvaluationRunner()

        report = runner.build_report(
            success=True,
            workflow=_mini_workflow,
            validation=_noisy_mitigation_result,
            dataset=_mini_dataset,
            stage_metrics=[],
            hallucination_metrics=HallucinationMetrics(),
            explainability=ExplainabilityEntry(),
            request_instruction="t",
            request_mode="workflow",
        )

        assert report.validation_error_count == _noisy_mitigation_result.error_count
        assert report.validation_warning_count == _noisy_mitigation_result.warning_count

    def test_timestamp_present(
        self,
        _mini_workflow: GeneratedWorkflow,
        _mini_dataset: DomainDataset,
    ) -> None:
        """Report should have a non-empty ISO timestamp."""
        runner = EvaluationRunner()

        report = runner.build_report(
            success=True,
            workflow=_mini_workflow,
            validation=ValidationResult(is_valid=True),
            dataset=_mini_dataset,
            stage_metrics=[],
            hallucination_metrics=HallucinationMetrics(),
            explainability=ExplainabilityEntry(),
            request_instruction="t",
            request_mode="workflow",
        )

        assert len(report.timestamp) > 10  # ISO format


# =====================================================================
# 5. Observability models serialisation tests
# =====================================================================


class TestObservabilityModels:
    """Schema-level tests for observability Pydantic models."""

    def test_stage_metric_defaults(self) -> None:
        m = StageMetric(stage=StageName.PARSE)
        assert m.duration_ms == 0.0
        assert m.memory_delta_kb == 0.0
        assert m.input_count == 0
        assert m.metadata == {}

    def test_hallucination_metrics_defaults(self) -> None:
        m = HallucinationMetrics()
        assert m.hallucination_score == 0.0
        assert m.node_grounding_rate == 1.0
        assert m.edge_grounding_rate == 1.0

    def test_observability_result_roundtrip(self) -> None:
        """ObservabilityResult should serialise and deserialise cleanly."""
        result = ObservabilityResult(
            stage_metrics=[StageMetric(stage=StageName.PARSE, duration_ms=5.0)],
            hallucination_metrics=HallucinationMetrics(hallucination_score=0.1),
        )
        data = result.model_dump()
        restored = ObservabilityResult.model_validate(data)
        assert restored.stage_metrics[0].duration_ms == 5.0
        assert restored.hallucination_metrics.hallucination_score == pytest.approx(0.1)

    def test_evaluation_report_roundtrip(self) -> None:
        """EvaluationReport should serialise and deserialise cleanly."""
        report = EvaluationReport(
            run_id="eval_test_abc",
            domain="test",
            mode="workflow",
            node_count=5,
            completeness_score=0.8,
            grounding_score=0.95,
        )
        data = report.model_dump()
        restored = EvaluationReport.model_validate(data)
        assert restored.run_id == "eval_test_abc"
        assert restored.completeness_score == pytest.approx(0.8)

    def test_node_provenance_defaults(self) -> None:
        p = NodeProvenance(node_id="n1")
        assert p.source == "dataset"
        assert p.grounding_status == "grounded"
        assert p.transition_count == 0

    def test_stage_name_values(self) -> None:
        """All expected stage names should be present."""
        names = set(StageName)
        assert StageName.PARSE in names
        assert StageName.GENERATE in names
        assert StageName.MITIGATE in names
        assert StageName.VALIDATE in names
        assert StageName.LAYOUT in names
        assert StageName.EXPLAINABILITY in names
        assert StageName.TOTAL in names


# =====================================================================
# 6. Pipeline integration — evaluation_mode
# =====================================================================


class TestPipelineEvaluationMode:
    """Integration tests: pipeline with evaluation_mode=True."""

    @pytest.mark.asyncio
    async def test_evaluation_mode_returns_observability(self, pipeline: Pipeline) -> None:
        """evaluation_mode should populate the observability field."""
        request = GenerateRequest(
            instruction="Create an online payment processing workflow",
            evaluation_mode=True,
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.observability is not None
        assert isinstance(response.observability, ObservabilityResult)

    @pytest.mark.asyncio
    async def test_evaluation_mode_has_stage_metrics(self, pipeline: Pipeline) -> None:
        """observability.stage_metrics should contain profiling data."""
        request = GenerateRequest(
            instruction="user registration signup flow",
            evaluation_mode=True,
        )
        response = await pipeline.generate(request)

        assert response.observability is not None
        metrics = response.observability.stage_metrics
        assert len(metrics) >= 5  # parse, generate, mitigate, validate, layout
        stages = {m.stage for m in metrics}
        assert StageName.PARSE in stages
        assert StageName.GENERATE in stages
        assert StageName.MITIGATE in stages
        assert StageName.VALIDATE in stages

    @pytest.mark.asyncio
    async def test_evaluation_mode_has_hallucination_metrics(self, pipeline: Pipeline) -> None:
        """observability should include hallucination metrics."""
        request = GenerateRequest(
            instruction="order fulfillment shipping workflow",
            evaluation_mode=True,
        )
        response = await pipeline.generate(request)

        assert response.observability is not None
        h = response.observability.hallucination_metrics
        assert isinstance(h, HallucinationMetrics)
        assert 0.0 <= h.node_grounding_rate <= 1.0
        assert 0.0 <= h.hallucination_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluation_mode_has_explainability(self, pipeline: Pipeline) -> None:
        """observability should include explainability data."""
        request = GenerateRequest(
            instruction="incident alert triage workflow",
            evaluation_mode=True,
            domain_hint="incident_response",
        )
        response = await pipeline.generate(request)

        assert response.observability is not None
        exp = response.observability.explainability
        assert isinstance(exp, ExplainabilityEntry)
        assert len(exp.node_provenance) > 0
        assert exp.domain_used == "incident_response"

    @pytest.mark.asyncio
    async def test_evaluation_mode_has_report(self, pipeline: Pipeline) -> None:
        """observability.evaluation should be a complete EvaluationReport."""
        request = GenerateRequest(
            instruction="data ETL pipeline",
            evaluation_mode=True,
            domain_hint="data_pipeline",
        )
        response = await pipeline.generate(request)

        assert response.observability is not None
        report = response.observability.evaluation
        assert isinstance(report, EvaluationReport)
        assert report.run_id.startswith("eval_data_pipeline_")
        assert report.domain == "data_pipeline"
        assert report.node_count > 0
        assert report.total_duration_ms > 0
        assert 0.0 <= report.completeness_score <= 1.0
        assert 0.0 <= report.grounding_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluation_mode_false_no_observability(self, pipeline: Pipeline) -> None:
        """Without evaluation_mode, observability should be None."""
        request = GenerateRequest(
            instruction="payment processing flow",
            evaluation_mode=False,
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.observability is None

    @pytest.mark.asyncio
    async def test_evaluation_mode_all_domains(self, pipeline: Pipeline) -> None:
        """Evaluation mode should work across all domains without error."""
        domains = [
            ("online_payment", "payment card transaction"),
            ("user_registration", "user signup registration"),
            ("order_fulfillment", "shipping order delivery"),
            ("incident_response", "incident alert triage"),
            ("data_pipeline", "data ETL pipeline"),
            ("ci_cd_deployment", "CI/CD deploy release"),
        ]
        for domain, instruction in domains:
            request = GenerateRequest(
                instruction=instruction,
                domain_hint=domain,
                evaluation_mode=True,
            )
            response = await pipeline.generate(request)

            assert response.success, f"Failed for {domain}: {response.errors}"
            assert response.observability is not None, f"No observability for {domain}"
            assert response.observability.evaluation is not None, f"No report for {domain}"
            report = response.observability.evaluation
            assert report.domain == domain
            assert report.node_count > 0

    @pytest.mark.asyncio
    async def test_evaluation_mode_flowchart(self, pipeline: Pipeline) -> None:
        """Evaluation mode should also work with flowchart generation."""
        request = GenerateRequest(
            instruction="payment processing flowchart",
            mode="flowchart",
            domain_hint="online_payment",
            evaluation_mode=True,
        )
        response = await pipeline.generate(request)

        assert response.success
        assert response.observability is not None
        assert response.observability.evaluation is not None
        assert response.observability.evaluation.mode == "flowchart"

    @pytest.mark.asyncio
    async def test_observability_serialisable(self, pipeline: Pipeline) -> None:
        """The full observability result should be JSON-serialisable."""
        request = GenerateRequest(
            instruction="payment flow",
            domain_hint="online_payment",
            evaluation_mode=True,
        )
        response = await pipeline.generate(request)

        assert response.observability is not None
        data = response.observability.model_dump()
        assert isinstance(data, dict)
        assert "stage_metrics" in data
        assert "hallucination_metrics" in data
        assert "explainability" in data
        assert "evaluation" in data

        # Round-trip
        restored = ObservabilityResult.model_validate(data)
        assert len(restored.stage_metrics) == len(response.observability.stage_metrics)

    @pytest.mark.asyncio
    async def test_stage_timings_positive(self, pipeline: Pipeline) -> None:
        """Each captured stage should have a positive duration."""
        request = GenerateRequest(
            instruction="CI/CD deployment pipeline",
            domain_hint="ci_cd_deployment",
            evaluation_mode=True,
        )
        response = await pipeline.generate(request)

        assert response.observability is not None
        for metric in response.observability.stage_metrics:
            assert metric.duration_ms >= 0.0, f"Negative timing for {metric.stage}"

    @pytest.mark.asyncio
    async def test_explainability_nodes_annotated(self, pipeline: Pipeline) -> None:
        """After evaluation_mode, workflow nodes should have _explainability metadata."""
        request = GenerateRequest(
            instruction="user registration flow",
            domain_hint="user_registration",
            evaluation_mode=True,
        )
        response = await pipeline.generate(request)

        assert response.workflow is not None
        for node in response.workflow.nodes:
            assert "_explainability" in node.metadata
            exp = node.metadata["_explainability"]
            assert "source" in exp
            assert "grounding_status" in exp
