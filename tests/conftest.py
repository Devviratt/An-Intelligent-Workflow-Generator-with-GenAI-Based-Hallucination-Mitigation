"""Shared test fixtures."""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.engines.domain_engine import DomainDatasetEngine
from src.engines.hallucination_mitigation import HallucinationMitigator
from src.engines.instruction_parser import InstructionParser
from src.engines.layout_engine import LayoutEngine
from src.engines.validation_engine import ValidationEngine
from src.engines.flowchart_generator import FlowchartGenerator
from src.engines.workflow_generator import WorkflowGenerator
from src.models.domain import DomainDataset
from src.pipeline import Pipeline


@pytest.fixture(scope="session")
def dataset_engine() -> DomainDatasetEngine:
    engine = DomainDatasetEngine()
    engine.load_all_sync()
    return engine


@pytest.fixture(scope="session")
def all_datasets(dataset_engine: DomainDatasetEngine) -> list[DomainDataset]:
    return dataset_engine.all_datasets()


@pytest.fixture(scope="session")
def parser(all_datasets: list[DomainDataset]) -> InstructionParser:
    p = InstructionParser()
    p.fit(all_datasets)
    return p


@pytest.fixture
def generator() -> WorkflowGenerator:
    return WorkflowGenerator()


@pytest.fixture
def flowchart_generator() -> FlowchartGenerator:
    return FlowchartGenerator()


@pytest.fixture
def mitigator() -> HallucinationMitigator:
    return HallucinationMitigator()


@pytest.fixture
def validator() -> ValidationEngine:
    return ValidationEngine()


@pytest.fixture
def layout_engine() -> LayoutEngine:
    return LayoutEngine()


@pytest_asyncio.fixture
async def pipeline() -> Pipeline:
    p = Pipeline()
    await p.initialise()
    return p
