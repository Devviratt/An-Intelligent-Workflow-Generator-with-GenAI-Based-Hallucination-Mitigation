"""Tests for the Domain Dataset Engine."""

from __future__ import annotations

import pytest

from src.engines.domain_engine import DomainDatasetEngine
from src.models.domain import DomainDataset


class TestDomainDatasetEngine:
    """Tests for dataset loading and querying."""

    def test_loads_all_datasets(self, dataset_engine: DomainDatasetEngine) -> None:
        assert dataset_engine.loaded
        assert len(dataset_engine.domains) >= 5

    def test_known_domains_present(self, dataset_engine: DomainDatasetEngine) -> None:
        expected = [
            "online_payment",
            "user_registration",
            "order_fulfillment",
            "incident_response",
            "data_pipeline",
        ]
        for domain in expected:
            assert dataset_engine.get(domain) is not None, f"Missing domain: {domain}"

    def test_dataset_structure(self, dataset_engine: DomainDatasetEngine) -> None:
        ds = dataset_engine.get_strict("online_payment")
        assert ds.start_node == "initiate_payment"
        assert ds.end_node == "transaction_complete"
        assert len(ds.steps) > 5
        assert len(ds.transitions) > 5
        assert ds.start_node in ds.step_ids
        assert ds.end_node in ds.step_ids

    def test_step_types(self, dataset_engine: DomainDatasetEngine) -> None:
        ds = dataset_engine.get_strict("online_payment")
        types = {s.type for s in ds.steps}
        assert "start" in types
        assert "end" in types
        assert "process" in types
        assert "decision" in types

    def test_decision_nodes_have_branches(self, dataset_engine: DomainDatasetEngine) -> None:
        for ds in dataset_engine.all_datasets():
            for step in ds.steps:
                if step.type == "decision":
                    assert step.branches is not None, (
                        f"Decision node {step.id} in {ds.domain} has no branches"
                    )
                    assert len(step.branches) >= 2, (
                        f"Decision node {step.id} in {ds.domain} has < 2 branches"
                    )

    def test_transitions_reference_valid_steps(
        self, dataset_engine: DomainDatasetEngine
    ) -> None:
        for ds in dataset_engine.all_datasets():
            ids = ds.step_ids
            for t in ds.transitions:
                assert t.from_step in ids, f"Invalid from: {t.from_step} in {ds.domain}"
                assert t.to_step in ids, f"Invalid to: {t.to_step} in {ds.domain}"

    def test_search_by_keyword(self, dataset_engine: DomainDatasetEngine) -> None:
        results = dataset_engine.search_by_keyword("payment")
        assert len(results) >= 1
        assert results[0][0] == "online_payment"

    def test_unknown_domain_returns_none(self, dataset_engine: DomainDatasetEngine) -> None:
        assert dataset_engine.get("nonexistent_domain") is None

    def test_get_strict_raises(self, dataset_engine: DomainDatasetEngine) -> None:
        with pytest.raises(KeyError):
            dataset_engine.get_strict("nonexistent_domain")
