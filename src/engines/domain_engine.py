"""
Domain Dataset Engine — loads, indexes, and queries domain datasets.

Responsibilities:
  - Load JSON datasets from disk at startup
  - Validate datasets against Pydantic schema
  - Provide fast lookup by domain name
  - Support keyword-based search
  - Thread-safe read-only access after initialisation
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Sequence

import orjson

from src.config import settings
from src.models.domain import DomainDataset

logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Raised when a dataset file cannot be loaded or validated."""


class DomainDatasetEngine:
    """
    Registry of domain datasets loaded from disk.

    Lifecycle:
        engine = DomainDatasetEngine()
        await engine.load_all()          # or engine.load_all_sync()
        ds = engine.get("online_payment")
    """

    def __init__(self, datasets_dir: Path | None = None) -> None:
        self._dir = datasets_dir or settings.datasets_dir
        self._registry: dict[str, DomainDataset] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load_all(self) -> None:
        """Asynchronously load all .json datasets from the datasets directory."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.load_all_sync)

    def load_all_sync(self) -> None:
        """Synchronously load all .json datasets."""
        if not self._dir.is_dir():
            raise DatasetLoadError(f"Datasets directory not found: {self._dir}")

        json_files = sorted(self._dir.glob("*.json"))
        if not json_files:
            raise DatasetLoadError(f"No .json files found in {self._dir}")

        for fp in json_files:
            try:
                self._load_file(fp)
            except Exception as exc:
                logger.error("Failed to load dataset %s: %s", fp.name, exc)
                raise DatasetLoadError(f"Error loading {fp.name}: {exc}") from exc

        self._loaded = True
        logger.info(
            "Loaded %d domain datasets: %s",
            len(self._registry),
            ", ".join(sorted(self._registry)),
        )

    def _load_file(self, path: Path) -> None:
        """Load and validate a single dataset file."""
        raw = path.read_bytes()
        data = orjson.loads(raw)
        dataset = DomainDataset.model_validate(data)

        if dataset.domain in self._registry:
            raise DatasetLoadError(
                f"Duplicate domain '{dataset.domain}' — "
                f"already loaded from another file"
            )

        # Integrity check: start_node and end_node must exist in steps
        step_ids = dataset.step_ids
        if dataset.start_node not in step_ids:
            raise DatasetLoadError(
                f"start_node '{dataset.start_node}' not found in steps"
            )
        if dataset.end_node not in step_ids:
            raise DatasetLoadError(
                f"end_node '{dataset.end_node}' not found in steps"
            )

        # Verify all transitions reference existing steps
        for t in dataset.transitions:
            if t.from_step not in step_ids:
                raise DatasetLoadError(
                    f"Transition from unknown step '{t.from_step}'"
                )
            if t.to_step not in step_ids:
                raise DatasetLoadError(
                    f"Transition to unknown step '{t.to_step}'"
                )

        self._registry[dataset.domain] = dataset

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def domains(self) -> list[str]:
        return sorted(self._registry.keys())

    def get(self, domain: str) -> DomainDataset | None:
        """Get dataset by domain name."""
        return self._registry.get(domain)

    def get_strict(self, domain: str) -> DomainDataset:
        """Get dataset or raise."""
        ds = self._registry.get(domain)
        if ds is None:
            raise KeyError(f"Domain '{domain}' not found in registry")
        return ds

    def all_datasets(self) -> list[DomainDataset]:
        """Return all loaded datasets (deterministic order)."""
        return [self._registry[k] for k in sorted(self._registry)]

    def search_by_keyword(self, keyword: str) -> list[tuple[str, float]]:
        """Return domains matching a keyword, with match score."""
        keyword = keyword.lower()
        results: list[tuple[str, float]] = []
        for domain, ds in self._registry.items():
            kw_set = {k.lower() for k in ds.keywords}
            if keyword in kw_set:
                results.append((domain, 1.0))
            else:
                # Partial match
                partial = [k for k in kw_set if keyword in k or k in keyword]
                if partial:
                    results.append((domain, 0.5))
        results.sort(key=lambda r: r[1], reverse=True)
        return results
