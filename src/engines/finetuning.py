"""
Fine-tuning Data Preparation — generates training dataset for Ollama LLM.

Creates instruction-workflow pairs from domain datasets for fine-tuning
the local LLM to generate workflows correctly for each domain.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.domain import DomainDataset

logger = logging.getLogger(__name__)


class FineTuneDataGenerator:
    """
    Generate fine-tuning datasets for the local LLM.

    Usage:
        generator = FineTuneDataGenerator()
        jsonl_data = generator.generate_jsonl(dataset)
        generator.save_jsonl(jsonl_data, Path("data/training.jsonl"))
    """

    @staticmethod
    def generate_jsonl(dataset: DomainDataset) -> list[str]:
        """
        Generate JSONL format training data from a domain dataset.

        Each line is a JSON object with "system", "user", "assistant" fields
        compatible with Ollama fine-tuning.
        """
        training_examples = []

        # Example 1: Full workflow with all required steps
        required_ids = set(dataset.validation_rules.required_steps)
        required_steps = [s for s in dataset.steps if s.id in required_ids]

        if required_steps:
            full_workflow = FineTuneDataGenerator._build_workflow_json(
                dataset, required_ids
            )
            example1 = {
                "system": (
                    f"You are an expert {dataset.domain} workflow designer. "
                    "Generate accurate workflows based on the domain specification."
                ),
                "user": f"Generate a complete {dataset.display_name} workflow with all required steps. "
                        f"Domain: {dataset.domain}",
                "assistant": json.dumps(full_workflow),
            }
            training_examples.append(json.dumps(example1))

        # Example 2: Minimal workflow (start + end only)
        minimal_ids = {
            dataset.start_node,
            dataset.end_node,
        }
        minimal_workflow = FineTuneDataGenerator._build_workflow_json(
            dataset, minimal_ids
        )
        example2 = {
            "system": (
                f"You are an expert {dataset.domain} workflow designer. "
                "Generate accurate workflows based on the domain specification."
            ),
            "user": f"Generate a minimal {dataset.display_name} workflow. Domain: {dataset.domain}",
            "assistant": json.dumps(minimal_workflow),
        }
        training_examples.append(json.dumps(example2))

        # Example 3-5: Domain-specific scenarios
        scenarios = [
            ("successful", "happy path", dataset.validation_rules.required_steps),
            ("with decision", "include all decision nodes", None),
            ("with retries", "include retry steps", None),
        ]

        for scenario_name, description, step_ids in scenarios:
            if step_ids is None:
                step_ids = set(s.id for s in dataset.steps)

            workflow = FineTuneDataGenerator._build_workflow_json(dataset, step_ids)
            example = {
                "system": (
                    f"You are an expert {dataset.domain} workflow designer. "
                    "Generate accurate workflows based on the domain specification."
                ),
                "user": (
                    f"Generate a {dataset.display_name} workflow with {description}. "
                    f"Domain: {dataset.domain}"
                ),
                "assistant": json.dumps(workflow),
            }
            training_examples.append(json.dumps(example))

        return training_examples

    @staticmethod
    def _build_workflow_json(
        dataset: DomainDataset, selected_ids: set[str]
    ) -> dict:
        """Build workflow JSON for training data."""
        nodes = []
        for step in dataset.steps:
            if step.id in selected_ids:
                nodes.append({
                    "id": step.id,
                    "label": step.label,
                    "type": step.type,
                    "domain_step_id": step.id,
                })

        edges = []
        edge_set = set()
        for transition in dataset.transitions:
            if (transition.from_step in selected_ids and
                transition.to_step in selected_ids):
                edge_key = (transition.from_step, transition.to_step)
                if edge_key not in edge_set:
                    edges.append({
                        "source": transition.from_step,
                        "target": transition.to_step,
                        "condition": transition.condition,
                    })
                    edge_set.add(edge_key)

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def save_jsonl(data: list[str], output_path: Path) -> None:
        """Save training data to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for line in data:
                f.write(line + "\n")
        logger.info("Saved %d training examples to %s", len(data), output_path)

    @staticmethod
    def generate_for_all_datasets(
        datasets: list[DomainDataset],
        output_dir: Path,
    ) -> None:
        """Generate training data for all datasets."""
        all_examples = []

        for dataset in datasets:
            examples = FineTuneDataGenerator.generate_jsonl(dataset)
            all_examples.extend(examples)
            logger.info(
                "Generated %d examples for domain: %s",
                len(examples),
                dataset.domain,
            )

        # Save combined JSONL
        combined_path = output_dir / "all_domains_training.jsonl"
        FineTuneDataGenerator.save_jsonl(all_examples, combined_path)

        logger.info(
            "Generated %d total training examples across %d domains",
            len(all_examples),
            len(datasets),
        )
