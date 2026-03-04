"""
RAG Engine — Retrieval-Augmented Generation for dataset context.

Retrieves relevant dataset steps, transitions, and examples
to provide context to the LLM for accurate workflow generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.models.parser import ParsedInstruction

if TYPE_CHECKING:
    from src.models.domain import DomainDataset

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieves relevant workflow context from domain datasets.

    Usage:
        rag = RAGEngine()
        context = rag.build_context(dataset, parsed_instruction)
    """

    def build_context(
        self,
        dataset: DomainDataset,
        parsed: ParsedInstruction,
    ) -> str:
        """
        Build comprehensive structured context for the LLM.

        Returns a formatted string with:
        - Domain description
        - All available steps with types and descriptions
        - Allowed transitions
        - Decision node rules
        - Example workflow structure
        """
        lines = [
            f"# Domain: {dataset.display_name}",
            f"Description: {dataset.description}",
            f"SLA: {dataset.metadata.sla_seconds}s, Compliance: {', '.join(dataset.metadata.compliance)}",
            "",
            "## Available Steps",
            "Each step is identified by its ID and can be included in the workflow.",
        ]

        # Group steps by type
        start_steps = [s for s in dataset.steps if s.type == "start"]
        end_steps = [s for s in dataset.steps if s.type == "end"]
        process_steps = [s for s in dataset.steps if s.type == "process"]
        decision_steps = [s for s in dataset.steps if s.type == "decision"]

        if start_steps:
            lines.append("\n### Start Nodes (workflow entry points)")
            for step in start_steps:
                lines.append(f"- **{step.id}**: {step.label}")
                if step.description:
                    lines.append(f"  {step.description}")

        if process_steps:
            lines.append("\n### Process Nodes (processing/actions)")
            for step in process_steps:
                req = " (required)" if step.required else " (optional)"
                lines.append(f"- **{step.id}**: {step.label}{req}")
                if step.description:
                    lines.append(f"  {step.description}")

        if decision_steps:
            lines.append("\n### Decision Nodes (conditional branches)")
            for step in decision_steps:
                lines.append(f"- **{step.id}**: {step.label}")
                if step.description:
                    lines.append(f"  {step.description}")
                if step.branches:
                    lines.append("  Possible branches:")
                    for branch_key, branch_label in step.branches.items():
                        lines.append(f"    - {branch_key}: {branch_label}")

        if end_steps:
            lines.append("\n### End Nodes (workflow termination points)")
            for step in end_steps:
                lines.append(f"- **{step.id}**: {step.label}")
                if step.description:
                    lines.append(f"  {step.description}")

        # Allowed transitions
        lines.append("\n## Allowed Transitions")
        lines.append("Valid step-to-step paths in this domain:")
        for transition in dataset.transitions:
            cond = f" [{transition.condition}]" if transition.condition else ""
            lines.append(
                f"- {transition.from_step} → {transition.to_step}{cond}"
            )

        # Decision rules (if flowchart)
        if dataset.decision_rules:
            lines.append("\n## Decision Rules (for flowcharts)")
            for node_id, rule in dataset.decision_rules.items():
                lines.append(f"\n### {node_id}")
                for branch in rule.branches:
                    lines.append(f"- **{branch.label}** → {branch.target}")

        # Retry constraints
        if dataset.flowchart_retry_constraints:
            lines.append("\n## Retry Constraints")
            for rc in dataset.flowchart_retry_constraints:
                lines.append(
                    f"- {rc.node} can retry up to {rc.max_attempts} times → {rc.loop_back_to}"
                )

        # Validation rules
        if dataset.validation_rules.required_steps:
            lines.append("\n## Required Steps (must be in every workflow)")
            for step_id in dataset.validation_rules.required_steps:
                lines.append(f"- {step_id}")

        if dataset.validation_rules.forbidden_direct_transitions:
            lines.append("\n## Forbidden Transitions (never allowed)")
            for ft in dataset.validation_rules.forbidden_direct_transitions:
                lines.append(f"- ❌ {ft.from_step} → {ft.to_step}")

        # Instruction context
        if parsed.keywords.cleaned_tokens:
            lines.append(f"\n## User Intent Keywords")
            lines.append(f"Keywords: {', '.join(parsed.keywords.cleaned_tokens[:15])}")

        if parsed.intent_flags:
            lines.append(f"\n## Detected Intents")
            for flag, value in parsed.intent_flags.items():
                if value:
                    lines.append(f"- {flag}: YES")

        lines.append("\n## Task")
        lines.append(
            "Generate a valid workflow JSON based on the instruction and allowed transitions."
        )

        return "\n".join(lines)

    def build_training_examples(
        self,
        dataset: DomainDataset,
    ) -> list[dict[str, str]]:
        """
        Build training examples from a domain dataset.

        Returns list of (instruction, workflow_json) pairs for fine-tuning.
        """
        examples = []

        # Example 1: Basic workflow covering all required steps
        required_ids = set(dataset.validation_rules.required_steps)
        basic_workflow = {
            "nodes": [
                {
                    "id": s.id,
                    "label": s.label,
                    "type": s.type,
                    "domain_step_id": s.id,
                }
                for s in dataset.steps
                if s.id in required_ids or s.type in ("start", "end")
            ],
            "edges": [
                {
                    "source": t.from_step,
                    "target": t.to_step,
                    "condition": t.condition,
                }
                for t in dataset.transitions
                if (t.from_step in required_ids or t.from_step in {s.id for s in dataset.steps if s.type == "start"})
                and (t.to_step in required_ids or t.to_step in {s.id for s in dataset.steps if s.type in ("end", "process")})
            ],
        }

        examples.append({
            "instruction": f"Generate a complete {dataset.domain} workflow including all required steps",
            "workflow": str(basic_workflow),
        })

        return examples
