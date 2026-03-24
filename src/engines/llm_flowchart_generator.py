"""
LLM flowchart generator using Ollama.

Uses a lightweight default model and retries with a smaller fallback model
when Ollama reports a low-memory error.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

from src.config import settings
from src.engines.rag_engine import RAGEngine
from src.models.workflow import EdgeStyle, GeneratedWorkflow, WorkflowEdge, WorkflowNode

if TYPE_CHECKING:
    from src.models.domain import DomainDataset
    from src.models.parser import ParsedInstruction

logger = logging.getLogger(__name__)


class LLMFlowchartGenerationError(Exception):
    """Raised when LLM flowchart generation fails."""


class LLMFlowchartGenerator:
    """LLM-based flowchart generator using Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._base_url = base_url or settings.ollama_base_url
        self._model = model or settings.ollama_model
        self._timeout = timeout or settings.ollama_timeout
        self._rag = RAGEngine()

    async def generate(
        self,
        dataset: DomainDataset,
        parsed: ParsedInstruction,
        include_optional: bool = True,
    ) -> GeneratedWorkflow:
        context = self._build_concise_context(dataset, parsed)
        prompt = self._build_prompt(
            instruction=parsed.original_text,
            context=context,
            domain=dataset.domain,
            has_decision_rules=bool(dataset.decision_rules),
            has_retry_constraints=bool(dataset.flowchart_retry_constraints),
            include_optional=include_optional,
        )

        try:
            response, used_model = await self._call_ollama(prompt)
        except Exception as exc:
            raise LLMFlowchartGenerationError(f"Ollama call failed: {exc}") from exc

        try:
            flowchart_data = self._parse_flowchart_json(response)
        except Exception as exc:
            raise LLMFlowchartGenerationError(
                f"Failed to parse LLM response: {exc}"
            ) from exc

        return self._build_flowchart(flowchart_data, dataset, parsed, used_model)

    @staticmethod
    def _build_concise_context(
        dataset: DomainDataset,
        parsed: ParsedInstruction,
    ) -> str:
        lines = [
            f"Domain: {dataset.display_name}",
            f"Description: {dataset.description}",
            "",
            "Available steps:",
        ]

        for step in dataset.steps:
            req = "[REQUIRED]" if step.required else "[optional]"
            lines.append(f"- {step.id}: {step.label} {req} ({step.type})")

        if dataset.decision_rules:
            lines.append("\nDecision nodes:")
            for node_id, rule in list(dataset.decision_rules.items())[:5]:
                lines.append(f"- {node_id}: {len(rule.branches)} branches")

        if dataset.flowchart_retry_constraints:
            lines.append("\nRetry allowed:")
            for constraint in dataset.flowchart_retry_constraints[:3]:
                lines.append(f"- {constraint.node} (max {constraint.max_attempts} attempts)")

        return "\n".join(lines)

    @staticmethod
    def _build_prompt(
        instruction: str,
        context: str,
        domain: str,
        has_decision_rules: bool,
        has_retry_constraints: bool,
        include_optional: bool,
    ) -> str:
        optional_text = (
            "Include all optional steps where relevant."
            if include_optional
            else "Include only required steps."
        )

        special_instructions = []
        if has_decision_rules:
            special_instructions.append(
                "- Include ALL decision nodes with their branches as specified in Decision Rules"
            )
        if has_retry_constraints:
            special_instructions.append(
                "- Include retry loops as specified in Retry Constraints (backward edges)"
            )

        special_text = "\n".join(special_instructions) if special_instructions else ""

        return f"""{context}

## User Instruction
"{instruction}"

{optional_text}

## Flowchart-Specific Requirements
{special_text}
- Decision nodes must have at least 2 branches
- Each branch must lead to a valid next step
- Retry edges should loop back to earlier steps
- All edges must follow Allowed Transitions or Decision Rules
- No isolated nodes - every node must be reachable from start

## Output Format
Return ONLY valid JSON (no markdown, no explanations):
{{
  "nodes": [
    {{"id": "step_id", "label": "Step Label", "type": "start|process|decision|end", "domain_step_id": "step_id"}}
  ],
  "edges": [
    {{"source": "step1", "target": "step2", "condition": null, "style": "normal"}}
  ]
}}

Generate now:
"""

    async def _call_ollama(self, prompt: str) -> tuple[str, str]:
        primary_model = self._model
        try:
            return await self._call_ollama_model(prompt, primary_model)
        except LLMFlowchartGenerationError as exc:
            if not self._is_memory_error(str(exc)):
                raise

            fallback_model = settings.ollama_fallback_model
            if fallback_model == primary_model:
                raise

            logger.warning(
                "Primary Ollama model '%s' is too heavy, retrying with '%s'",
                primary_model,
                fallback_model,
            )
            return await self._call_ollama_model(prompt, fallback_model)

    async def _call_ollama_model(self, prompt: str, model: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 1536,
                        },
                    },
                )

                if response.status_code != 200:
                    raise LLMFlowchartGenerationError(
                        f"Ollama returned {response.status_code}: {response.text[:500]}"
                    )

                data = response.json()
                return data.get("response", ""), model
        except httpx.ConnectError as exc:
            raise LLMFlowchartGenerationError(
                f"Cannot connect to Ollama at {self._base_url}"
            ) from exc

    @staticmethod
    def _is_memory_error(message: str) -> bool:
        lowered = message.lower()
        return "requires more system memory" in lowered or "not enough memory" in lowered

    @staticmethod
    def _parse_flowchart_json(response: str) -> dict[str, Any]:
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            match = re.search(r"\{[\s\S]*\}", response)
            if not match:
                raise ValueError("No JSON found in response")
            json_str = match.group()

        data = json.loads(json_str)

        if "nodes" not in data or "edges" not in data:
            raise ValueError("Missing required keys")

        return data

    @staticmethod
    def _build_flowchart(
        data: dict[str, Any],
        dataset: DomainDataset,
        parsed: ParsedInstruction,
        used_model: str,
    ) -> GeneratedWorkflow:
        nodes: list[WorkflowNode] = []
        node_ids: set[str] = set()

        for node_data in data.get("nodes", []):
            node = WorkflowNode(
                id=str(node_data.get("id", "")).strip(),
                label=str(node_data.get("label", "")).strip(),
                type=node_data.get("type", "process"),
                domain_step_id=str(node_data.get("domain_step_id", "")).strip(),
                metadata=node_data.get("metadata", {}),
            )
            if node.id:
                nodes.append(node)
                node_ids.add(node.id)

        edges: list[WorkflowEdge] = []
        edge_ids: set[str] = set()

        for edge_data in data.get("edges", []):
            source = str(edge_data.get("source", "")).strip()
            target = str(edge_data.get("target", "")).strip()

            if source not in node_ids or target not in node_ids:
                logger.warning("Skipping edge %s -> %s", source, target)
                continue

            condition_data = edge_data.get("condition")
            condition = None
            if condition_data and isinstance(condition_data, dict):
                condition = {
                    "label": str(condition_data.get("label", "")),
                    "branch_key": condition_data.get("branch_key"),
                }

            style_str = edge_data.get("style", "normal")
            style = EdgeStyle.RETRY_LOOP if style_str == "retry_loop" else EdgeStyle.NORMAL

            edge_id = f"e_{source}__{target}"
            if condition_data:
                edge_id += f"__{condition_data.get('branch_key', '')}"

            if edge_id not in edge_ids:
                edges.append(
                    WorkflowEdge(
                        id=edge_id,
                        source=source,
                        target=target,
                        condition=condition,
                        style=style,
                    )
                )
                edge_ids.add(edge_id)

        import hashlib

        workflow_id = hashlib.sha256(
            f"{dataset.domain}_{parsed.cleaned_text}_flowchart".encode()
        ).hexdigest()[:16]

        return GeneratedWorkflow(
            workflow_id=f"llm_fc_{workflow_id}",
            domain=dataset.domain,
            title=f"{dataset.display_name} Flowchart (LLM-Generated)",
            description=dataset.description,
            is_flowchart=True,
            nodes=nodes,
            edges=edges,
            metadata={
                "generator": "llm_flowchart_generator",
                "llm_model": used_model,
                "dataset_version": dataset.version,
            },
        )
