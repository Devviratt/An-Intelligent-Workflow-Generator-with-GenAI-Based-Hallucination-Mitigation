"""
LLM workflow generator using Ollama.

This version defaults to a lightweight model and automatically retries with a
fallback model when Ollama reports a low-memory error.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

from src.config import settings
from src.engines.llm_post_processor import LLMPostProcessor
from src.engines.rag_engine import RAGEngine
from src.models.workflow import GeneratedWorkflow, WorkflowEdge, WorkflowNode

if TYPE_CHECKING:
    from src.models.domain import DomainDataset
    from src.models.parser import ParsedInstruction

logger = logging.getLogger(__name__)


class LLMWorkflowGenerationError(Exception):
    """Raised when LLM workflow generation fails."""


class LLMWorkflowGenerator:
    """LLM-based workflow generator using Ollama."""

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
        self._post_processor = LLMPostProcessor()

    async def generate(
        self,
        dataset: DomainDataset,
        parsed: ParsedInstruction,
        include_optional: bool = True,
    ) -> GeneratedWorkflow:
        """Generate workflow using Ollama with RAG-grounded dataset context."""
        # Retrieve grounded context from RAG engine using dataset knowledge
        rag_context = self._rag.build_context(dataset, parsed)

        prompt = f"""You are a workflow generation assistant. Use ONLY the provided context to generate valid workflows.

DATASET CONTEXT (from domain knowledge):
{rag_context}

USER INSTRUCTION:
"{parsed.original_text}"

CRITICAL REQUIREMENTS:
1. NODES: Use ONLY steps from the context above
2. EDGES: Create edges to connect ALL nodes in a continuous path
   - Start node MUST have at least one outgoing edge
   - Every process/decision node MUST have incoming AND outgoing edges
   - End node MUST have at least one incoming edge
3. NO ISOLATED NODES: Every node must be reachable from start node
4. FOR DECISION NODES: Include "branches" with conditional labels
5. TRANSITIONS: All edges must follow allowed transitions in the dataset

OUTPUT FORMAT (REQUIRED - VALID JSON ONLY):
{{
  "nodes": [
    {{"id": "start", "label": "Start", "type": "start", "domain_step_id": "", "description": ""}},
    {{"id": "process_1", "label": "Process", "type": "process", "domain_step_id": "step_id", "description": ""}},
    {{"id": "end", "label": "End", "type": "end", "domain_step_id": "", "description": ""}}
  ],
  "edges": [
    {{"source": "start", "target": "process_1", "condition": null}},
    {{"source": "process_1", "target": "end", "condition": null}}
  ]
}}

Generate valid JSON with complete node-to-node connections:
"""

        try:
            response, used_model = await self._call_ollama(prompt)
        except Exception as exc:
            raise LLMWorkflowGenerationError(f"Ollama call failed: {exc}") from exc

        try:
            workflow_data = self._parse_workflow_json(response)
        except Exception as exc:
            raise LLMWorkflowGenerationError(
                f"Failed to parse LLM response: {exc}\n\nRaw response:\n{response}"
            ) from exc

        workflow = self._build_workflow(workflow_data, dataset, parsed, used_model)
        workflow = self._post_processor.process(workflow, dataset)
        return workflow

    async def _call_ollama(self, prompt: str) -> tuple[str, str]:
        """Call Ollama and retry with a smaller model on low-memory errors."""
        primary_model = self._model
        try:
            return await self._call_ollama_model(prompt, primary_model)
        except LLMWorkflowGenerationError as exc:
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
                    error_text = response.text[:500]
                    raise LLMWorkflowGenerationError(
                        f"Ollama returned {response.status_code}: {error_text}"
                    )

                data = response.json()
                return data.get("response", ""), model
        except httpx.ConnectError as exc:
            raise LLMWorkflowGenerationError(
                f"Cannot connect to Ollama at {self._base_url}. Ensure Ollama is running."
            ) from exc
        except LLMWorkflowGenerationError:
            raise
        except Exception as exc:
            raise LLMWorkflowGenerationError(
                f"Ollama API error: {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc

    @staticmethod
    def _is_memory_error(message: str) -> bool:
        lowered = message.lower()
        return "requires more system memory" in lowered or "not enough memory" in lowered

    @staticmethod
    def _parse_workflow_json(response: str) -> dict[str, Any]:
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            match = re.search(r"\{[\s\S]*\}", response)
            if not match:
                raise ValueError("No JSON found in response")
            json_str = match.group()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            if '{"nodes"' in json_str or '"nodes"' in json_str:
                open_braces = json_str.count("{")
                close_braces = json_str.count("}")
                open_brackets = json_str.count("[")
                close_brackets = json_str.count("]")
                json_str += "]" * (open_brackets - close_brackets)
                json_str += "}" * (open_braces - close_braces)
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON, returning empty workflow: %s", exc)
                    data = {"nodes": [], "edges": []}
            else:
                raise ValueError(f"Invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Root must be an object")

        if "workflows" in data and "nodes" not in data:
            workflows = data.get("workflows", {})
            if isinstance(workflows, dict) and workflows:
                first_workflow = next(iter(workflows.values()))
                if isinstance(first_workflow, dict) and "nodes" in first_workflow:
                    data = first_workflow

        if "nodes" not in data or "edges" not in data:
            raise ValueError("Missing 'nodes' or 'edges' key")
        if not isinstance(data["nodes"], list):
            raise ValueError("'nodes' must be an array")
        if not isinstance(data["edges"], list):
            raise ValueError("'edges' must be an array")

        return data

    @staticmethod
    def _build_workflow(
        data: dict[str, Any],
        dataset: DomainDataset,
        parsed: ParsedInstruction,
        used_model: str,
    ) -> GeneratedWorkflow:
        nodes: list[WorkflowNode] = []
        node_ids: set[str] = set()

        for node_data in data.get("nodes", []):
            if "idiname" in node_data and "id" not in node_data:
                node_data["id"] = node_data["idiname"]
            if "name" in node_data and "label" not in node_data and not node_data.get("label"):
                node_data["label"] = node_data["name"]

            node_type = str(node_data.get("type", "process")).strip().lower()
            type_map = {
                "start": "start",
                "end": "end",
                "stop": "end",
                "process": "process",
                "task": "process",
                "action": "process",
                "step": "process",
                "decision": "decision",
                "branch": "decision",
                "choice": "decision",
            }
            node_type = type_map.get(node_type, "process")

            node = WorkflowNode(
                id=str(node_data.get("id", "")).strip(),
                label=str(node_data.get("label", "")).strip(),
                type=node_type,
                description=str(node_data.get("description", "")).strip(),
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
                logger.warning("Skipping edge %s -> %s (node not in workflow)", source, target)
                continue

            condition_data = edge_data.get("condition")
            condition = None
            if condition_data:
                condition = {
                    "label": str(condition_data.get("label", "")),
                    "branch_key": condition_data.get("branch_key"),
                }

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
                    )
                )
                edge_ids.add(edge_id)

        import hashlib

        workflow_id = hashlib.sha256(
            f"{dataset.domain}_{parsed.cleaned_text}".encode()
        ).hexdigest()[:16]

        return GeneratedWorkflow(
            workflow_id=f"llm_{workflow_id}",
            domain=dataset.domain,
            title=f"{dataset.display_name} Workflow (LLM-Generated)",
            description=dataset.description,
            nodes=nodes,
            edges=edges,
            metadata={
                "generator": "llm_workflow_generator",
                "llm_model": used_model,
                "dataset_version": dataset.version,
            },
        )
