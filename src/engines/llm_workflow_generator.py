"""
LLM Workflow Generator — trained LLM-based workflow synthesis.

Replaces the deterministic WorkflowGenerator with an LLM-powered approach:
  - Ollama local LLM (fine-tuned on dataset examples)
  - RAG context for domain-specific knowledge
  - Structured JSON output parsing
  - No external API dependencies
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

from src.config import settings
from src.engines.rag_engine import RAGEngine
from src.models.workflow import GeneratedWorkflow, WorkflowEdge, WorkflowNode

if TYPE_CHECKING:
    from src.models.domain import DomainDataset
    from src.models.parser import ParsedInstruction

logger = logging.getLogger(__name__)


class LLMWorkflowGenerationError(Exception):
    """Raised when LLM workflow generation fails."""


class LLMWorkflowGenerator:
    """
    LLM-based workflow generator using Ollama.

    Lifecycle:
        generator = LLMWorkflowGenerator()
        workflow = await generator.generate(dataset, parsed_instruction)
    """

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
        """
        Generate workflow using finetuned Ollama LLM with RAG context (optimized).

        Returns a GeneratedWorkflow with nodes and edges from LLM output.
        """
        # Build concise RAG context (summarized for performance)
        rag_context = self._build_concise_context(dataset, parsed)
        
        # Build prompt with RAG context
        prompt = f"""{rag_context}

User instruction: "{parsed.original_text}"

Generate a valid JSON workflow. Output ONLY the JSON object, no other text.

JSON format (REQUIRED):
{{"nodes": [{{"id": "start", "label": "Start", "type": "start", "domain_step_id": ""}}], "edges": []}}

JSON:
"""

        # Call Ollama
        try:
            response = await self._call_ollama(prompt)
        except Exception as exc:
            raise LLMWorkflowGenerationError(f"Ollama call failed: {exc}") from exc

        # Parse workflow JSON from response
        try:
            workflow_data = self._parse_workflow_json(response)
        except Exception as exc:
            raise LLMWorkflowGenerationError(
                f"Failed to parse LLM response: {exc}\n\nRaw response:\n{response}"
            ) from exc

        # Build GeneratedWorkflow object
        workflow = self._build_workflow(
            workflow_data, dataset, parsed
        )

        return workflow

    # ------------------------------------------------------------------
    # Context building (optimized for RAG)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_concise_context(
        dataset: DomainDataset,
        parsed: ParsedInstruction,
    ) -> str:
        """Build concise RAG context for fast LLM inference."""
        lines = [
            f"Domain: {dataset.display_name}",
            f"Description: {dataset.description}",
            "",
            "Available steps:",
        ]

        # List all steps concisely
        for step in dataset.steps:
            req = "[REQUIRED]" if step.required else "[optional]"
            lines.append(f"- {step.id}: {step.label} {req} ({step.type})")

        # Required steps summary
        required = [s for s in dataset.steps if s.required]
        if required:
            lines.append(f"\nMust include steps: {', '.join([s.id for s in required])}")

        # Key transitions
        lines.append("\nKey transitions:")
        for t in dataset.transitions[:10]:  # First 10 transitions only
            lines.append(f"- {t.from_step} → {t.to_step}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        instruction: str,
        context: str,
        domain: str,
        include_optional: bool,
    ) -> str:
        """Build structured prompt for the LLM."""
        return f"""{context}

## Instruction
{instruction}

## Output
Return ONLY this JSON format (no markdown):
{{"nodes": [{{"id": "1", "label": "Start", "type": "start", "domain_step_id": ""}}], "edges": []}}
"""

    # ------------------------------------------------------------------
    # Ollama interaction
    # ------------------------------------------------------------------

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and get response."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Low temp for determinism
                            "top_p": 0.9,
                            "num_predict": 2048,  # Allow complete JSON generation with RAG context
                        },
                    },
                )
                
                # Log response status
                logger.debug(f"Ollama response status: {response.status_code}")
                
                # Check for HTTP errors
                if response.status_code != 200:
                    error_text = response.text[:500]
                    raise LLMWorkflowGenerationError(
                        f"Ollama returned {response.status_code}: {error_text}"
                    )
                
                data = response.json()
                return data.get("response", "")
        except httpx.ConnectError as exc:
            raise LLMWorkflowGenerationError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Ensure Ollama is running: ollama serve"
            ) from exc
        except LLMWorkflowGenerationError:
            raise
        except Exception as exc:
            raise LLMWorkflowGenerationError(
                f"Ollama API error: {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_workflow_json(response: str) -> dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        # Find JSON block (might be wrapped in markdown code fence)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            match = re.search(r"\{[\s\S]*\}", response)
            if not match:
                raise ValueError("No JSON found in response")
            json_str = match.group()

        # Try to parse, but if incomplete, try to auto-complete
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix incomplete JSON by closing open brackets
            if '{"nodes"' in json_str or '"nodes"' in json_str:
                # Count opening and closing braces/brackets
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                open_brackets = json_str.count('[')
                close_brackets = json_str.count(']')
                
                # Auto-complete
                json_str += ']' * (open_brackets - close_brackets)
                json_str += '}' * (open_braces - close_braces)
                
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # If still failing, return empty workflow structure
                    logger.warning(f"Could not parse JSON, returning empty workflow: {e}")
                    data = {"nodes": [], "edges": []}
            else:
                raise ValueError(f"Invalid JSON: {e}")

        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Root must be an object")
        
        # Handle nested structure: if LLM wrapped output in "workflows", extract it
        if "workflows" in data and "nodes" not in data:
            workflows = data.get("workflows", {})
            if isinstance(workflows, dict) and workflows:
                # Get the first workflow
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

    # ------------------------------------------------------------------
    # Workflow construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_workflow(
        data: dict[str, Any],
        dataset: DomainDataset,
        parsed: ParsedInstruction,
    ) -> GeneratedWorkflow:
        """Convert LLM output to GeneratedWorkflow."""
        nodes: list[WorkflowNode] = []
        node_ids: set[str] = set()

        # Build nodes
        for node_data in data.get("nodes", []):
            # Normalize node data: fix common field name typos
            if "idiname" in node_data and "id" not in node_data:
                node_data["id"] = node_data["idiname"]
            if "name" in node_data and "label" not in node_data and not node_data.get("label"):
                node_data["label"] = node_data["name"]
            
            # Normalize node type
            node_type = str(node_data.get("type", "process")).strip().lower()
            # Map common aliases to valid types
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

        # Build edges (only for nodes that exist)
        edges: list[WorkflowEdge] = []
        edge_ids: set[str] = set()

        for edge_data in data.get("edges", []):
            source = str(edge_data.get("source", "")).strip()
            target = str(edge_data.get("target", "")).strip()

            if source not in node_ids or target not in node_ids:
                logger.warning(
                    "Skipping edge %s → %s (node not in workflow)",
                    source, target,
                )
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

        # Generate workflow ID
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
                "llm_model": settings.ollama_model,
                "dataset_version": dataset.version,
            },
        )
