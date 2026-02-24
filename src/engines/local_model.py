"""
Optional Local Model Integration — RAG with Ollama.

Architecture:
  1. Retrieve relevant domain dataset as context
  2. Construct constrained prompt with strict output format
  3. Call local model (Ollama)
  4. Parse response as JSON
  5. Validate parsed output against domain dataset
  6. Reject output that fails validation — fall back to dataset-only generation

IMPORTANT: This module is OPTIONAL. The system works fully without it.
The local model is only used for natural-language refinement of step
descriptions, NOT for generating workflow structure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.config import settings
from src.models.domain import DomainDataset
from src.models.workflow import GeneratedWorkflow

logger = logging.getLogger(__name__)


class LocalModelError(Exception):
    """Raised when local model interaction fails."""


class LocalModelIntegration:
    """
    RAG-based local model integration for optional description refinement.

    The model is NEVER used to generate structure. It can only refine
    human-readable labels and descriptions for nodes that already exist
    in the validated workflow.
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

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self._model in m for m in models)
        except Exception:
            return False

    async def refine_descriptions(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """
        Use local model to refine node descriptions via RAG.

        The workflow structure is NEVER modified — only descriptions.
        If the model fails or returns invalid output, the original
        workflow is returned unchanged.
        """
        if not settings.use_local_model:
            return workflow

        try:
            available = await self.is_available()
            if not available:
                logger.warning("Local model not available — skipping refinement")
                return workflow

            context = self._build_context(dataset)
            prompt = self._build_prompt(workflow, context)
            response = await self._call_model(prompt)
            refined = self._parse_and_validate(response, workflow, dataset)
            return refined

        except Exception as exc:
            logger.error("Local model refinement failed: %s", exc)
            return workflow  # Safe fallback

    # ------------------------------------------------------------------
    # RAG context building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(dataset: DomainDataset) -> str:
        """Build structured context from the domain dataset for RAG."""
        lines = [
            f"Domain: {dataset.display_name}",
            f"Description: {dataset.description}",
            f"Compliance: {', '.join(dataset.metadata.compliance)}",
            "",
            "Available steps:",
        ]
        for step in dataset.steps:
            lines.append(f"  - {step.id}: {step.label} ({step.type}) — {step.description}")

        return "\n".join(lines)

    @staticmethod
    def _build_prompt(workflow: GeneratedWorkflow, context: str) -> str:
        """Build constrained prompt for local model."""
        node_list = json.dumps(
            [
                {"id": n.id, "label": n.label, "description": n.description}
                for n in workflow.nodes
            ],
            indent=2,
        )

        return f"""You are a workflow documentation assistant. Your task is to improve
the descriptions of workflow nodes. You must NOT add, remove, or rename any nodes.
Only improve the human-readable descriptions.

DOMAIN CONTEXT:
{context}

CURRENT NODES:
{node_list}

Return a JSON array with the same node IDs and improved descriptions.
Format: [{{"id": "node_id", "description": "improved description"}}]

RULES:
- Keep every node ID exactly as-is
- Only change the "description" field
- Keep descriptions concise (under 100 characters)
- Descriptions must be factually grounded in the domain context above
- Do NOT invent capabilities not mentioned in the context
"""

    # ------------------------------------------------------------------
    # Model interaction
    # ------------------------------------------------------------------

    async def _call_model(self, prompt: str) -> str:
        """Call Ollama API."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for determinism
                        "top_p": 0.9,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    # ------------------------------------------------------------------
    # Output validation (strict gate)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_and_validate(
        response: str,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """
        Parse model output and validate against the workflow.

        Only description changes are accepted. Any structural change
        causes full rejection.
        """
        try:
            # Extract JSON from response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start == -1 or end == 0:
                logger.warning("Model response contains no JSON array")
                return workflow

            raw_json = response[start:end]
            refinements: list[dict[str, Any]] = json.loads(raw_json)

        except json.JSONDecodeError:
            logger.warning("Model response is not valid JSON — rejecting")
            return workflow

        # Build lookup
        node_map = workflow.node_map
        valid_ids = {n.id for n in workflow.nodes}

        applied = 0
        for item in refinements:
            nid = item.get("id", "")
            desc = item.get("description", "")

            if nid not in valid_ids:
                logger.warning("Model returned unknown node '%s' — skipping", nid)
                continue

            if not isinstance(desc, str) or len(desc) > 200:
                continue

            node_map[nid].description = desc
            applied += 1

        logger.info("Applied %d description refinements from local model", applied)
        return workflow
