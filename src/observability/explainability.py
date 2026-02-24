"""
Explainability Engine — post-generation provenance annotation.

Attaches per-node lineage metadata and generation rationale to the
workflow WITHOUT modifying any core generation logic.  Operates purely
as a read-only inspector of the finished workflow + dataset.
"""

from __future__ import annotations

from src.models.domain import DomainDataset
from src.models.parser import ParsedInstruction
from src.models.workflow import EdgeStyle, GeneratedWorkflow, NodeType
from src.observability.models import ExplainabilityEntry, NodeProvenance


class ExplainabilityEngine:
    """
    Build explainability metadata for a generated workflow.

    Usage::

        engine = ExplainabilityEngine()
        entry = engine.build(workflow, dataset, parsed, mode="workflow")
    """

    def build(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
        parsed: ParsedInstruction,
        *,
        mode: str = "workflow",
    ) -> ExplainabilityEntry:
        """
        Produce a complete ExplainabilityEntry for the workflow.

        This is a pure read — no mutations to workflow or dataset.
        """
        step_map = dataset.step_map
        valid_ids = dataset.step_ids

        # Per-node provenance
        provenance: list[NodeProvenance] = []
        for node in workflow.nodes:
            ds_step = step_map.get(node.domain_step_id)

            # Determine source origin
            if node.domain_step_id in valid_ids:
                source = "dataset"
                grounding = "grounded"
            else:
                source = "inferred"
                grounding = "unverified"

            # Count outgoing edges
            outgoing = sum(1 for e in workflow.edges if e.source == node.id)

            provenance.append(
                NodeProvenance(
                    node_id=node.id,
                    domain_step_id=node.domain_step_id,
                    source=source,
                    dataset_domain=dataset.domain,
                    dataset_version=dataset.version,
                    step_required=ds_step.required if ds_step else False,
                    step_type=ds_step.type if ds_step else node.type.value,
                    grounding_status=grounding,
                    transition_count=outgoing,
                )
            )

        # Aggregate counts
        decision_count = sum(
            1 for n in workflow.nodes if n.type == NodeType.DECISION
        )
        retry_count = sum(
            1 for e in workflow.edges if e.style == EdgeStyle.RETRY_LOOP
        )
        steps_pruned = len(dataset.steps) - len(workflow.nodes)

        # Instruction keywords
        keywords = (
            parsed.keywords.cleaned_tokens[:20]
            if parsed.keywords.cleaned_tokens
            else parsed.keywords.raw_tokens[:20]
        )

        return ExplainabilityEntry(
            node_provenance=provenance,
            generation_mode=mode,
            domain_used=dataset.domain,
            dataset_version=dataset.version,
            instruction_keywords=keywords,
            domain_confidence=(
                parsed.best_match.confidence if parsed.best_match else 0.0
            ),
            total_dataset_steps=len(dataset.steps),
            steps_selected=len(workflow.nodes),
            steps_pruned=max(0, steps_pruned),
            decision_nodes_count=decision_count,
            retry_edges_count=retry_count,
        )

    def annotate_nodes(
        self,
        workflow: GeneratedWorkflow,
        dataset: DomainDataset,
    ) -> GeneratedWorkflow:
        """
        Attach provenance metadata directly to each node's metadata dict.

        Non-destructive: adds an ``_explainability`` key to node.metadata.
        """
        step_map = dataset.step_map

        for node in workflow.nodes:
            ds_step = step_map.get(node.domain_step_id)
            node.metadata["_explainability"] = {
                "source": "dataset" if node.domain_step_id in dataset.step_ids else "inferred",
                "grounding_status": "grounded" if node.domain_step_id in dataset.step_ids else "unverified",
                "step_required": ds_step.required if ds_step else False,
                "dataset_domain": dataset.domain,
                "dataset_version": dataset.version,
            }

        return workflow
