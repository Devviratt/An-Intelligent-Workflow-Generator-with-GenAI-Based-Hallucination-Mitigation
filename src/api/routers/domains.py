"""
Domains Router — list and inspect available domain datasets.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.models.request import DomainInfo, DomainListResponse

router = APIRouter(tags=["Domains"])


@router.get(
    "/domains",
    response_model=DomainListResponse,
    summary="List available domains",
    description=(
        "Return every loaded domain dataset with its display name, "
        "description, top keywords, and step/transition counts."
    ),
)
async def list_domains() -> DomainListResponse:
    from src.api.server import get_pipeline

    pipeline = get_pipeline()
    engine = pipeline.dataset_engine
    domains: list[DomainInfo] = []
    for ds in engine.all_datasets():
        domains.append(
            DomainInfo(
                domain=ds.domain,
                display_name=ds.display_name,
                description=ds.description,
                keywords=ds.keywords[:10],
                step_count=len(ds.steps),
                transition_count=len(ds.transitions),
            )
        )
    return DomainListResponse(domains=domains, count=len(domains))


@router.get(
    "/domains/{domain}",
    summary="Get domain detail",
    description="Return the full JSON dataset definition for a single domain.",
)
async def get_domain(domain: str) -> dict:
    from src.api.server import get_pipeline

    pipeline = get_pipeline()
    ds = pipeline.dataset_engine.get(domain)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")
    return ds.model_dump(by_alias=True)
