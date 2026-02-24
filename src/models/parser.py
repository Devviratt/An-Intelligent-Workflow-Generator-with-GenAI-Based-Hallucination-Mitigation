"""Instruction parser output models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedKeywords(BaseModel):
    """Keywords extracted from user instruction."""

    raw_tokens: list[str] = Field(default_factory=list)
    cleaned_tokens: list[str] = Field(default_factory=list)
    tfidf_top_terms: list[tuple[str, float]] = Field(default_factory=list)


class DomainMatch(BaseModel):
    """A candidate domain match with confidence score."""

    domain: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str] = Field(default_factory=list)


class ParsedInstruction(BaseModel):
    """Result of parsing a user instruction."""

    original_text: str
    cleaned_text: str
    keywords: ExtractedKeywords
    domain_matches: list[DomainMatch] = Field(default_factory=list)
    selected_domain: str | None = None
    custom_steps_requested: list[str] = Field(default_factory=list)
    intent_flags: dict[str, bool] = Field(default_factory=dict)

    @property
    def best_match(self) -> DomainMatch | None:
        if not self.domain_matches:
            return None
        return max(self.domain_matches, key=lambda m: m.confidence)

    @property
    def has_confident_match(self) -> bool:
        best = self.best_match
        return best is not None and best.confidence >= 0.15
