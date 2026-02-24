"""
Instruction Parser — NLP-based intent extraction without external AI APIs.

Pipeline:
  1. Text normalisation (lowercase, punctuation strip, stopword removal)
  2. Keyword extraction (n-gram tokenisation)
  3. TF-IDF vectorisation against domain keyword corpus
  4. Cosine-similarity domain matching
  5. Simple heuristic intent-flag extraction
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import TYPE_CHECKING, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings
from src.models.parser import DomainMatch, ExtractedKeywords, ParsedInstruction

if TYPE_CHECKING:
    from src.models.domain import DomainDataset

# ---------------------------------------------------------------------------
# Stopwords — lightweight built-in set (avoids NLTK dependency)
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "about", "up", "it",
        "its", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "they", "them", "their", "theirs",
        "themselves", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "that's", "want", "please",
    }
)

# Intent-detection patterns
_INTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "include_retry": re.compile(r"\bretry|retries|re-?attempt\b", re.IGNORECASE),
    "include_error_handling": re.compile(r"\berror|fail|exception|fallback\b", re.IGNORECASE),
    "minimal": re.compile(r"\bminimal|simple|basic|short\b", re.IGNORECASE),
    "detailed": re.compile(r"\bdetailed|full|complete|comprehensive\b", re.IGNORECASE),
    "include_notifications": re.compile(r"\bnotif|email|sms|alert|message\b", re.IGNORECASE),
}


class InstructionParser:
    """
    Stateful parser that scores user instructions against loaded domains.

    Lifecycle:
        parser = InstructionParser()
        parser.fit(domain_datasets)      # build TF-IDF model
        result = parser.parse("...")     # parse instruction
    """

    def __init__(self) -> None:
        self._vectoriser: TfidfVectorizer | None = None
        self._domain_vectors: np.ndarray | None = None
        self._domain_names: list[str] = []
        self._domain_keyword_sets: dict[str, set[str]] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, datasets: Sequence[DomainDataset]) -> None:
        """Build TF-IDF model from domain keyword lists."""
        if not datasets:
            raise ValueError("Cannot fit parser with zero datasets")

        self._domain_names = []
        corpus: list[str] = []

        for ds in datasets:
            self._domain_names.append(ds.domain)
            self._domain_keyword_sets[ds.domain] = {k.lower() for k in ds.keywords}
            # Build a pseudo-document per domain from its keywords + step labels
            doc_tokens = list(ds.keywords)
            doc_tokens.extend(s.label for s in ds.steps)
            doc_tokens.extend(s.id.replace("_", " ") for s in ds.steps)
            doc_tokens.append(ds.display_name)
            doc_tokens.append(ds.description)
            corpus.append(" ".join(doc_tokens))

        self._vectoriser = TfidfVectorizer(
            max_features=settings.tfidf_max_features,
            ngram_range=(1, 2),
            stop_words=list(_STOPWORDS),
            lowercase=True,
        )
        self._domain_vectors = self._vectoriser.fit_transform(corpus).toarray()
        self._fitted = True

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse(self, instruction: str) -> ParsedInstruction:
        """Parse a user instruction into a structured result."""
        if not self._fitted:
            raise RuntimeError("InstructionParser.fit() must be called before parse()")

        cleaned = self._normalise(instruction)
        keywords = self._extract_keywords(instruction, cleaned)
        domain_matches = self._match_domains(cleaned)
        intent_flags = self._detect_intents(instruction)

        selected = None
        if domain_matches:
            best = max(domain_matches, key=lambda m: m.confidence)
            if best.confidence >= settings.keyword_match_threshold:
                selected = best.domain

        return ParsedInstruction(
            original_text=instruction,
            cleaned_text=cleaned,
            keywords=keywords,
            domain_matches=domain_matches,
            selected_domain=selected,
            intent_flags=intent_flags,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _extract_keywords(raw: str, cleaned: str) -> ExtractedKeywords:
        raw_tokens = cleaned.split()
        cleaned_tokens = [t for t in raw_tokens if t not in _STOPWORDS and len(t) > 2]
        return ExtractedKeywords(
            raw_tokens=raw_tokens,
            cleaned_tokens=cleaned_tokens,
            tfidf_top_terms=[],  # filled by domain match phase
        )

    def _match_domains(self, cleaned_text: str) -> list[DomainMatch]:
        """Score the instruction against every domain using hybrid TF-IDF + keyword overlap."""
        assert self._vectoriser is not None
        assert self._domain_vectors is not None

        # TF-IDF cosine similarity
        query_vec = self._vectoriser.transform([cleaned_text]).toarray()
        cos_scores = cosine_similarity(query_vec, self._domain_vectors)[0]

        # Keyword overlap score
        tokens = set(cleaned_text.split())
        matches: list[DomainMatch] = []
        for idx, domain_name in enumerate(self._domain_names):
            kw_set = self._domain_keyword_sets[domain_name]
            overlap = tokens & kw_set
            kw_score = len(overlap) / max(len(kw_set), 1)

            # Hybrid: 60 % TF-IDF + 40 % keyword overlap
            combined = 0.6 * cos_scores[idx] + 0.4 * kw_score
            combined = min(combined, 1.0)

            matches.append(
                DomainMatch(
                    domain=domain_name,
                    confidence=round(float(combined), 4),
                    matched_keywords=sorted(overlap),
                )
            )

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    @staticmethod
    def _detect_intents(text: str) -> dict[str, bool]:
        return {name: bool(pat.search(text)) for name, pat in _INTENT_PATTERNS.items()}
