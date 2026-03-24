"""Input adaptation helpers for flexible user instruction formats."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import orjson

from src.models.domain import DomainDataset
from src.models.request import GenerationMode

_DOMAIN_PATTERN = re.compile(r"\bdomain(?:_hint)?\s*[:=]\s*([a-z0-9_/\- ]+)", re.IGNORECASE)
_MODE_PATTERN = re.compile(r"\bmode\s*[:=]\s*(workflow|flowchart)\b", re.IGNORECASE)
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_ARROW_SPLIT_PATTERN = re.compile(r"\s*(?:->|=>|→|>>)\s*")
_STRUCTURED_LINE_PATTERN = re.compile(r"^\s*(?:domain|mode|format|input_format)\s*[:=]", re.IGNORECASE)
_TEXT_KEYS = ("instruction", "prompt", "task", "query", "title", "description", "summary")
_STEP_KEYS = ("steps", "nodes", "workflow", "activities", "tasks", "sequence")


@dataclass(slots=True)
class AdaptedInstruction:
    """Normalized representation of a user request."""

    original_text: str
    normalized_instruction: str
    detected_format: str = "plain_text"
    domain_hint: str | None = None
    mode_hint: GenerationMode | None = None
    step_hints: list[str] = field(default_factory=list)


class InputAdapter:
    """Normalize plain text, bullet lists, arrow flows, and JSON-like inputs."""

    def adapt(self, instruction: str) -> AdaptedInstruction:
        text = instruction.strip()
        if not text:
            return AdaptedInstruction(
                original_text=instruction,
                normalized_instruction=instruction,
            )

        json_payload = self._try_parse_json(text)
        if json_payload is not None:
            return self._adapt_json(instruction, json_payload)

        return self._adapt_text(instruction)

    def resolve_custom_steps(
        self,
        dataset: DomainDataset,
        step_hints: Iterable[str],
        existing_custom_steps: Iterable[str] | None = None,
    ) -> list[str]:
        """Resolve free-form step hints to dataset step ids."""
        resolved: list[str] = []
        seen: set[str] = set()

        for step_id in existing_custom_steps or []:
            normalised_id = step_id.strip().lower().replace(" ", "_")
            if normalised_id and normalised_id not in seen:
                resolved.append(normalised_id)
                seen.add(normalised_id)

        match_index: dict[str, str] = {}
        for step in dataset.steps:
            step_id = step.id.lower()
            match_index[step_id] = step.id
            match_index[self._slugify(step.id)] = step.id
            match_index[self._slugify(step.label)] = step.id

        for hint in step_hints:
            candidate = hint.strip()
            if not candidate:
                continue

            slug = self._slugify(candidate)
            step_id = match_index.get(slug)
            if step_id is None:
                step_id = self._find_partial_step_match(dataset, slug)
            if step_id and step_id not in seen:
                resolved.append(step_id)
                seen.add(step_id)

        return resolved

    def _adapt_json(self, original_text: str, payload: Any) -> AdaptedInstruction:
        domain_hint = self._extract_domain_hint(payload)
        mode_hint = self._extract_mode_hint(payload)
        text_parts = self._extract_text_segments(payload)
        step_hints = self._extract_step_hints(payload)

        normalized_parts: list[str] = []
        if domain_hint:
            normalized_parts.append(domain_hint.replace("_", " "))
        normalized_parts.extend(text_parts)
        if step_hints:
            normalized_parts.append("steps " + " ".join(step_hints))

        normalized_instruction = self._collapse_parts(normalized_parts) or original_text
        return AdaptedInstruction(
            original_text=original_text,
            normalized_instruction=normalized_instruction,
            detected_format="json",
            domain_hint=domain_hint,
            mode_hint=mode_hint,
            step_hints=step_hints,
        )

    def _adapt_text(self, original_text: str) -> AdaptedInstruction:
        text = original_text.strip()
        domain_hint = self._extract_domain_from_text(text)
        mode_hint = self._extract_mode_from_text(text)
        step_hints: list[str] = []
        detected_format = "plain_text"

        arrow_steps = self._extract_arrow_steps(text)
        if len(arrow_steps) >= 2:
            step_hints.extend(arrow_steps)
            detected_format = "arrow_flow"

        bullet_steps = self._extract_bullet_steps(text)
        if bullet_steps:
            step_hints.extend(bullet_steps)
            if detected_format == "plain_text":
                detected_format = "step_list"

        normalized_parts: list[str] = []
        if domain_hint:
            normalized_parts.append(domain_hint.replace("_", " "))

        filtered_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not _STRUCTURED_LINE_PATTERN.match(line)
        ]
        if filtered_lines:
            normalized_parts.append(" ".join(filtered_lines))
        if step_hints:
            normalized_parts.append("steps " + " ".join(step_hints))

        normalized_instruction = self._collapse_parts(normalized_parts) or text
        return AdaptedInstruction(
            original_text=original_text,
            normalized_instruction=normalized_instruction,
            detected_format=detected_format,
            domain_hint=domain_hint,
            mode_hint=mode_hint,
            step_hints=self._dedupe_preserve_order(step_hints),
        )

    @staticmethod
    def _try_parse_json(text: str) -> Any | None:
        stripped = text.strip()
        if not stripped or stripped[0] not in "{[":
            return None
        try:
            return orjson.loads(stripped)
        except orjson.JSONDecodeError:
            return None

    def _extract_domain_hint(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            for key in ("domain", "domain_hint"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return self._normalise_domain_value(value)
            workflow = payload.get("workflow")
            if workflow is not None:
                return self._extract_domain_hint(workflow)
        return None

    def _extract_mode_hint(self, payload: Any) -> GenerationMode | None:
        if isinstance(payload, dict):
            value = payload.get("mode")
            if isinstance(value, str):
                try:
                    return GenerationMode(value.strip().lower())
                except ValueError:
                    return None
            workflow = payload.get("workflow")
            if workflow is not None:
                return self._extract_mode_hint(workflow)
        return None

    def _extract_text_segments(self, payload: Any) -> list[str]:
        segments: list[str] = []
        if isinstance(payload, dict):
            for key in _TEXT_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    segments.append(value.strip())
            workflow = payload.get("workflow")
            if workflow is not None:
                segments.extend(self._extract_text_segments(workflow))
        return self._dedupe_preserve_order(segments)

    def _extract_step_hints(self, payload: Any) -> list[str]:
        hints: list[str] = []
        self._collect_step_hints(payload, hints)
        return self._dedupe_preserve_order(hints)

    def _collect_step_hints(self, value: Any, hints: list[str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _STEP_KEYS:
                    self._collect_step_hints(item, hints)
                    continue
                if key in {"id", "label", "name", "step"} and isinstance(item, str):
                    hints.append(item)
            return

        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    hints.append(item)
                else:
                    self._collect_step_hints(item, hints)

    def _extract_domain_from_text(self, text: str) -> str | None:
        match = _DOMAIN_PATTERN.search(text)
        if match:
            return self._normalise_domain_value(match.group(1))
        return None

    @staticmethod
    def _extract_mode_from_text(text: str) -> GenerationMode | None:
        match = _MODE_PATTERN.search(text)
        if not match:
            return None
        try:
            return GenerationMode(match.group(1).lower())
        except ValueError:
            return None

    def _extract_bullet_steps(self, text: str) -> list[str]:
        steps: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _STRUCTURED_LINE_PATTERN.match(stripped):
                continue
            if _BULLET_PATTERN.match(stripped):
                steps.append(_BULLET_PATTERN.sub("", stripped).strip())
        return self._dedupe_preserve_order(steps)

    def _extract_arrow_steps(self, text: str) -> list[str]:
        if not _ARROW_SPLIT_PATTERN.search(text):
            return []

        flattened = " ".join(line.strip() for line in text.splitlines() if line.strip())
        segments = [segment.strip(" ,.;") for segment in _ARROW_SPLIT_PATTERN.split(flattened)]
        return self._dedupe_preserve_order([segment for segment in segments if segment])

    @staticmethod
    def _normalise_domain_value(value: str) -> str:
        normalised = value.strip().lower()
        normalised = re.sub(r"\s+", "_", normalised)
        normalised = normalised.replace("/", "_")
        return re.sub(r"[^a-z0-9_]+", "", normalised).strip("_")

    @staticmethod
    def _slugify(value: str) -> str:
        value = value.lower().replace("_", " ")
        value = re.sub(r"[^a-z0-9 ]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def _find_partial_step_match(self, dataset: DomainDataset, slug: str) -> str | None:
        if len(slug) < 4:
            return None
        for step in dataset.steps:
            candidates = (self._slugify(step.id), self._slugify(step.label))
            if any(slug == candidate or slug in candidate or candidate in slug for candidate in candidates):
                return step.id
        return None

    @staticmethod
    def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = value.strip()
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _collapse_parts(parts: Iterable[str]) -> str:
        cleaned = [part.strip() for part in parts if part and part.strip()]
        return " ".join(cleaned).strip()
