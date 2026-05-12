# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SVOExtractor — LLM-driven triple extraction with chonk-owned prompt."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ._llm import LLMClient
from ._svo import VERB_SET, SVOTriple

_SYSTEM_PROMPT = """\
You are a knowledge-graph extractor. Given a text passage, extract Subject-Verb-Object triples.

Rules:
- subject_id and object_id must be concise entity identifiers (snake_case or PascalCase, no spaces)
- verb must be EXACTLY one value from the allowed vocabulary (no synonyms, no invention)
- confidence must be a float in [0.0, 1.0] reflecting how certain you are of the relationship
- Return ONLY a JSON array of objects; no prose, no markdown fences
- Each object: {{"subject_id": "...", "verb": "...", "object_id": "...", "confidence": 0.0}}
- If no clear relationships exist, return []

Allowed verbs:
{verbs}
"""

_USER_TEMPLATE = """\
Chunk ID: {chunk_id}
Text:
{text}

Extract triples as JSON array.
"""

# ── Entity-anchored extraction ────────────────────────────────────────────────

_ENTITY_ANCHORED_SYSTEM_PROMPT = """\
You are a knowledge-graph extractor. Given a text passage and the entities that co-occur in it, \
extract Subject-Verb-Object triples between those entities and, for any entity lacking a description, \
generate a concise one-sentence description and a short list of aliases (alternate names or abbreviations).

Rules:
- subject_id and object_id must be EXACTLY entity IDs from the provided entity list — no others
- verb must be EXACTLY one value from the allowed vocabulary (no synonyms, no invention)
- confidence must be a float in [0.0, 1.0]
- descriptions must be one sentence, grounded in the text
- aliases must be a list of 1–3 alternate names or common abbreviations for the entity; omit if none apply
- Return ONLY a JSON object with exactly three keys — no prose, no markdown fences:
  {{
    "triples": [{{"subject_id": "...", "verb": "...", "object_id": "...", "confidence": 0.0}}],
    "descriptions": {{"entity_id": "one-sentence description"}},
    "aliases": {{"entity_id": ["alias1", "alias2"]}}
  }}
- Omit an entity from "descriptions" if it already has one (marked with ✓ below)
- Include aliases for ALL entities regardless of whether they already have a description
- If no clear relationships exist, return {{"triples": [], "descriptions": {{}}, "aliases": {{}}}}

Allowed verbs:
{verbs}
"""

_ENTITY_ANCHORED_USER_TEMPLATE = """\
Entities co-occurring in this chunk:
{entity_list}

Chunk ID: {chunk_id}
Text:
{text}

Extract triples and missing descriptions as JSON.
"""


class SVOExtractor:
    """Extracts SVOTriples from text using an injected LLM client.

    chonk owns the default prompt template and verb-set.
    All defaults are overridable for domain-specific benchmarking or tuning:

    Args:
        llm: Concrete LLMClient implementation (required — injected by caller).
        verb_set: Override the allowed verb vocabulary. Defaults to VERB_SET.
        system_prompt_template: Override the system prompt. Must contain a
            ``{verbs}`` placeholder. Defaults to the chonk standard prompt.
        user_template: Override the per-chunk user message. Must contain
            ``{chunk_id}`` and ``{text}`` placeholders.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        verb_set: frozenset[str] | None = None,
        system_prompt_template: str | None = None,
        user_template: str | None = None,
    ) -> None:
        if not isinstance(llm, LLMClient):
            raise TypeError(f"llm must implement LLMClient protocol, got {type(llm)}")
        self._llm = llm
        self._verb_set: frozenset[str] = verb_set if verb_set is not None else VERB_SET
        sys_tmpl = system_prompt_template if system_prompt_template is not None else _SYSTEM_PROMPT
        self._user_template: str = user_template if user_template is not None else _USER_TEMPLATE
        self._system = sys_tmpl.format(verbs=", ".join(sorted(self._verb_set)))

    def extract(self, text: str, chunk_id: str | None = None) -> list[SVOTriple]:
        """Extract SVOTriples from a text passage.

        Args:
            text: The passage to analyse.
            chunk_id: Optional source identifier stored on each triple.

        Returns:
            List of validated SVOTriples. Invalid rows are silently dropped.
        """
        prompt = self._system + "\n\n" + self._user_template.format(
            chunk_id=chunk_id or "",
            text=text,
        )
        raw = self._llm.complete(prompt)
        return self._parse(raw, chunk_id)

    def extract_batch(
        self,
        texts: Sequence[tuple[str, str | None]],
    ) -> list[SVOTriple]:
        """Extract from multiple (text, chunk_id) pairs.

        Args:
            texts: Sequence of (text, chunk_id) tuples.

        Returns:
            Flat list of all valid triples across all inputs.
        """
        results: list[SVOTriple] = []
        for text, chunk_id in texts:
            results.extend(self.extract(text, chunk_id))
        return results

    def extract_entity_anchored(
        self,
        text: str,
        chunk_id: str | None,
        entities: list[dict],
    ) -> tuple[list[SVOTriple], dict[str, str], dict[str, list[str]]]:
        """Entity-anchored extraction using co-occurrence hints.

        Args:
            text: Chunk content.
            chunk_id: Source chunk identifier.
            entities: List of dicts with keys ``id``, ``type``, and optionally
                ``description``.  Only pairs from this list are valid as
                subject/object.

        Returns:
            ``(triples, new_descriptions, new_aliases)`` where
            ``new_descriptions`` maps entity_id → description for entities
            that lacked one, and ``new_aliases`` maps entity_id → list of
            alternate names/abbreviations.
        """
        if len(entities) < 2:
            return [], {}, {}

        valid_ids: set[str] = {e["id"] for e in entities}
        lines = []
        for e in entities:
            has_desc = bool(e.get("description"))
            marker = "✓" if has_desc else " "
            desc_part = f" — {e['description']}" if has_desc else ""
            lines.append(f"  [{marker}] {e['id']} [{e.get('type', '?')}]{desc_part}")
        entity_list = "\n".join(lines)

        anchored_system = _ENTITY_ANCHORED_SYSTEM_PROMPT.format(
            verbs=", ".join(sorted(self._verb_set))
        )
        prompt = anchored_system + "\n\n" + _ENTITY_ANCHORED_USER_TEMPLATE.format(
            entity_list=entity_list,
            chunk_id=chunk_id or "",
            text=text,
        )
        raw = self._llm.complete(prompt)
        return self._parse_entity_anchored(raw, chunk_id, valid_ids)



    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: str, chunk_id: str | None) -> list[SVOTriple]:
        """Parse LLM response into validated SVOTriples, dropping bad rows."""
        # Strip markdown fences if the LLM ignored the instruction
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            rows = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

        if not isinstance(rows, list):
            return []

        triples: list[SVOTriple] = []
        for row in rows:
            try:
                verb = str(row["verb"])
                if verb not in self._verb_set:
                    continue
                triples.append(SVOTriple(
                    subject_id=str(row["subject_id"]),
                    verb=verb,
                    object_id=str(row["object_id"]),
                    confidence=float(row["confidence"]),
                    source_chunk_id=chunk_id,
                ))
            except (KeyError, TypeError, ValueError):
                continue

        return triples

    def _parse_entity_anchored(
        self,
        raw: str,
        chunk_id: str | None,
        valid_ids: set[str],
    ) -> tuple[list[SVOTriple], dict[str, str], dict[str, list[str]]]:
        """Parse entity-anchored LLM response."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return [], {}, {}

        if not isinstance(obj, dict):
            return [], {}, {}

        triples: list[SVOTriple] = []
        for row in obj.get("triples", []):
            try:
                verb = str(row["verb"])
                subj = str(row["subject_id"])
                obj_ = str(row["object_id"])
                if verb not in self._verb_set:
                    continue
                if subj not in valid_ids or obj_ not in valid_ids:
                    continue
                triples.append(SVOTriple(
                    subject_id=subj,
                    verb=verb,
                    object_id=obj_,
                    confidence=float(row["confidence"]),
                    source_chunk_id=chunk_id,
                ))
            except (KeyError, TypeError, ValueError):
                continue

        descriptions: dict[str, str] = {}
        for eid, desc in (obj.get("descriptions") or {}).items():
            if eid in valid_ids and isinstance(desc, str) and desc.strip():
                descriptions[eid] = desc.strip()

        aliases: dict[str, list[str]] = {}
        for eid, alias_list in (obj.get("aliases") or {}).items():
            if eid in valid_ids and isinstance(alias_list, list):
                clean = [a for a in alias_list if isinstance(a, str) and a.strip()]
                if clean:
                    aliases[eid] = clean

        return triples, descriptions, aliases
