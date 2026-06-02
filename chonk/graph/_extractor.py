# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d624dd7b-9fcb-48a2-a460-6d6d3527a0f0
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

_RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON. "
    "Return ONLY the JSON object or array — no prose, no markdown fences, no explanation."
)

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
generate a concise one-sentence description and a short list of aliases (alternate names or abbreviations). \
Also generate a one-sentence description for every extracted triple.

Rules:
- subject_id and object_id must be EXACTLY entity IDs from the provided entity list — no others
- verb must be EXACTLY one value from the allowed vocabulary (no synonyms, no invention); choose the most specific verb that fits — use "references" only when no more precise verb applies (e.g. prefer "part_of", "contains", "derived_from", "governs", "depends_on" when they accurately describe the relationship)
- confidence must be a float in [0.0, 1.0]
- descriptions must be one sentence describing what the entity means in the business domain — not technical implementation details, code identifiers, schema structure, or how the entity appears in data. Write as if explaining to a business analyst, not a developer. Bad: "customer info referenced by customer_id". Good: "An individual or organisation that purchases products or services."
- aliases must be a list of 1–3 meaningful alternate names or common abbreviations — never include plural or singular grammatical variants of the entity name itself (e.g. do not alias "order" with "orders"); omit if none apply
- rel_descriptions keys are "subject_id|verb|object_id" — one sentence describing the business meaning of the relationship, not technical implementation details (e.g. not "via a foreign key"). Write as if explaining to a business analyst.
- Return ONLY a JSON object with exactly four keys — no prose, no markdown fences:
  {{
    "triples": [{{"subject_id": "...", "verb": "...", "object_id": "...", "confidence": 0.0}}],
    "descriptions": {{"entity_id": "one-sentence description"}},
    "aliases": {{"entity_id": ["alias1", "alias2"]}},
    "rel_descriptions": {{"subject_id|verb|object_id": "one-sentence relationship description"}}
  }}
- Omit an entity from "descriptions" if it already has one (marked with ✓ below)
- Include aliases for ALL entities regardless of whether they already have a description
- Include a rel_description for every triple in "triples"
- If no clear relationships exist, return {{"triples": [], "descriptions": {{}}, "aliases": {{}}, "rel_descriptions": {{}}}}

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
        result = self._parse(raw, chunk_id)
        if result is None:
            raw = self._llm.complete(prompt + _RETRY_SUFFIX)
            result = self._parse(raw, chunk_id)
        return result or []

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
    ) -> tuple[list[SVOTriple], dict[str, str], dict[str, list[str]], dict[str, str]]:
        """Entity-anchored extraction using co-occurrence hints.

        Args:
            text: Chunk content.
            chunk_id: Source chunk identifier.
            entities: List of dicts with keys ``id``, ``type``, and optionally
                ``description``.  Only pairs from this list are valid as
                subject/object.

        Returns:
            ``(triples, new_descriptions, new_aliases, rel_descriptions)`` where
            ``new_descriptions`` maps entity_id → description for entities
            that lacked one, ``new_aliases`` maps entity_id → list of
            alternate names/abbreviations, and ``rel_descriptions`` maps
            ``"subject_id|verb|object_id"`` → one-sentence relationship description.
        """
        if len(entities) < 2:
            return [], {}, {}, {}

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
        result = self._parse_entity_anchored(raw, chunk_id, valid_ids)
        if result is None:
            raw = self._llm.complete(prompt + _RETRY_SUFFIX)
            result = self._parse_entity_anchored(raw, chunk_id, valid_ids)
        return result or ([], {}, {}, {})



    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: str, chunk_id: str | None) -> list[SVOTriple] | None:
        """Parse LLM response into validated SVOTriples. Returns None on JSON failure."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            rows = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(rows, list):
            return None

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
    ) -> tuple[list[SVOTriple], dict[str, str], dict[str, list[str]], dict[str, str]] | None:
        """Parse entity-anchored LLM response. Returns None on JSON failure."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(obj, dict):
            return None

        # Parse rel_descriptions first so they can be attached to triples
        rel_descriptions: dict[str, str] = {}
        for key, desc in (obj.get("rel_descriptions") or {}).items():
            if isinstance(key, str) and isinstance(desc, str) and desc.strip():
                rel_descriptions[key] = desc.strip()

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
                rel_key = f"{subj}|{verb}|{obj_}"
                triples.append(SVOTriple(
                    subject_id=subj,
                    verb=verb,
                    object_id=obj_,
                    confidence=float(row["confidence"]),
                    source_chunk_id=chunk_id,
                    description=rel_descriptions.get(rel_key, ""),
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

        return triples, descriptions, aliases, rel_descriptions
