# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SVOExtractor — LLM-driven triple extraction with chonk-owned prompt."""

from __future__ import annotations

import json
import re
from typing import Sequence

from ._llm import LLMClient
from ._svo import SVOTriple, VERB_SET

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
