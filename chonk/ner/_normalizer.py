# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Entity name normalizer for deduplication.

Pipeline per entity string:
1. Strip leading/trailing whitespace and symbols (brackets, quotes, punctuation).
2. Collapse internal whitespace.
3. Split on camelCase boundaries or underscores if no spaces are present.
4. Singularize the last token using ``inflect`` (head-noun rule).
   - ``inflect.singular_noun`` returns False for already-singular words → no-op.
   - Words in ``SINGULAR_EXCEPTIONS`` are never singularized.
5. Re-join tokens preserving original separator style.

``normalize_entity`` → canonical display form (preserves acronym casing).
``canonical_key``    → lowercase form used as the dedup dictionary key.
"""

from __future__ import annotations

import re

# Words that inflect mis-singularizes for typical data-domain use.
SINGULAR_EXCEPTIONS: frozenset[str] = frozenset({
    "data",
    "metadata",
    "criteria",   # domain often uses "criteria" as singular
    "media",
    "agenda",
    "stamina",
    "trivia",
    "insignia",
})

# Compiled patterns
_LEADING_TRAILING_SYMBOLS = re.compile(r"^[^\w\s]+|[^\w\s]+$", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Dotted acronyms: U.S.A. or U.S.A → USA
_DOTTED_ACRONYM = re.compile(r"^([A-Z0-9]\.){2,}[A-Z0-9]?\.?$")


def _is_acronym(word: str) -> bool:
    """True when every letter in *word* is uppercase and there are ≥2 letters."""
    letters = [c for c in word if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def _singularize(word: str) -> str:
    """Singularize *word*; return original if already singular, unknown, or an acronym."""
    if not word:
        return word
    if _is_acronym(word):
        return word
    lower = word.lower()
    if lower in SINGULAR_EXCEPTIONS:
        return word
    try:
        import inflect as _inflect_mod
        _engine = _get_engine()
        result = _engine.singular_noun(word)
        if result is False:
            return word
        # Preserve original casing style on the singular form
        if word[0].isupper() and not _is_acronym(word):
            return result.capitalize() if result else word
        return result
    except ImportError:
        return word


_inflect_engine = None


def _get_engine():
    global _inflect_engine
    if _inflect_engine is None:
        import inflect
        _inflect_engine = inflect.engine()
    return _inflect_engine


def _split_tokens(text: str) -> tuple[list[str], str]:
    """Split *text* into tokens; return (tokens, separator).

    For space-separated phrases the separator is ' '.
    For snake_case identifiers the separator is '_'.
    For CamelCase identifiers the separator is '' (re-joined without separator).
    """
    if " " in text:
        return text.split(" "), " "
    if "_" in text:
        return text.split("_"), "_"
    # CamelCase
    parts = _CAMEL_BOUNDARY.split(text)
    if len(parts) > 1:
        return parts, ""
    return [text], ""


def normalize_entity(entity: str) -> str:
    """Return the canonical display form of *entity*.

    - Strips leading/trailing symbols.
    - Singularizes the head noun (last token).
    - Preserves acronym casing (IBM → IBM).
    - Returns empty string if nothing is left after stripping.
    """
    if not entity:
        return ""

    # 1. Strip whitespace
    entity = entity.strip()

    # 2. Strip leading/trailing non-word/non-space characters
    entity = _LEADING_TRAILING_SYMBOLS.sub("", entity).strip()

    # 3. Collapse dotted acronyms: U.S.A. → USA (before whitespace collapse)
    entity = " ".join(
        re.sub(r"\.", "", tok) if _DOTTED_ACRONYM.match(tok + ("." if not tok.endswith(".") else "")) else tok
        for tok in entity.split(" ")
    )

    # 4. Collapse internal whitespace
    entity = _MULTI_SPACE.sub(" ", entity)

    if not entity:
        return ""

    # 4. Split, singularize last token, rejoin
    tokens, sep = _split_tokens(entity)
    tokens = [t for t in tokens if t]  # drop empty tokens from split artefacts
    if not tokens:
        return ""

    tokens[-1] = _singularize(tokens[-1])
    return sep.join(tokens)


def canonical_key(entity: str) -> str:
    """Lowercase normalized key for deduplication matching."""
    return normalize_entity(entity).lower()


class EntityNormalizer:
    """Stateful normalizer with a configurable exceptions list.

    For most use cases the module-level :func:`normalize_entity` and
    :func:`canonical_key` functions are sufficient.  Use this class when you
    need per-instance exception lists or want to subclass normalization behaviour.
    """

    def __init__(self, extra_exceptions: frozenset[str] | None = None) -> None:
        self._exceptions = SINGULAR_EXCEPTIONS | (extra_exceptions or frozenset())

    def normalize(self, entity: str) -> str:
        """Return canonical display form. Respects instance exception list."""
        if not entity:
            return ""
        entity = entity.strip()
        entity = _LEADING_TRAILING_SYMBOLS.sub("", entity).strip()
        entity = _MULTI_SPACE.sub(" ", entity)
        if not entity:
            return ""
        tokens, sep = _split_tokens(entity)
        tokens = [t for t in tokens if t]
        if not tokens:
            return ""
        last = tokens[-1]
        if not _is_acronym(last) and last.lower() not in self._exceptions:
            try:
                result = _get_engine().singular_noun(last)
                if result is not False:
                    tokens[-1] = result.capitalize() if last[0].isupper() and not _is_acronym(last) else result
            except ImportError:
                pass
        return sep.join(tokens)

    def canonical_key(self, entity: str) -> str:
        """Lowercase normalized key for deduplication matching."""
        return self.normalize(entity).lower()
