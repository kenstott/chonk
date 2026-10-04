# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Dictionary-based singular forms for entity and schema names.

Suffix rules get common names wrong: stripping a trailing "s" turns ``status``
into ``statu``, and ``inflect`` (which assumes its input is plural) turns
``address`` into ``addres``. This module checks WordNet first, so a word the
dictionary knows as singular is kept and an irregular plural (``indices``,
``vertices``) maps to its real singular.

Order, for one lowercase word:

1. Words used as singular in data work (``data``, ``media``) are kept.
2. A WordNet lemma that pluralises back to the word is its singular
   (``addresses`` -> ``address``, ``children`` -> ``child``). Requiring the
   round trip rejects lemmas that WordNet's suffix rules reach by accident
   (``cves`` -> ``cf``).
3. A word WordNet knows as a noun is already singular (``status``, ``bus``,
   ``lei``).
4. Otherwise a WordNet irregular plural maps to its lemma (``indices`` ->
   ``index``), even where the regular plural differs (``indexes``).
5. For a word WordNet does not know, ``inflect`` proposes a singular, kept only
   if it pluralises back to the word (``cves`` -> ``cve``).
6. Otherwise the word is unchanged.

Needs the ``ner`` extra (``inflect``, ``nltk``) and the WordNet data:
``python -m nltk.downloader wordnet``.
"""

from __future__ import annotations

from functools import cache
from typing import Any

# Words that are plural in origin but used as singular in data work.
SINGULAR_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "data",
        "metadata",
        "criteria",
        "media",
        "agenda",
        "stamina",
        "trivia",
        "insignia",
    }
)


@cache
def _wordnet() -> Any:  # noqa: ANN401
    try:
        from nltk.corpus import wordnet
    except ImportError as exc:
        raise ImportError("singular forms need nltk: install chonk-rag[ner]") from exc
    try:
        wordnet.ensure_loaded()
    except LookupError as exc:
        raise LookupError(
            "singular forms need the WordNet data: python -m nltk.downloader wordnet"
        ) from exc
    return wordnet


@cache
def _inflect() -> Any:  # noqa: ANN401
    try:
        import inflect
    except ImportError as exc:
        raise ImportError("singular forms need inflect: install chonk-rag[ner]") from exc
    return inflect.engine()


def _pluralises_to(candidate: str, word: str) -> bool:
    return _inflect().plural_noun(candidate) == word


@cache
def singularize(word: str) -> str:
    """Singular form of one lowercase word; the word itself if it is singular."""
    if len(word) < 3 or not word.isalpha() or word in SINGULAR_EXCEPTIONS:
        return word
    wn = _wordnet()
    for lemma in wn._morphy(word, wn.NOUN):
        if lemma != word and _pluralises_to(lemma, word):
            return lemma
    # The word itself is a noun lemma, so already singular (status, lei). Checked
    # with lemmas(), not synsets(), which also matches words that WordNet's suffix
    # rules map onto another lemma (cves -> cf).
    if wn.lemmas(word, pos=wn.NOUN):
        return word
    irregular = wn._exception_map["n"].get(word)  # indices -> [index]
    if irregular:
        return irregular[0]
    candidate = _inflect().singular_noun(word)
    if candidate and _pluralises_to(candidate, word):
        return candidate
    return word
