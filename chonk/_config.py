# Copyright (c) 2025 Kenneth Stott. MIT License.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbedConfig:
    model: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 256


@dataclass
class LoaderConfig:
    min_chunk_size: int = 1100
    max_chunk_size: int = 2200
    enrich_context: bool = True
    extra_extractors: list[str] = field(default_factory=list)


@dataclass
class IndexConfig:
    ner: bool = True
    community: bool = True
    svo: bool = False
    spacy_model: str = "en_core_web_sm"
    svo_model: str = "gpt-4o-mini"
    community_alpha: float = 0.2
    community_sim_threshold: float = 0.6
    svo_llm_client: Any = None  # noqa: ANN401


@dataclass
class ChonkConfig:
    store: dict[str, object]
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    sources: list[dict[str, object]] = field(default_factory=list)
    namespaces: dict[str, object] = field(default_factory=dict)
    search: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChonkConfig:  # noqa: ANN401
        ec = d.get("embed") or {}
        lc = d.get("loader") or {}
        ic = d.get("index") or {}
        return cls(
            store=d.get("store") or {},
            embed=EmbedConfig(
                model=ec.get("model", "BAAI/bge-large-en-v1.5"),
                batch_size=int(ec.get("batch_size", 256)),
            ),
            loader=LoaderConfig(
                min_chunk_size=int(lc.get("min_chunk_size", 1100)),
                max_chunk_size=int(lc.get("max_chunk_size", 2200)),
                enrich_context=bool(lc.get("enrich_context", True)),
                extra_extractors=list(lc.get("extra_extractors") or []),
            ),
            index=IndexConfig(
                ner=bool(ic.get("ner", True)),
                community=bool(ic.get("community", True)),
                svo=bool(ic.get("svo", False)),
                spacy_model=str(ic.get("spacy_model", "en_core_web_sm")),
                svo_model=str(ic.get("svo_model", "gpt-4o-mini")),
                community_alpha=float(ic.get("community_alpha", 0.2)),
                community_sim_threshold=float(ic.get("community_sim_threshold", 0.6)),
                svo_llm_client=ic.get("svo_llm_client"),
            ),
            sources=list(d.get("sources") or []),
            namespaces=dict(d.get("namespaces") or {}),
            search=dict(d.get("search") or {}),
        )
