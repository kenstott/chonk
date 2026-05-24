#!/usr/bin/env python3
"""Constraint-aware parallel runner for GraphRAG-Bench + FANG-2026 paper runs.

Constraints:
  - max 1 reranking job at a time (local reranker is a bottleneck)
  - max 1 job per gen API provider (openai / anthropic / together)
  - max 3 concurrent jobs total

Usage (from repo root on GPU):
  python work/run_parallel.py [--dry-run] [--grb-only] [--fang-only]
"""

import argparse
import asyncio
import os
import re
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass, field

# Load .env before anything else so API keys are available to subprocesses
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                if _k and _k == _k.strip() and _k not in os.environ:
                    os.environ[_k] = _v
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
PY = os.environ.get("CHONK_PYTHON", "/root/miniforge/envs/chonk/bin/python")
REPO = Path(__file__).parent.parent  # repo root

GRB_OUT = str(REPO / "work")
GRB_QIDs = str(REPO / "work/data/full_corpus_stratified_order.json")
GRB_CONFIGS = REPO / "work/configs/runs"

FANG_OUT = str(REPO / "work/fang2026")
FANG_QIDs = str(REPO / "work/fang2026/data/fang2026_question_ids.json")
FANG_DB = "chonk_nobc_1100_2200_gleif.duckdb"
FANG_BC_DB = "chonk_bc_1100_2200.duckdb"
FANG_VANILLA_DB = "vanilla_rag.duckdb"

GRB_EVAL = [
    "--judge",
    "gpt-4o-mini",
    "--eval-rpm",
    "8000",
    "--eval-batch-size",
    "20",
    "--concurrency",
    "50",
    "--nan-limit",
    "136",
]
FANG_EVAL = [
    "--judge",
    "gpt-4o-mini",
    "--eval-rpm",
    "8000",
    "--eval-batch-size",
    "20",
    "--concurrency",
    "50",
    "--nan-limit",
    "5",
]

# Configs whose run_names match these patterns are excluded from GRB runs
GRB_EXCLUDE_PATTERNS = [
    r"_llama8b_sr",  # Llama 8B SR/SRR excluded
    r"_qwen",  # Qwen excluded
]


# ── Job definition ─────────────────────────────────────────────────────────────
@dataclass
class Job:
    name: str
    run_flags: list[str]  # flags for `graphrag_bench.py run ...` (after --out-dir, --run-name)
    eval_flags: list[str]  # extra flags for eval command
    out_dir: str
    uses_rerank: bool
    provider: str  # openai | anthropic | together
    db_name: str = "chunkymonkey_1100_2200.duckdb"  # source DB filename (for isolation copy)
    depends_on: list[str] = field(default_factory=list)  # job names that must complete first

    @property
    def gen_file(self) -> Path:
        return Path(self.out_dir) / "results" / f"{self.name}.jsonl"

    @property
    def scores_file(self) -> Path:
        return Path(self.out_dir) / "results" / f"bench_eval_{self.name}_rp.json"

    def is_done(self) -> bool:
        return self.scores_file.exists()

    def gen_done(self) -> bool:
        return self.gen_file.exists()


def _replace_db_name(flags: list[str], new_db: str) -> list[str]:
    out = list(flags)
    try:
        i = out.index("--db-name")
        out[i + 1] = new_db
    except ValueError:
        out += ["--db-name", new_db]
    return out


# ── GRB jobs from TOML configs ─────────────────────────────────────────────────
def _parse_toml_job(toml_path: Path) -> Job | None:
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    run_name = raw.get("run_name")
    if not run_name:
        return None

    for pat in GRB_EXCLUDE_PATTERNS:
        if re.search(pat, run_name):
            return None

    uses_rerank = raw.get("rerank", {}).get("enabled", False)
    provider = raw.get("gen", {}).get("provider", "openai")
    db_name = raw.get("index", {}).get("db_name", "chunkymonkey_1100_2200.duckdb")

    run_flags = [
        "--config",
        str(toml_path),
        "--run-name",
        run_name,
        "--question-ids",
        GRB_QIDs,
    ]

    return Job(
        name=run_name,
        run_flags=run_flags,
        eval_flags=GRB_EVAL,
        out_dir=GRB_OUT,
        uses_rerank=uses_rerank,
        provider=provider,
        db_name=db_name,
    )


def build_grb_jobs() -> list[Job]:
    jobs = []
    for p in sorted(GRB_CONFIGS.glob("*_full.toml")):
        if p.name.startswith(".") or "_grid" in p.name:
            continue
        job = _parse_toml_job(p)
        if job:
            jobs.append(job)
    return jobs


# ── FANG jobs (inline) ─────────────────────────────────────────────────────────
def _fang(
    name: str,
    flags: list[str],
    *,
    rerank: bool,
    provider: str,
    db_name: str,
    depends_on: list[str] | None = None,
) -> Job:
    run_flags = flags + ["--run-name", name, "--question-ids", FANG_QIDs]
    return Job(
        name=name,
        run_flags=run_flags,
        eval_flags=FANG_EVAL,
        out_dir=FANG_OUT,
        uses_rerank=rerank,
        provider=provider,
        db_name=db_name,
        depends_on=depends_on or [],
    )


MINI = ["--gen-provider", "openai", "--gen-model", "gpt-4o-mini"]
HAIKU = ["--gen-provider", "anthropic", "--gen-model", "claude-haiku-4-5-20251001"]
LANED60 = [
    "--enhanced",
    "--entity-ref-expansion",
    "--lane-entity-min-sim",
    "0.60",
    "--community-context",
    "--community-min-coherence",
    "0.5",
    "--top-k",
    "10",
]
LANED60_K30 = [
    "--enhanced",
    "--entity-ref-expansion",
    "--lane-entity-min-sim",
    "0.60",
    "--community-context",
    "--community-min-coherence",
    "0.5",
    "--top-k",
    "30",
]
LANED60_K50 = [
    "--enhanced",
    "--entity-ref-expansion",
    "--lane-entity-min-sim",
    "0.60",
    "--community-context",
    "--community-min-coherence",
    "0.5",
    "--top-k",
    "50",
]
LANED50_K50 = [
    "--enhanced",
    "--entity-ref-expansion",
    "--lane-entity-min-sim",
    "0.50",
    "--community-context",
    "--community-min-coherence",
    "0.5",
    "--top-k",
    "50",
]
LANED45_K50 = [
    "--enhanced",
    "--entity-ref-expansion",
    "--lane-entity-min-sim",
    "0.45",
    "--community-context",
    "--community-min-coherence",
    "0.5",
    "--top-k",
    "50",
]
CLUSTER = [
    "--enhanced",
    "--entity-ref-expansion",
    "--cluster",
    "--community-context",
    "--community-min-coherence",
    "0.5",
    "--top-k",
    "10",
]
GF = ["--enhanced", "--entity-ref-expansion", "--search-mode", "graph_first", "--top-k", "10"]

FANG_INDEX_JOB = "fang_index_vanilla"  # sentinel — not a real run, just prereq tracking
FANG_BC_INDEX_JOB = "fang_index_bc"    # sentinel for BC index


def build_fang_jobs() -> list[Job]:
    db = ["--db-name", FANG_DB]
    bcdb = ["--db-name", FANG_BC_DB]
    vdb = ["--db-name", FANG_VANILLA_DB]
    gf_deps = [FANG_INDEX_JOB]
    bc_deps = [FANG_BC_INDEX_JOB]

    return [
        # ── gpt-4o-mini ──────────────────────────────────────────────────────
        _fang(
            "fang_vanilla_rerank_mini",
            ["--vanilla", "--rerank"] + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_VANILLA_DB,
        ),
        _fang(
            "fang_vanilla_rerank_srr_mini",
            ["--vanilla", "--rerank", "--srr"] + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_VANILLA_DB,
        ),
        _fang(
            "fang_ner_ref_cluster_community_k10_mini",
            ["--rerank"] + CLUSTER + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_cluster_community_k10_srr_mini",
            ["--rerank", "--srr"] + CLUSTER + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_graph_first_k10_mini",
            GF + vdb + MINI,
            rerank=False,
            provider="openai",
            db_name=FANG_VANILLA_DB,
            depends_on=gf_deps,
        ),
        # ── claude-haiku-4-5 ─────────────────────────────────────────────────
        _fang(
            "fang_vanilla_rerank_haiku",
            ["--vanilla", "--rerank"] + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_VANILLA_DB,
        ),
        _fang(
            "fang_vanilla_rerank_srr_haiku",
            ["--vanilla", "--rerank", "--srr"] + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_VANILLA_DB,
        ),
        _fang(
            "fang_ner_ref_cluster_community_k10_haiku",
            ["--rerank"] + CLUSTER + db + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_cluster_community_k10_srr_haiku",
            ["--rerank", "--srr"] + CLUSTER + db + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_graph_first_k10_haiku",
            GF + vdb + HAIKU,
            rerank=False,
            provider="anthropic",
            db_name=FANG_VANILLA_DB,
            depends_on=gf_deps,
        ),
        # ── laned60 k30 ──────────────────────────────────────────────────────
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_mini",
            ["--rerank"] + LANED60_K30 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_srr_mini",
            ["--rerank", "--srr"] + LANED60_K30 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_srr_mini",
            ["--srr"] + LANED60_K30 + db + MINI,
            rerank=False,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_haiku",
            ["--rerank"] + LANED60_K30 + db + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_srr_haiku",
            ["--rerank", "--srr"] + LANED60_K30 + db + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        # ── sonnet + OSS (top config, model comparison) ──────────────────────
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_srr_sonnet",
            ["--rerank", "--srr"] + LANED60_K30 + db + ["--gen-provider", "anthropic", "--gen-model", "claude-sonnet-4-6"],
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_srr_gptoss120b",
            ["--rerank", "--srr"] + LANED60_K30 + db + ["--gen-provider", "together", "--gen-model", "openai/gpt-oss-120b"],
            rerank=True,
            provider="together",
            db_name=FANG_DB,
        ),
        # ── laned k50 grid (cross-domain threshold sweep, mini only) ─────────
        _fang(
            "fang_ner_ref_laned60_community_k50_rerank_srr_mini",
            ["--rerank", "--srr"] + LANED60_K50 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned50_community_k50_rerank_srr_mini",
            ["--rerank", "--srr"] + LANED50_K50 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned45_community_k50_rerank_srr_mini",
            ["--rerank", "--srr"] + LANED45_K50 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        # ── laned60 k10 rerank ───────────────────────────────────────────────
        _fang(
            "fang_ner_ref_laned60_community_k10_rerank_mini",
            ["--rerank"] + LANED60 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k10_rerank_srr_mini",
            ["--rerank", "--srr"] + LANED60 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k10_rerank_haiku",
            ["--rerank"] + LANED60 + db + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k10_rerank_srr_haiku",
            ["--rerank", "--srr"] + LANED60 + db + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        # ── laned60 k30 no-rerank ─────────────────────────────────────────────
        _fang(
            "fang_ner_ref_laned60_community_k30_mini",
            LANED60_K30 + db + MINI,
            rerank=False,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_haiku",
            LANED60_K30 + db + HAIKU,
            rerank=False,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_srr_haiku",
            ["--srr"] + LANED60_K30 + db + HAIKU,
            rerank=False,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_srr_bm25_mini",
            ["--srr", "--bm25"] + LANED60_K30 + db + MINI,
            rerank=False,
            provider="openai",
            db_name=FANG_DB,
        ),
        # ── laned60 k30 ADF / no_gleif variants ──────────────────────────────
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_srr_mini_adf",
            ["--rerank", "--srr", "--auto-domain-filter"] + LANED60_K30 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_srr_mini_no_gleif",
            ["--rerank", "--srr", "--domain-ids", "patents", "sec_10k", "cve", "fed_reg"] + LANED60_K30 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_srr_haiku_adf",
            ["--rerank", "--srr", "--auto-domain-filter"] + LANED60_K30 + db + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k30_rerank_srr_gptoss120b_adf",
            ["--rerank", "--srr", "--auto-domain-filter"] + LANED60_K30 + db + ["--gen-provider", "together", "--gen-model", "openai/gpt-oss-120b"],
            rerank=True,
            provider="together",
            db_name=FANG_DB,
        ),
        # ── laned60 k50 haiku / sonnet / ADF / no_gleif ──────────────────────
        _fang(
            "fang_ner_ref_laned60_community_k50_rerank_srr_haiku",
            ["--rerank", "--srr"] + LANED60_K50 + db + HAIKU,
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k50_rerank_srr_mini_adf",
            ["--rerank", "--srr", "--auto-domain-filter"] + LANED60_K50 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k50_rerank_srr_mini_no_gleif",
            ["--rerank", "--srr", "--domain-ids", "patents", "sec_10k", "cve", "fed_reg"] + LANED60_K50 + db + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_DB,
        ),
        _fang(
            "fang_ner_ref_laned60_community_k50_rerank_srr_sonnet",
            ["--rerank", "--srr"] + LANED60_K50 + db + ["--gen-provider", "anthropic", "--gen-model", "claude-sonnet-4-6"],
            rerank=True,
            provider="anthropic",
            db_name=FANG_DB,
        ),
        # ── BC laned60 k30 (BC index required) ───────────────────────────────
        _fang(
            "fang_ner_ref_bc_laned60_community_k30_mini",
            ["--rerank"] + LANED60_K30 + bcdb + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k30_srr_mini",
            ["--rerank", "--srr"] + LANED60_K30 + bcdb + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k30_srr_bm25_mini",
            ["--rerank", "--srr", "--bm25"] + LANED60_K30 + bcdb + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k30_srr_bm25_mini_adf",
            ["--rerank", "--srr", "--bm25", "--auto-domain-filter"] + LANED60_K30 + bcdb + MINI + ["--community-min-coherence", "0.0"],
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k30_srr_bm25_mini_pruned",
            ["--rerank", "--srr", "--bm25", "--redundancy-threshold", "0.92"] + LANED60_K30 + bcdb + MINI + ["--community-min-coherence", "0.0"],
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k30_srr_bm25_mini_adf_pruned",
            ["--rerank", "--srr", "--bm25", "--redundancy-threshold", "0.92", "--auto-domain-filter"] + LANED60_K30 + bcdb + MINI + ["--community-min-coherence", "0.0"],
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k30_srr_bm25_mini_pruned99",
            ["--rerank", "--srr", "--bm25", "--redundancy-threshold", "0.99"] + LANED60_K30 + bcdb + MINI + ["--community-min-coherence", "0.0"],
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k30_srr_bm25_mini_adf_pruned99",
            ["--rerank", "--srr", "--bm25", "--redundancy-threshold", "0.99", "--auto-domain-filter"] + LANED60_K30 + bcdb + MINI + ["--community-min-coherence", "0.0"],
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        # ── BC laned60 k50 ───────────────────────────────────────────────────
        _fang(
            "fang_ner_ref_bc_laned60_community_k50_rerank_srr_bm25_mini",
            ["--rerank", "--srr", "--bm25"] + LANED60_K50 + bcdb + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k50_rerank_srr_bm25_mini_adf",
            ["--rerank", "--srr", "--bm25", "--auto-domain-filter"] + LANED60_K50 + bcdb + MINI,
            rerank=True,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k50_srr_bm25_mini",
            ["--srr", "--bm25"] + LANED60_K50 + bcdb + MINI,
            rerank=False,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
        _fang(
            "fang_ner_ref_bc_laned60_community_k50_srr_bm25_mini_adf",
            ["--srr", "--bm25", "--auto-domain-filter"] + LANED60_K50 + bcdb + MINI,
            rerank=False,
            provider="openai",
            db_name=FANG_BC_DB,
            depends_on=bc_deps,
        ),
    ]


# ── Runner ─────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


async def run_cmd(cmd: list[str], tag: str) -> int:
    log(f"START {tag}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout
    async for line in proc.stdout:
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
    rc = await proc.wait()
    log(f"{'OK' if rc == 0 else 'FAIL'} {tag} (exit={rc})")
    return rc


async def run_job(
    job: Job,
    global_sem: asyncio.Semaphore,
    rerank_sem: asyncio.Semaphore,
    provider_sems: dict[str, asyncio.Semaphore],
    eval_sem: asyncio.Semaphore,
    completed: set[str],
    dry_run: bool,
) -> bool:
    # wait for dependencies
    while not all(d in completed or d == FANG_INDEX_JOB for d in job.depends_on):
        await asyncio.sleep(5)

    if job.is_done():
        log(f"SKIP {job.name} (eval done)")
        completed.add(job.name)
        return True

    # ── Gen phase: hold global + provider (+ rerank) semaphores ──────────────
    if not job.gen_done():
        gen_sems = [global_sem, provider_sems[job.provider]]
        if job.uses_rerank:
            gen_sems.append(rerank_sem)

        loop = asyncio.get_event_loop()

        async with gen_sems[0]:
            async with gen_sems[1]:
                ctx = gen_sems[2] if len(gen_sems) > 2 else None

                async def _run_gen() -> bool:
                    data_dir = Path(job.out_dir) / "data"
                    src_db = data_dir / job.db_name
                    iso_name = f"{job.name}.duckdb"
                    iso_db = data_dir / iso_name
                    iso_wal = data_dir / f"{iso_name}.wal"

                    if dry_run:
                        log(f"DRY-RUN COPY-DB {job.db_name} → {iso_name}")
                    else:
                        log(f"COPY-DB {job.db_name} → {iso_name}")
                        await loop.run_in_executor(None, shutil.copy2, str(src_db), str(iso_db))
                        wal = src_db.with_suffix(".duckdb.wal")
                        if wal.exists():
                            await loop.run_in_executor(None, shutil.copy2, str(wal), str(iso_wal))

                    iso_flags = _replace_db_name(job.run_flags, iso_name)
                    cmd = [PY, "demo/graphrag_bench.py", "run", "--out-dir", job.out_dir] + iso_flags
                    if dry_run:
                        log(f"DRY-RUN GEN {job.name}: {' '.join(cmd)}")
                        return True
                    rc = await run_cmd(cmd, f"GEN {job.name}")
                    for p in (iso_db, iso_wal):
                        if p.exists():
                            p.unlink()
                    if rc != 0:
                        log(f"ERROR GEN {job.name} — skipping eval")
                        return False
                    return True

                if ctx:
                    async with ctx:
                        ok = await _run_gen()
                else:
                    ok = await _run_gen()

        if not ok:
            return False

    if not job.gen_done() and not dry_run:
        log(f"SKIP EVAL {job.name} — no gen output")
        return False

    # ── Score phase: typed scorer replaces LLM judge ─────────────────────────
    schemas = str(Path(job.out_dir) / "data" / "fang2026_gold_schemas.jsonl")
    score_cmd = [
        PY, "work/score_typed.py",
        "--schemas", schemas,
        "--results", str(job.gen_file),
        "--out", str(job.scores_file),
        "--local-embed-model", "BAAI/bge-large-en-v1.5",
    ]
    if dry_run:
        log(f"DRY-RUN SCORE {job.name}: {' '.join(score_cmd)}")
    else:
        async with eval_sem:
            rc = await run_cmd(score_cmd, f"SCORE {job.name}")
        if rc != 0:
            log(f"ERROR SCORE {job.name}")
            return False

    completed.add(job.name)
    return True


async def run_fang_index_bc(dry_run: bool) -> bool:
    bc_db = Path(FANG_OUT) / "data" / FANG_BC_DB
    if bc_db.exists():
        log("SKIP index-bc (exists)")
        return True
    cmd = [PY, "work/chonk_ingest.py", "work/fang2026_bc.yaml"]
    if dry_run:
        log(f"DRY-RUN: {' '.join(cmd)}")
        return True
    return await run_cmd(cmd, "FANG index-bc") == 0


async def run_fang_index_vanilla(dry_run: bool) -> bool:
    fang_results = Path(FANG_OUT) / "results"
    fang_results.mkdir(parents=True, exist_ok=True)
    (Path(FANG_OUT) / "logs").mkdir(parents=True, exist_ok=True)

    vanilla_db = Path(FANG_OUT) / "data" / FANG_VANILLA_DB
    if vanilla_db.exists():
        log("SKIP index-vanilla (exists)")
        return True

    cmd = [
        PY,
        "demo/graphrag_bench.py",
        "index-vanilla",
        "--out-dir",
        FANG_OUT,
        "--from-store",
        str(Path(FANG_OUT) / "data" / FANG_DB),
    ]
    if dry_run:
        log(f"DRY-RUN: {' '.join(cmd)}")
        return True
    return await run_cmd(cmd, "FANG index-vanilla") == 0


async def prime_caches(out_dirs: list[str], dry_run: bool) -> None:
    for out_dir in out_dirs:
        cmd = [PY, "demo/graphrag_bench.py", "prime-cache", "--out-dir", out_dir]
        if dry_run:
            log(f"DRY-RUN: {' '.join(cmd)}")
        else:
            await run_cmd(cmd, f"prime-cache {out_dir}")


async def main(grb_only: bool, fang_only: bool, dry_run: bool) -> None:
    jobs: list[Job] = []
    if not fang_only:
        jobs += build_grb_jobs()
    if not grb_only:
        jobs += build_fang_jobs()

    total = len(jobs)
    already_done = sum(1 for j in jobs if j.is_done())
    log(f"Jobs: {total} total, {already_done} already done, {total - already_done} to run")

    global_sem = asyncio.Semaphore(3)
    rerank_sem = asyncio.Semaphore(1)
    eval_sem = asyncio.Semaphore(4)  # local BAAI embedder — no RPM limit; parallel scoring safe
    provider_sems = {
        "openai": asyncio.Semaphore(1),
        "anthropic": asyncio.Semaphore(1),
        "together": asyncio.Semaphore(1),
    }
    completed: set[str] = set()

    # prime caches
    out_dirs = []
    if not fang_only:
        out_dirs.append(GRB_OUT)
    if not grb_only:
        out_dirs.append(FANG_OUT)
    await prime_caches(out_dirs, dry_run)

    # FANG prerequisites
    if not grb_only:
        ok = await run_fang_index_vanilla(dry_run)
        if ok:
            completed.add(FANG_INDEX_JOB)
        else:
            log("FATAL: FANG index-vanilla failed — graph_first jobs will stall")

        ok = await run_fang_index_bc(dry_run)
        if ok:
            completed.add(FANG_BC_INDEX_JOB)
        else:
            log("FATAL: FANG index-bc failed — bc_laned60 jobs will stall")

    tasks = [
        asyncio.create_task(run_job(job, global_sem, rerank_sem, provider_sems, eval_sem, completed, dry_run))
        for job in jobs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = sum(1 for r in results if r is True)
    fail = sum(1 for r in results if r is False or isinstance(r, Exception))
    log(f"DONE: {ok} succeeded, {fail} failed")

    # final report
    report_cmd = [PY, "demo/graphrag_bench.py", "report", "--out-dir", GRB_OUT]
    if not dry_run:
        await run_cmd(report_cmd, "report")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument("--grb-only", action="store_true", help="Run GraphRAG-Bench jobs only")
    ap.add_argument("--fang-only", action="store_true", help="Run FANG-2026 jobs only")
    args = ap.parse_args()
    asyncio.run(main(args.grb_only, args.fang_only, args.dry_run))
