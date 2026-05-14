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
import re
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
PY = "/root/miniforge/envs/chonk/bin/python"
REPO = Path(__file__).parent.parent  # repo root

GRB_OUT = str(REPO / "work")
GRB_QIDs = str(REPO / "work/data/full_corpus_stratified_order.json")
GRB_CONFIGS = REPO / "work/configs/runs"

FANG_OUT = str(REPO / "work/fang2026")
FANG_QIDs = str(REPO / "work/fang2026/data/fang2026_question_ids.json")
FANG_DB = "chonk_nobc_1100_2200.duckdb"
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
    depends_on: list[str] = field(default_factory=list)  # job names that must complete first

    @property
    def gen_file(self) -> Path:
        return Path(self.out_dir) / "results" / f"{self.name}.jsonl"

    @property
    def eval_file(self) -> Path:
        return Path(self.out_dir) / "results" / f"bench_eval_{self.name}_rp.json"

    def is_done(self) -> bool:
        return self.eval_file.exists()

    def gen_done(self) -> bool:
        return self.gen_file.exists()


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
    name: str, flags: list[str], *, rerank: bool, provider: str, depends_on: list[str] | None = None
) -> Job:
    run_flags = flags + ["--run-name", name, "--question-ids", FANG_QIDs]
    return Job(
        name=name,
        run_flags=run_flags,
        eval_flags=FANG_EVAL,
        out_dir=FANG_OUT,
        uses_rerank=rerank,
        provider=provider,
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


def build_fang_jobs() -> list[Job]:
    db = ["--db-name", FANG_DB]
    vdb = ["--db-name", FANG_VANILLA_DB]
    gf_deps = [FANG_INDEX_JOB]

    return [
        # ── gpt-4o-mini ──────────────────────────────────────────────────────
        _fang(
            "fang_vanilla_rerank_mini",
            ["--vanilla", "--rerank"] + MINI,
            rerank=True,
            provider="openai",
        ),
        _fang(
            "fang_vanilla_rerank_srr_mini",
            ["--vanilla", "--rerank", "--srr"] + MINI,
            rerank=True,
            provider="openai",
        ),
        _fang(
            "fang_ner_ref_laned60_community_k10_mini",
            ["--rerank"] + LANED60 + db + MINI,
            rerank=True,
            provider="openai",
        ),
        _fang(
            "fang_ner_ref_laned60_community_k10_srr_mini",
            ["--rerank", "--srr"] + LANED60 + db + MINI,
            rerank=True,
            provider="openai",
        ),
        _fang(
            "fang_ner_ref_cluster_community_k10_mini",
            ["--rerank"] + CLUSTER + db + MINI,
            rerank=True,
            provider="openai",
        ),
        _fang(
            "fang_ner_ref_cluster_community_k10_srr_mini",
            ["--rerank", "--srr"] + CLUSTER + db + MINI,
            rerank=True,
            provider="openai",
        ),
        _fang(
            "fang_ner_ref_graph_first_k10_mini",
            GF + vdb + MINI,
            rerank=False,
            provider="openai",
            depends_on=gf_deps,
        ),
        _fang(
            "fang_ner_ref_graph_first_k10_srr_mini",
            GF + ["--srr"] + vdb + MINI,
            rerank=False,
            provider="openai",
            depends_on=gf_deps,
        ),
        # ── claude-haiku-4-5 ─────────────────────────────────────────────────
        _fang(
            "fang_vanilla_rerank_haiku",
            ["--vanilla", "--rerank"] + HAIKU,
            rerank=True,
            provider="anthropic",
        ),
        _fang(
            "fang_vanilla_rerank_srr_haiku",
            ["--vanilla", "--rerank", "--srr"] + HAIKU,
            rerank=True,
            provider="anthropic",
        ),
        _fang(
            "fang_ner_ref_laned60_community_k10_haiku",
            ["--rerank"] + LANED60 + db + HAIKU,
            rerank=True,
            provider="anthropic",
        ),
        _fang(
            "fang_ner_ref_laned60_community_k10_srr_haiku",
            ["--rerank", "--srr"] + LANED60 + db + HAIKU,
            rerank=True,
            provider="anthropic",
        ),
        _fang(
            "fang_ner_ref_cluster_community_k10_haiku",
            ["--rerank"] + CLUSTER + db + HAIKU,
            rerank=True,
            provider="anthropic",
        ),
        _fang(
            "fang_ner_ref_cluster_community_k10_srr_haiku",
            ["--rerank", "--srr"] + CLUSTER + db + HAIKU,
            rerank=True,
            provider="anthropic",
        ),
        _fang(
            "fang_ner_ref_graph_first_k10_haiku",
            GF + vdb + HAIKU,
            rerank=False,
            provider="anthropic",
            depends_on=gf_deps,
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

    sems = [global_sem, provider_sems[job.provider]]
    if job.uses_rerank:
        sems.append(rerank_sem)

    async with sems[0]:
        async with sems[1]:
            if len(sems) > 2:
                async with sems[2]:
                    return await _execute_job(job, eval_sem, completed, dry_run)
            else:
                return await _execute_job(job, eval_sem, completed, dry_run)


async def _execute_job(job: Job, eval_sem: asyncio.Semaphore, completed: set[str], dry_run: bool) -> bool:
    base_run = [PY, "demo/graphrag_bench.py", "run", "--out-dir", job.out_dir]
    base_eval = [PY, "demo/graphrag_bench.py", "eval", "--out-dir", job.out_dir]

    if not job.gen_done():
        cmd = base_run + job.run_flags
        if dry_run:
            log(f"DRY-RUN GEN {job.name}: {' '.join(cmd)}")
        else:
            rc = await run_cmd(cmd, f"GEN {job.name}")
            if rc != 0:
                log(f"ERROR GEN {job.name} — skipping eval")
                return False

    if not job.gen_done() and not dry_run:
        log(f"SKIP EVAL {job.name} — no gen output")
        return False

    # copy gen → rp if needed
    rp_src = job.gen_file
    rp_dst = Path(job.out_dir) / "results" / f"{job.name}_rp.jsonl"
    if dry_run:
        log(f"DRY-RUN COPY {rp_src.name} → {rp_dst.name}")
    elif rp_src.exists() and not rp_dst.exists():
        shutil.copy(str(rp_src), str(rp_dst))

    eval_cmd = base_eval + ["--run-name", f"{job.name}_rp"] + job.eval_flags
    if dry_run:
        log(f"DRY-RUN EVAL {job.name}_rp: {' '.join(eval_cmd)}")
    else:
        async with eval_sem:
            rc = await run_cmd(eval_cmd, f"EVAL {job.name}_rp")
        if rc != 0:
            log(f"ERROR EVAL {job.name}_rp")
            return False

    completed.add(job.name)
    return True


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
    eval_sem = asyncio.Semaphore(1)  # all evals share gpt-4o-mini judge; serialize to avoid RPM compounding
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

    # FANG prerequisite: index-vanilla
    if not grb_only:
        ok = await run_fang_index_vanilla(dry_run)
        if ok:
            completed.add(FANG_INDEX_JOB)
        else:
            log("FATAL: FANG index-vanilla failed — graph_first jobs will stall")

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
