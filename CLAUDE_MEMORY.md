# Project Memory

## User
Kenneth Stott, Member of Technical Staff & Senior Adviser at Logick. This project is his Logick research.

## Behavioral Rules
- No unsolicited advice or opinions — report only what was asked, never nudge toward decisions
- Answer the question asked — no piggybacking extra actions onto simple questions
- No parallel embedding/reranking — too memory intensive, always run sequentially

## GPU
- IP: 140.82.47.26
- SSH key: `work/gpu/vultr_key`
- Python: `/root/miniforge/envs/chonk/bin/python`
- Working dir: `/root/chunkymonkey`

## Session Start Checklist
1. Sync results from GPU:
   ```bash
   rsync -az --include="bench_eval_*.json" --include="*_flags.json" --include="*.jsonl" --exclude="*" \
     -e "ssh -i work/gpu/vultr_key -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
     root@140.82.47.26:/root/chunkymonkey/work/results/ work/results/
   python work/update_runs_csv.py
   ```
2. Check if runs are still active: `ssh ... root@140.82.47.26 "tmux capture-pane -t runs -p | tail -20"`

## Reattach to Running GPU Session
```bash
ssh -i work/gpu/vultr_key -o StrictHostKeyChecking=no -o IdentitiesOnly=yes root@140.82.47.26
tmux attach -t runs
```

## Launch Runs
```bash
# On GPU, in tmux:
tmux new -s runs
cd /root/chunkymonkey
PY=/root/miniforge/envs/chonk/bin/python bash work/run_full_all.sh
# detach: Ctrl-b d
```

## Workflow Rules
- All evals use JR (judge reprompt) — `_rp` suffix is the canonical score, plain evals are unreliable
- After every GPU sync or manifest change: `python work/update_runs_csv.py` (rsyncs CSV to GPU automatically)
- `work/run_full_all.sh` is the single authoritative script for all full-corpus runs
- `work/run_manifest.jsonl` is the feature-flag database — append new runs before launching

## Experiment Plan
- **Phase 1** (complete): gpt-4o-mini full runs, all configs, 4072Q
- **Phase 2** (in progress): gpt-4o on best config (laned60+community+k30) + vanilla — awaiting eval
- **Phase 3** (conditional on Phase 2): gpt-4o full grid+funnel if model effect confirmed
- **Phase 4** (conditional on Phase 3): judge isolation (gpt-4o judge vs gpt-4o-mini judge)

## Key Findings to Date
- Rerank is the dominant feature; full NER+community pipeline adds ~0.002 over rerank-alone on full corpus
- Grid scores don't transfer to full corpus (rank #1 grid dropped 0.032 on full)
- Lane sim optimum ~0.55–0.60; use large k and let sim govern rather than tuning k independently
- Pruning consistently hurts regardless of sim threshold
- NaN bias ~0.010 inflation in original evals; _rp runs are corrected scores
- Generator model (gpt-4o) is the key open question — a stronger model may extract value from NER/community features that gpt-4o-mini cannot
