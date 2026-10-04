#!/usr/bin/env bash
# Reproduce the HARE-Bench results from the published index.
#
# This is the reproduction method. It downloads the published index stores and
# embedding caches from the artifact bucket, verifies each file against the
# bucket's manifest.json, generates and scores answers for every run config, and
# compares the scores with the reported ones. It does no chunking or indexing: the index is a
# published artifact (see scripts/hare/02_publish_index.sh).
#
# Usage:
#   scripts/hare/reproduce.sh [--config-dir DIR] [--runs FILE] [--out-dir DIR]
#                             [--compare-retrieval] [--gpu gcp|vultr]
#
#   --config-dir          run configs to execute (default: work/configs/fang, all runs)
#   --runs FILE           only the run names listed in FILE, one per line
#                         (work/configs/hare_bench_paper_runs.txt: the runs the paper cites)
#   --out-dir             working directory (default: work/reproduce)
#   --compare-retrieval   also download the published reference run outputs
#                         (diagnostics/) and compare retrieval question by question
#   --gpu PROVIDER        generate and score on a GPU instance at gcp or vultr (see below)
#   --no-compare          stop after scoring; --gpu uses this on the remote host
#
# The reranker runs on CPU by default on Apple silicon; RERANKER_DEVICE=mps (or
# cuda) runs it on the GPU. The device used is recorded in <run>_flags.json.
#
# --gpu copies the repository and any results already in --out-dir to a new
# instance, runs this script there with RERANKER_DEVICE=cuda, copies the results
# back every minute, and destroys the instance when the run ends (also on
# failure). Finished runs are skipped, so an interrupted --gpu run can be
# repeated. The model API keys and HARE_PUBLIC_URL are passed to the remote
# process environment; they are not written to its disk. Needs HARE_PUBLIC_URL
# (environment or .env), ssh, and rsync, plus what the provider's helper needs:
# scripts/hare/gcp_gpu.sh (GCP_PROJECT, gcloud) or scripts/hare/vultr_gpu.sh
# (VULTR_API_KEY, curl, sshpass). Only gcp sets a deadline at the provider, so
# that the instance is deleted even if this machine dies; it also asks before
# creating anything. --config-dir, --runs, and --out-dir must be relative paths
# inside the repository.
#
# Needs, in the environment or .env: OPENAI_API_KEY (generator, ADF classifier)
# and HARE_PUBLIC_URL (public bucket URL; see .env.example), or HARE_BUCKET_URL
# with AWS_* credentials to download through the S3 API. Runs that use Claude or
# Together models also need ANTHROPIC_API_KEY or TOGETHER_API_KEY.
# Each 500-question run with gpt-4o-mini costs roughly US$5-10.
set -euo pipefail

CONFIG_DIR="work/configs/fang"
RUNS=""
OUT="work/reproduce"
COMPARE_RETRIEVAL=0
COMPARE=1
GPU=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config-dir)        CONFIG_DIR="$2"; shift ;;
        --runs)              RUNS="$2"; shift ;;
        --out-dir)           OUT="$2"; shift ;;
        --compare-retrieval) COMPARE_RETRIEVAL=1 ;;
        --no-compare)        COMPARE=0 ;;
        --gpu)               GPU="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

cd "$(dirname "$0")/../.."
PY="${PY:-uv run python}"
mkdir -p "$OUT/data"
if [ "$(cd "$OUT" && pwd -P)" = "$(cd work/fang2026 && pwd -P)" ]; then
    echo "--out-dir must not be work/fang2026: its committed results would make every run skip" >&2
    exit 2
fi
if [ -z "${OPENAI_API_KEY:-}" ] && ! grep -q '^OPENAI_API_KEY=.' .env 2>/dev/null; then
    echo "OPENAI_API_KEY is not set (environment or .env)" >&2
    exit 1
fi

compare() {
    echo "=== Reported vs reproduced ==="
    if [ "$COMPARE_RETRIEVAL" -eq 1 ]; then
        $PY scripts/hare/artifacts.py download diagnostics --out-dir "$OUT/reference"
        $PY scripts/hare/compare_results.py --reported work/fang2026/results \
            --reproduced "$OUT/results" --retrieval "$OUT/reference"
    else
        $PY scripts/hare/compare_results.py --reported work/fang2026/results --reproduced "$OUT/results"
    fi
}

run_on_gpu() {
    local path name value code lost=0
    for path in "$CONFIG_DIR" "$OUT" "$RUNS"; do
        case "$path" in
            /*|..|../*|*/..|*/../*)
                echo "--gpu needs --config-dir, --runs, and --out-dir inside the repository: $path" >&2
                exit 2 ;;
        esac
    done
    case "$GPU" in
        gcp|vultr) ;;
        *) echo "--gpu takes gcp or vultr, not: $GPU" >&2; exit 2 ;;
    esac
    # shellcheck source=scripts/hare/gcp_gpu.sh
    source "scripts/hare/${GPU}_gpu.sh"
    HARE_PUBLIC_URL=$(env_or_dotenv HARE_PUBLIC_URL)
    : "${HARE_PUBLIC_URL:?HARE_PUBLIC_URL not set (environment or .env); --gpu downloads the index through it}"
    # Passed in the remote process environment only: the remote host gets no .env.
    REMOTE_ENV="export RERANKER_DEVICE=cuda HARE_PUBLIC_URL=$(printf %q "$HARE_PUBLIC_URL")"
    for name in OPENAI_API_KEY ANTHROPIC_API_KEY TOGETHER_API_KEY; do
        value=$(env_or_dotenv "$name")
        if [ -n "$value" ]; then
            REMOTE_ENV+=" $name=$(printf %q "$value")"
        fi
    done

    gpu_up chonk-hare-reproduce
    copy_back() {
        rsync -az -e "$RSYNC_SSH" --exclude data "$REMOTE:$REMOTE_DIR/$OUT/" "$OUT/"
    }
    # Results cost API fees: save whatever exists before the instance goes away.
    trap 'copy_back || echo "could not copy results back from $REMOTE" >&2; gpu_destroy' EXIT

    echo "=== Copying repository ==="
    rsync -az --delete -e "$RSYNC_SSH" \
        --exclude .git --exclude .venv --exclude __pycache__ --exclude .env \
        --exclude work/corpus --exclude 'work/fang2026/data/*.duckdb*' --exclude work/gpu \
        --exclude /training --exclude "/$OUT/data" \
        ./ "$REMOTE:$REMOTE_DIR/"

    echo "=== Starting the run on the GPU ==="
    "${SSH[@]}" bash -s <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# A fresh image may still be installing its NVIDIA driver and holding the apt lock.
for _ in \$(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$SUDO apt-get -o DPkg::Lock::Timeout=600 update -qq
$SUDO apt-get -o DPkg::Lock::Timeout=600 install -y -qq rsync curl build-essential >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
cd $REMOTE_DIR
uv sync --all-extras --group dev --frozen
uv run python -m spacy download en_core_web_sm
uv run python -m nltk.downloader -q wordnet
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"
$REMOTE_ENV
mkdir -p $OUT
# Detached, so a dropped connection does not stop a run that takes hours.
nohup bash -c 'scripts/hare/reproduce.sh --no-compare --config-dir $CONFIG_DIR ${RUNS:+--runs $RUNS} --out-dir $OUT; echo \$? > $REMOTE_DIR.exit' \
    > $OUT/gpu.log 2>&1 < /dev/null &
REMOTE

    echo "=== Generating and scoring on the GPU (log: $OUT/gpu.log) ==="
    until "${SSH[@]}" "test -f $REMOTE_DIR.exit"; do
        sleep 60
        if copy_back; then
            lost=0
            tail -n 1 "$OUT/gpu.log"
        else
            lost=$((lost + 1))
        fi
        if [ "$lost" -ge 30 ]; then
            echo "no contact with $REMOTE for 30 minutes" >&2
            exit 1
        fi
    done
    copy_back
    code=$("${SSH[@]}" "cat $REMOTE_DIR.exit")
    trap - EXIT
    gpu_destroy
    if [ "$code" != 0 ]; then
        echo "the run failed on the GPU host (exit $code); see $OUT/gpu.log" >&2
        exit 1
    fi
}

if [ -n "$GPU" ]; then
    run_on_gpu
    compare
    exit 0
fi

echo "=== Checking the environment ==="
$PY - <<'PYEOF'
import importlib, sys
missing = []
for mod in ("openai", "langchain_openai", "quantulum3", "boto3", "spacy"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit(f"missing: {', '.join(missing)} -- run: uv sync --all-extras")
import spacy
try:
    spacy.load("en_core_web_sm")
except OSError:
    sys.exit("spaCy model missing -- run: uv run python -m spacy download en_core_web_sm")
from nltk.corpus import wordnet
try:
    wordnet.ensure_loaded()
except LookupError:
    sys.exit("WordNet data missing -- run: uv run python -m nltk.downloader wordnet")
print("ok")
PYEOF

echo "=== Downloading and verifying the published index ==="
$PY scripts/hare/artifacts.py download index --out-dir "$OUT/data"
cp work/fang2026/data/fang2026_questions.jsonl work/fang2026/data/fang2026_gold_schemas.jsonl "$OUT/data/"

echo "=== Preparing the scorer ==="
# The evaluator loads the typed scorer from the data directory and imports its
# metric helpers from the GraphRAG-Bench repository. HARE-Bench scores come from
# the typed scorer alone, so the revision of that repository does not affect them.
cp work/score_typed.py "$OUT/data/"
if [ ! -d "$OUT/GraphRAG-Benchmark" ]; then
    git clone --quiet --depth=1 https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git "$OUT/GraphRAG-Benchmark"
fi

echo "=== Generating and scoring ==="
$PY demo/graphrag_bench.py run-all --config-dir "$CONFIG_DIR" ${RUNS:+--runs "$RUNS"} --out-dir "$OUT"

if [ "$COMPARE" -eq 1 ]; then
    compare
fi
