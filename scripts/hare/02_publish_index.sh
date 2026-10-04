#!/usr/bin/env bash
# Initialization, stage 2: build the HARE-Bench index stores from the corpus in
# the artifact bucket and publish them back to it.
#
# By default the index is built on this machine. --gpu builds it on a Vultr GPU
# instance instead: it copies the repository there, installs the locked
# environment (uv.lock), runs scripts/hare/build_index.sh, copies the stores
# back, and destroys the instance (also on failure). Publishing always runs
# locally, so bucket credentials never leave this machine.
#
# Embeddings differ slightly across hardware (CPU, Apple MPS, CUDA), so the
# build environment is recorded in index_build_info.json (scripts/hare/build_info.py)
# and published with the stores. The published stores are the reference;
# reproduce.sh uses them as-is.
#
# Usage:
#   scripts/hare/02_publish_index.sh
#   VULTR_API_KEY=... scripts/hare/02_publish_index.sh --gpu
#
# Configuration (environment or .env): HARE_BUCKET_URL, AWS_ACCESS_KEY_ID,
# AWS_SECRET_ACCESS_KEY, optional AWS_ENDPOINT_OVERRIDE / AWS_REGION.
# With --gpu also VULTR_API_KEY, and curl, ssh, rsync, sshpass.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-uv run python}"
OUT="work/fang2026"

if [ "${1:-}" != "--gpu" ]; then
    $PY scripts/hare/artifacts.py download corpus
    scripts/hare/build_index.sh "$OUT"
    $PY scripts/hare/build_info.py local
    $PY scripts/hare/artifacts.py publish index
    exit 0
fi

# The remote host gets no .env (no secrets leave this machine); it downloads the
# corpus over the public URL.
# shellcheck source=scripts/hare/vultr_gpu.sh
source scripts/hare/vultr_gpu.sh
HARE_PUBLIC_URL=$(env_or_dotenv HARE_PUBLIC_URL)
: "${HARE_PUBLIC_URL:?HARE_PUBLIC_URL not set (environment or .env); --gpu downloads the corpus through it}"

gpu_up chonk-hare-index

echo "=== Copying repository ==="
rsync -az --delete -e "$RSYNC_SSH" \
    --exclude .git --exclude .venv --exclude __pycache__ --exclude .env \
    --exclude work/corpus --exclude 'work/fang2026/data/*.duckdb*' --exclude work/gpu \
    ./ "root@$IP:$REMOTE_DIR/"

echo "=== Building index on GPU ==="
"${SSH[@]}" bash -s <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq rsync curl build-essential >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
cd $REMOTE_DIR
uv sync --all-extras --group dev --frozen
uv run python -m spacy download en_core_web_sm
uv run python -m nltk.downloader -q wordnet
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"
HARE_PUBLIC_URL="$HARE_PUBLIC_URL" uv run python scripts/hare/artifacts.py download corpus
scripts/hare/build_index.sh $OUT
uv run python scripts/hare/build_info.py "vultr $PLAN"
REMOTE

echo "=== Copying index back ==="
rsync -az -e "$RSYNC_SSH" "root@$IP:$REMOTE_DIR/$OUT/data/" "$OUT/data/"

$PY scripts/hare/artifacts.py publish index
