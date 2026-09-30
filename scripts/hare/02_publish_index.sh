#!/usr/bin/env bash
# Maintainer stage 2: build the HARE-Bench index stores from the published
# corpus and publish them to R2.
#
# By default the index is built on this machine. --gpu builds it on a Vultr GPU
# instance instead: it copies the repository there, installs the locked
# environment (uv.lock), runs scripts/hare/build_index.sh, copies the stores
# back, and destroys the instance (also on failure). Publishing always runs
# locally, so R2 credentials never leave this machine.
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
# Needs: rclone (publish); with --gpu also curl, ssh, rsync, sshpass.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-uv run python}"
OUT="work/fang2026"

if [ "${1:-}" != "--gpu" ]; then
    $PY scripts/hare/artifacts.py download corpus
    scripts/hare/build_index.sh "$OUT"
    $PY scripts/hare/build_info.py local
    $PY scripts/hare/artifacts.py publish index
    echo "Commit work/fang2026/artifact_manifest.json."
    exit 0
fi

KEY="${VULTR_API_KEY:?VULTR_API_KEY not set}"
API="https://api.vultr.com/v2"
REGION="${VULTR_REGION:-ewr}"
PLAN="${VULTR_PLAN:-vcg-a16-3c-32g-8vram}"   # 8 GB VRAM: embedder and NER fit
OS_ID=2284                                    # Ubuntu 24.04 LTS x64 (NVIDIA driver preinstalled)
SSH_KEY_FILE="work/gpu/hare_key"
REMOTE_DIR="/root/chonk"

vtapi() {
    local method=$1 path=$2; shift 2
    local response code
    response=$(curl -s -w "\n%{http_code}" -X "$method" \
        -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        "$@" "${API}${path}")
    code=$(tail -n1 <<<"$response")
    if [ "$code" -ge 400 ]; then
        echo "Vultr API error $code: $(sed '$d' <<<"$response")" >&2
        return 1
    fi
    sed '$d' <<<"$response"
}
field() { python3 -c "import sys,json; print(json.load(sys.stdin)['instance']['$1'])"; }

mkdir -p work/gpu
[ -f "$SSH_KEY_FILE" ] || ssh-keygen -q -t ed25519 -N "" -f "$SSH_KEY_FILE" -C "chonk-hare-gpu"
PUBKEY=$(cat "$SSH_KEY_FILE.pub")

echo "=== Creating $PLAN instance in $REGION ==="
INSTANCE_ID=$(vtapi POST /instances -d "{\"region\":\"$REGION\",\"plan\":\"$PLAN\",\"os_id\":$OS_ID,\"label\":\"chonk-hare-index\",\"backups\":\"disabled\"}" | field id)
destroy() {
    echo "=== Destroying instance $INSTANCE_ID ==="
    vtapi DELETE "/instances/$INSTANCE_ID" >/dev/null
}
trap destroy EXIT

for _ in $(seq 1 80); do
    INST=$(vtapi GET "/instances/$INSTANCE_ID")
    [ "$(field power_status <<<"$INST")" = running ] \
        && [ "$(field server_status <<<"$INST")" = ok ] \
        && [ "$(field main_ip <<<"$INST")" != 0.0.0.0 ] && break
    sleep 15
done
IP=$(field main_ip <<<"$INST")
PASS=$(field default_password <<<"$INST")
echo "  $IP"

# Vultr's ssh-key injection is unreliable on Ubuntu 24.04: bootstrap with the
# root password, install our key, then use key auth only.
for _ in $(seq 1 40); do
    sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
        -o ConnectTimeout=10 "root@$IP" \
        "mkdir -p ~/.ssh && echo '$PUBKEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" \
        && break
    sleep 10
done
SSH=(ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes "root@$IP")
RSYNC_SSH="ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"

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
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"
uv run python scripts/hare/artifacts.py download corpus
scripts/hare/build_index.sh $OUT
uv run python scripts/hare/build_info.py "vultr $PLAN"
REMOTE

echo "=== Copying index back ==="
rsync -az -e "$RSYNC_SSH" "root@$IP:$REMOTE_DIR/$OUT/data/" "$OUT/data/"

$PY scripts/hare/artifacts.py publish index
echo "Commit work/fang2026/artifact_manifest.json."
