#!/usr/bin/env bash
# Restore GPU server from snapshot, run generation phase, rsync results back, destroy.
#
# Usage: VULTR_API_KEY=<key> OPENAI_API_KEY=<key> bash work/gpu/run_gpu.sh
# Reads:  work/gpu/snapshot_id.txt, work/gpu/ssh_key_id.txt, work/gpu/vultr_key
# Writes: work/results/*.jsonl  (rsynced back from GPU)
set -euo pipefail

API="https://api.vultr.com/v2"
KEY="${VULTR_API_KEY:?VULTR_API_KEY not set}"
OPENAI_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY not set}"
GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$GPU_DIR/../.." && pwd)"

SNAPSHOT_ID=$(cat "$GPU_DIR/snapshot_id.txt")
SSH_KEY_ID=$(cat "$GPU_DIR/ssh_key_id.txt")
SSH_KEY_FILE="$GPU_DIR/vultr_key"
INSTANCE_ID_FILE="$GPU_DIR/instance_id.txt"

REGION="ewr"
PLAN="vcg-a16-3c-32g-8vram"
LABEL="chunkymonkey-gpu-run"
PY_ENV="/root/miniforge/envs/chonk/bin/python"

vtapi() {
    local method=$1 path=$2; shift 2
    local response http_code body
    response=$(curl -s -w "\n__HTTP_CODE__:%{http_code}" -X "$method" \
        -H "Authorization: Bearer $KEY" \
        -H "Content-Type: application/json" \
        "$@" "${API}${path}")
    http_code=$(echo "$response" | grep '__HTTP_CODE__:' | cut -d: -f2)
    body=$(echo "$response" | grep -v '__HTTP_CODE__:')
    if [[ "$http_code" -ge 400 ]]; then
        echo "Vultr API error $http_code: $body" >&2; return 1
    fi
    echo "$body"
}

# ── 1. Restore from snapshot ───────────────────────────────────────────────────
echo "=== Restoring from snapshot $SNAPSHOT_ID ==="
INSTANCE_ID=$(vtapi POST "/instances" -d "{
    \"region\": \"$REGION\",
    \"plan\": \"$PLAN\",
    \"snapshot_id\": \"$SNAPSHOT_ID\",
    \"label\": \"$LABEL\",
    \"sshkey_id\": [\"$SSH_KEY_ID\"],
    \"backups\": \"disabled\"
}" | python3 -c "import sys,json; print(json.load(sys.stdin)['instance']['id'])")
echo "$INSTANCE_ID" > "$INSTANCE_ID_FILE"
echo "  instance_id: $INSTANCE_ID"

# ── 2. Wait for instance ready ─────────────────────────────────────────────────
echo "=== Waiting for instance ==="
echo -n "  power_status=running"
for _ in $(seq 1 80); do
    status=$(vtapi GET "/instances/$INSTANCE_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['instance']['power_status'])" 2>/dev/null || echo "")
    [ "$status" = "running" ] && { echo " done"; break; }
    echo -n "[$status]"; sleep 15
done

echo -n "  server_status=ok"
for _ in $(seq 1 80); do
    status=$(vtapi GET "/instances/$INSTANCE_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['instance']['server_status'])" 2>/dev/null || echo "")
    [ "$status" = "ok" ] && { echo " done"; break; }
    echo -n "[$status]"; sleep 15
done

MAIN_IP=$(vtapi GET "/instances/$INSTANCE_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['instance']['main_ip'])")
echo "  IP: $MAIN_IP"

SSH="ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o ConnectTimeout=10 root@$MAIN_IP"

echo -n "  Waiting for SSH"
for _ in $(seq 1 40); do
    if $SSH "echo ok" &>/dev/null; then echo " ready"; break; fi
    echo -n "."; sleep 10
done

# ── 3. Sync project + data ─────────────────────────────────────────────────────
echo "=== Syncing project ==="
rsync -az \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='work/data/runs/' --exclude='work/results/' --exclude='work/logs/' \
    -e "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
    "$PROJECT_ROOT/" root@"$MAIN_IP":/root/chunkymonkey/

echo "=== Syncing data files ==="
rsync -az \
    -e "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
    "$PROJECT_ROOT/work/data/chunkymonkey_nobc_1100_2200.duckdb" \
    "$PROJECT_ROOT/work/data/full_corpus_stratified_order.json" \
    "$PROJECT_ROOT/work/data/question_embeddings.npy" \
    "$PROJECT_ROOT/work/data/medical_questions.jsonl" \
    "$PROJECT_ROOT/work/data/novel_questions.jsonl" \
    root@"$MAIN_IP":/root/chunkymonkey/work/data/

# Sync existing results so GPU skips already-generated runs
echo "=== Syncing existing results ==="
rsync -az \
    -e "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
    "$PROJECT_ROOT/work/results/" \
    root@"$MAIN_IP":/root/chunkymonkey/work/results/

# ── 4. Install/update package ──────────────────────────────────────────────────
echo "=== Installing chunkymonkey ==="
$SSH "$PY_ENV -m pip install -q -e '/root/chunkymonkey[dev]'"

# ── 5. Run gen + eval (with periodic local rsync every 10 min) ────────────────
echo "=== Running gen + eval ==="

_sync_results() {
    rsync -az --quiet \
        -e "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o ConnectTimeout=10" \
        root@"$MAIN_IP":/root/chunkymonkey/work/results/ "$PROJECT_ROOT/work/results/" 2>/dev/null
    rsync -az --quiet \
        -e "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o ConnectTimeout=10" \
        root@"$MAIN_IP":/root/chunkymonkey/work/data/runs/ "$PROJECT_ROOT/work/data/runs/" 2>/dev/null
}

_periodic_sync() {
    while kill -0 "$1" 2>/dev/null; do
        sleep 600
        echo "=== Periodic sync ===" >&2
        _sync_results
    done
}

$SSH "cd /root/chunkymonkey && mkdir -p work/logs && \
    PATH=/root/miniforge/envs/chonk/bin:\$PATH \
    OPENAI_API_KEY='$OPENAI_KEY' \
    EMBED_DEVICE=cuda \
    RERANKER_DEVICE=cuda \
    bash work/run_full_all.sh 2>&1 | tee work/logs/gpu_run.log" &
_SSH_PID=$!
_periodic_sync $_SSH_PID &
_SYNC_PID=$!
wait $_SSH_PID
kill $_SYNC_PID 2>/dev/null

# ── 6. Rsync results back ──────────────────────────────────────────────────────
echo "=== Syncing results back ==="
rsync -az \
    -e "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
    root@"$MAIN_IP":/root/chunkymonkey/work/results/ \
    "$PROJECT_ROOT/work/results/"

rsync -az \
    -e "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
    root@"$MAIN_IP":/root/chunkymonkey/work/data/runs/ \
    "$PROJECT_ROOT/work/data/runs/"

rsync -az \
    -e "ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
    root@"$MAIN_IP":/root/chunkymonkey/work/logs/ \
    "$PROJECT_ROOT/work/logs/"

# ── 7. Destroy instance ────────────────────────────────────────────────────────
echo "=== Destroying instance ==="
vtapi DELETE "/instances/$INSTANCE_ID" > /dev/null
rm -f "$INSTANCE_ID_FILE"
echo "  Instance destroyed."

echo ""
echo "=== DONE: results and run DBs synced to work/ ==="
