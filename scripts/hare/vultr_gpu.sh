# Vultr GPU instance helpers. Sourced by 02_publish_index.sh and reproduce.sh
# from the repository root; not executable on its own.
#
# gpu_up LABEL creates an instance, waits for it to boot, installs an SSH key,
# and sets IP, REMOTE (user@host), SSH (a command array), and RSYNC_SSH. The
# instance is destroyed when the calling script exits, also on failure. Vultr has
# no provider-side deadline: if this machine dies first, the instance keeps
# billing until it is destroyed by hand.
#
# Needs VULTR_API_KEY (environment or .env), and curl, ssh, rsync, sshpass.
# Optional: VULTR_REGION, VULTR_PLAN.

API="https://api.vultr.com/v2"
REGION="${VULTR_REGION:-ewr}"
PLAN="${VULTR_PLAN:-vcg-a16-3c-32g-8vram}"   # 8 GB VRAM: embedder, NER, and reranker fit
OS_ID=2284                                    # Ubuntu 24.04 LTS x64 (NVIDIA driver preinstalled)
SSH_KEY_FILE="work/gpu/hare_key"
REMOTE_DIR="/root/chonk"
SUDO=""                                       # the remote user is root

# shellcheck source=scripts/hare/dotenv.sh
source scripts/hare/dotenv.sh

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

gpu_destroy() {
    echo "=== Destroying instance $INSTANCE_ID ==="
    vtapi DELETE "/instances/$INSTANCE_ID" >/dev/null
}

gpu_up() {
    local label=$1
    KEY=$(env_or_dotenv VULTR_API_KEY)
    : "${KEY:?VULTR_API_KEY not set (environment or .env)}"

    mkdir -p work/gpu
    [ -f "$SSH_KEY_FILE" ] || ssh-keygen -q -t ed25519 -N "" -f "$SSH_KEY_FILE" -C "chonk-hare-gpu"
    PUBKEY=$(cat "$SSH_KEY_FILE.pub")

    echo "=== Creating $PLAN instance in $REGION ==="
    INSTANCE_ID=$(vtapi POST /instances -d "{\"region\":\"$REGION\",\"plan\":\"$PLAN\",\"os_id\":$OS_ID,\"label\":\"$label\",\"backups\":\"disabled\"}" | field id)
    trap gpu_destroy EXIT

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
    REMOTE="root@$IP"
    SSH=(ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes "$REMOTE")
    RSYNC_SSH="ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
}
