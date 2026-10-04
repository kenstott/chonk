# Google Compute Engine GPU instance helpers. Sourced by reproduce.sh from the
# repository root; not executable on its own.
#
# gpu_up LABEL (a name starting with chonk-hare-) creates one GPU instance and sets IP, REMOTE (user@host), SSH (a
# command array), and RSYNC_SSH. The instance is deleted when the calling script
# exits, also on failure. Nothing it creates can outlive the job:
#   - Compute Engine itself deletes the instance GCP_MAX_HOURS after it starts,
#     whatever happens to this machine (--max-run-duration, action DELETE);
#   - the boot disk is deleted with the instance, and no other disk, address,
#     or snapshot is created;
#   - instance and disk names start with chonk-hare-, and gpu_destroy fails
#     when anything with that prefix is still there afterwards.
# gpu_up states the worst-case cost and asks before it creates anything;
# HARE_GPU_YES=1 answers for a caller that has already agreed to it.
#
# Needs gcloud (authenticated) and GCP_PROJECT (environment or .env). The
# project needs a GPU quota (GPUS_ALL_REGIONS and the per-region GPU type) of at
# least 1. Optional: GCP_ZONES, GCP_MACHINE_TYPE, GCP_ACCELERATOR, GCP_MAX_HOURS,
# GCP_HOURLY_USD.

# shellcheck source=scripts/hare/dotenv.sh
source scripts/hare/dotenv.sh

# Tried in order; a zone that has no capacity for the GPU right now is skipped.
GCP_ZONES="${GCP_ZONES:-us-central1-b us-central1-c us-central1-f us-central1-a}"
GCP_MACHINE_TYPE="${GCP_MACHINE_TYPE:-n1-standard-8}"   # 8 vCPU, 30 GB
# 16 GB VRAM. Set GCP_ACCELERATOR= (empty) for machine types with a built-in GPU (g2-*).
GCP_ACCELERATOR="${GCP_ACCELERATOR-nvidia-tesla-t4}"
GCP_MAX_HOURS="${GCP_MAX_HOURS:-4}"
# Estimate for the default machine and GPU, used only for the worst-case figure
# shown before creation. Set it when you change either.
GCP_HOURLY_USD="${GCP_HOURLY_USD:-0.75}"
GCP_IMAGE_FAMILY="common-cu129-ubuntu-2404-nvidia-580"   # Deep Learning VM: NVIDIA driver included
GCP_IMAGE_PROJECT="deeplearning-platform-release"
NAME_PREFIX="chonk-hare-"
SSH_KEY_FILE="work/gpu/hare_key"
SSH_USER="hare"
REMOTE_DIR="/home/$SSH_USER/chonk"
SUDO="sudo"

# Instances and disks this script could have created that still exist. Fails,
# and says so, when there are any or when they cannot be listed.
gpu_leftovers() {
    local left
    if ! left=$(
        gcloud compute instances list --project "$GCP_PROJECT" --filter="name~^$NAME_PREFIX" \
            --format='value(name,zone.basename(),status)' \
        && gcloud compute disks list --project "$GCP_PROJECT" --filter="name~^$NAME_PREFIX" \
            --format='value(name,zone.basename(),sizeGb)'
    ); then
        echo "COULD NOT LIST resources in $GCP_PROJECT: check for $NAME_PREFIX* instances and disks by hand" >&2
        return 1
    fi
    if [ -n "$left" ]; then
        echo "STILL PRESENT in $GCP_PROJECT (billing until deleted):" >&2
        echo "$left" >&2
        return 1
    fi
    echo "  no $NAME_PREFIX* instance or disk remains in $GCP_PROJECT"
}

gpu_destroy() {
    echo "=== Deleting instance $INSTANCE ==="
    # The instance may already be gone: Compute Engine deletes it at its deadline.
    gcloud compute instances delete "$INSTANCE" --project "$GCP_PROJECT" --zone "$GCP_ZONE" --quiet \
        || echo "delete did not succeed; checking what remains" >&2
    gpu_leftovers
}

gpu_up() {
    local label=$1 answer output created=0
    GCP_PROJECT=$(env_or_dotenv GCP_PROJECT)
    : "${GCP_PROJECT:?GCP_PROJECT not set (environment or .env)}"
    case "$label" in
        "$NAME_PREFIX"*) ;;
        *) echo "instance label must start with $NAME_PREFIX: $label" >&2; exit 2 ;;
    esac
    INSTANCE="$label-$(date +%Y%m%d-%H%M%S)"

    echo "=== About to create $INSTANCE in $GCP_PROJECT (first of: $GCP_ZONES) ==="
    echo "  $GCP_MACHINE_TYPE${GCP_ACCELERATOR:+ + $GCP_ACCELERATOR}, about \$$GCP_HOURLY_USD/h (estimate)"
    echo "  Compute Engine deletes it after ${GCP_MAX_HOURS}h at the latest: worst case about" \
        "\$$(awk "BEGIN { printf \"%.2f\", $GCP_MAX_HOURS * $GCP_HOURLY_USD }")"
    if [ "${HARE_GPU_YES:-}" != 1 ]; then
        read -r -p "Type yes to create it: " answer || answer=""
        if [ "$answer" != yes ]; then
            echo "not created" >&2
            exit 1
        fi
    fi

    mkdir -p work/gpu
    [ -f "$SSH_KEY_FILE" ] || ssh-keygen -q -t ed25519 -N "" -f "$SSH_KEY_FILE" -C "chonk-hare-gpu"
    PUBKEY=$(cat "$SSH_KEY_FILE.pub")

    # Set before the create call: an interrupted create can still leave an instance.
    trap gpu_destroy EXIT
    for GCP_ZONE in $GCP_ZONES; do
        if output=$(gcloud compute instances create "$INSTANCE" --project "$GCP_PROJECT" --zone "$GCP_ZONE" \
            --machine-type "$GCP_MACHINE_TYPE" \
            ${GCP_ACCELERATOR:+--accelerator "type=$GCP_ACCELERATOR,count=1"} \
            --maintenance-policy TERMINATE \
            --max-run-duration "${GCP_MAX_HOURS}h" --instance-termination-action DELETE \
            --image-family "$GCP_IMAGE_FAMILY" --image-project "$GCP_IMAGE_PROJECT" \
            --boot-disk-size 100GB --boot-disk-type pd-balanced --boot-disk-auto-delete \
            --labels "chonk-hare=$label" \
            --metadata "enable-oslogin=FALSE,install-nvidia-driver=True,ssh-keys=$SSH_USER:$PUBKEY" \
            --no-service-account --no-scopes --quiet 2>&1); then
            echo "  created in $GCP_ZONE"
            created=1
            break
        fi
        if ! grep -q ZONE_RESOURCE_POOL_EXHAUSTED <<<"$output"; then
            echo "$output" >&2
            exit 1
        fi
        echo "  no capacity in $GCP_ZONE"
    done
    if [ "$created" -eq 0 ]; then
        echo "no GPU capacity in any of: $GCP_ZONES" >&2
        exit 1
    fi

    IP=$(gcloud compute instances describe "$INSTANCE" --project "$GCP_PROJECT" --zone "$GCP_ZONE" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
    echo "  $IP"
    REMOTE="$SSH_USER@$IP"
    # Compute Engine reuses addresses, so a host key seen before means nothing here.
    RSYNC_SSH="ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o IdentitiesOnly=yes -o ConnectTimeout=10"
    # shellcheck disable=SC2206  # RSYNC_SSH is a list of words by construction
    SSH=($RSYNC_SSH "$REMOTE")
    for _ in $(seq 1 40); do
        "${SSH[@]}" true && return 0
        sleep 10
    done
    echo "no SSH access to $REMOTE after 400 seconds" >&2
    exit 1
}
