#!/usr/bin/env bash
# =============================================================================
# AIPA Training Pipeline — Setup Script
# =============================================================================
# Compatible with bash 3.2+ (macOS default shell).
#
# Installs three Claude Code skills and shared ref files:
#
#   .claude/refs/                       — shared reference files
#     skill_clusters.md
#     modernization_manifest_schema.md
#     manifest_delta_schema.md
#
#   .claude/skills/
#     aipa-training-interviewer/SKILL.md
#     aipa-training-generator/SKILL.md
#     aipa-training-judge/SKILL.md
#
#   training/                           — runtime artifacts land here
#     methodology.md
#     skill_clusters.md
#     .gitignore
#
# Ref files are referenced from skills using project-root paths:
#   .claude/refs/skill_clusters.md
#
# Usage:
#   chmod +x setup_training.sh
#   ./setup_training.sh              # normal run
#   ./setup_training.sh --restart    # back up and clear all manifests for a fresh start
#
# All source .md files must be in the same directory as this script.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Parse flags
# -----------------------------------------------------------------------------
RESTART=false
for arg in "$@"; do
  case "$arg" in
    --restart) RESTART=true ;;
    *) warn "Unknown flag: $arg" ;;
  esac
done

# -----------------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------------
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}ℹ ${RESET}$*"; }
success() { echo -e "${GREEN}✓ ${RESET}$*"; }
warn()    { echo -e "${YELLOW}⚠ ${RESET}$*"; }
error()   { echo -e "${RED}✗ ${RESET}$*"; }
header()  { echo -e "\n${BOLD}$*${RESET}"; }

# -----------------------------------------------------------------------------
# Cluster name lookup — bash 3.2 compatible (no associative arrays)
# -----------------------------------------------------------------------------
cluster_name() {
  case "$1" in
    A) echo "Code Transformation" ;;
    B) echo "Documentation" ;;
    C) echo "Test Generation" ;;
    D) echo "Analysis & Audit" ;;
    E) echo "Data & SQL" ;;
    *) echo "Unknown" ;;
  esac
}

# -----------------------------------------------------------------------------
# Locate script directory and repo root
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local dir="$PWD"
  while [ "$dir" != "/" ]; do
    if [ -d "$dir/.git" ]; then
      echo "$dir"
      return
    fi
    dir="$(dirname "$dir")"
  done
  echo "$PWD"
}

REPO_ROOT="$(find_repo_root)"

# -----------------------------------------------------------------------------
# Target directories
# -----------------------------------------------------------------------------
CLAUDE_DIR="$REPO_ROOT/.claude"
REFS_DIR="$CLAUDE_DIR/refs"
SKILLS_DIR="$CLAUDE_DIR/skills"
TRAINING_DIR="$REPO_ROOT/training"

SKILL_INTERVIEWER="$SKILLS_DIR/aipa-training-interviewer"
SKILL_GENERATOR="$SKILLS_DIR/aipa-training-generator"
SKILL_JUDGE="$SKILLS_DIR/aipa-training-judge"

# -----------------------------------------------------------------------------
# Source files required
# -----------------------------------------------------------------------------
SKILL_FILES="skill_interviewer.md skill_generator.md skill_judge.md"
REF_FILES="skill_clusters.md modernization_manifest_schema.md manifest_delta_schema.md"
TRAINING_FILES="methodology.md skill_clusters.md"

ALL_SOURCE_FILES="$SKILL_FILES $REF_FILES methodology.md"

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}=================================================${RESET}"
echo -e "${BOLD}  AIPA Training Pipeline — Setup v2.1          ${RESET}"
echo -e "${BOLD}=================================================${RESET}"
echo ""
info "Script location : $SCRIPT_DIR"
info "Codebase root   : $REPO_ROOT"
info "Refs dir        : $REFS_DIR"
info "Skills dir      : $SKILLS_DIR"
info "Training dir    : $TRAINING_DIR"

# -----------------------------------------------------------------------------
# 1. Environment checks
# -----------------------------------------------------------------------------
header "1. Environment checks"

ENV_OK=true

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  success "ANTHROPIC_API_KEY is set"
else
  warn "ANTHROPIC_API_KEY is not set — Claude Code environment may not be active"
  ENV_OK=false
fi

if [ -n "${CLAUDE_WORKING_DIR:-}" ]; then
  success "CLAUDE_WORKING_DIR is set: $CLAUDE_WORKING_DIR"
else
  info "CLAUDE_WORKING_DIR not set — using detected repo root: $REPO_ROOT"
fi

if [ -d "$REPO_ROOT/.git" ]; then
  success "Git repository detected at $REPO_ROOT"
else
  warn "No git repository detected — deploying to current directory"
fi

FILE_COUNT=$(find "$REPO_ROOT" \
  -not -path "*/.git/*" \
  -not -path "*/.claude/*" \
  -not -path "*/node_modules/*" \
  -not -path "*/vendor/*" \
  -not -path "*/training/*" \
  -type f 2>/dev/null | wc -l | tr -d ' ')

info "Codebase file count: ~$FILE_COUNT files"

if [ "$FILE_COUNT" -gt 5000 ]; then
  warn "Large codebase ($FILE_COUNT files) — Generator may need to run module by module"
fi

# -----------------------------------------------------------------------------
# 2. Verify source files
# -----------------------------------------------------------------------------
header "2. Verifying source files"

MISSING=""
for f in $ALL_SOURCE_FILES; do
  if [ -f "$SCRIPT_DIR/$f" ]; then
    success "Found: $f"
  else
    error "Missing: $f"
    MISSING="$MISSING $f"
  fi
done

if [ -n "$MISSING" ]; then
  echo ""
  error "Missing files in $SCRIPT_DIR:"
  for f in $MISSING; do
    echo "  - $f"
  done
  error "Ensure all .md files are in the same directory as this script."
  exit 1
fi

# -----------------------------------------------------------------------------
# 3. Create directory structure
# -----------------------------------------------------------------------------
header "3. Creating directories"

for dir in "$CLAUDE_DIR" "$REFS_DIR" "$SKILLS_DIR" \
           "$SKILL_INTERVIEWER" "$SKILL_GENERATOR" "$SKILL_JUDGE" \
           "$TRAINING_DIR"; do
  if [ -d "$dir" ]; then
    info "Exists : $dir"
  else
    mkdir -p "$dir"
    success "Created: $dir"
  fi
done

# -----------------------------------------------------------------------------
# 4. Check for existing manifests
# -----------------------------------------------------------------------------
header "4. Checking existing training artifacts"

EXISTING_CLUSTERS=""
ITERATION_RUN=false

for cluster in A B C D E; do
  if [ -f "$TRAINING_DIR/modernization_manifest_${cluster}.md" ]; then
    EXISTING_CLUSTERS="$EXISTING_CLUSTERS $cluster"
    ITERATION_RUN=true
  fi
done

SKILL_AUDIT_EXISTS=false
if [ -f "$TRAINING_DIR/skill_audit.md" ]; then
  SKILL_AUDIT_EXISTS=true
fi

if [ "$RESTART" = true ] && [ "$ITERATION_RUN" = true ]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  for cluster in $EXISTING_CLUSTERS; do
    MANIFEST="$TRAINING_DIR/modernization_manifest_${cluster}.md"
    cp "$MANIFEST" "$TRAINING_DIR/modernization_manifest_${cluster}_backup_${STAMP}.md"
    rm "$MANIFEST"
    success "Cluster $cluster manifest backed up and removed"
  done
  if [ "$SKILL_AUDIT_EXISTS" = true ]; then
    cp "$TRAINING_DIR/skill_audit.md" "$TRAINING_DIR/skill_audit_backup_${STAMP}.md"
    rm "$TRAINING_DIR/skill_audit.md"
    success "Skill audit backed up and removed"
  fi
  info "Restart complete — all manifests cleared"
  ITERATION_RUN=false
  EXISTING_CLUSTERS=""
elif [ "$ITERATION_RUN" = true ]; then
  echo ""
  info "Existing manifests found for clusters:$EXISTING_CLUSTERS"
  if [ "$SKILL_AUDIT_EXISTS" = true ]; then
    info "Skill audit already exists — cluster scope previously defined"
  fi

  echo ""
  echo -e "  ${BOLD}Options:${RESET}"
  echo "  [k] Keep all existing manifests (iteration run)"
  echo "  [s] Select which manifests to keep or replace"
  echo "  [f] Back up all and start completely fresh"
  echo "  [a] Abort"
  echo "  (tip: run with --restart to skip this prompt and clear all manifests)"
  echo ""
  printf "  Your choice [k/s/f/a]: "
  read -r MANIFEST_CHOICE

  case "$MANIFEST_CHOICE" in
    k|K)
      success "Keeping all existing manifests — iteration run"
      ;;
    s|S)
      for cluster in $EXISTING_CLUSTERS; do
        MANIFEST="$TRAINING_DIR/modernization_manifest_${cluster}.md"
        CNAME="$(cluster_name "$cluster")"
        ITERATION=$(grep -m1 "^\*\*Version:\*\*" "$MANIFEST" 2>/dev/null | grep -o '[0-9]*' | head -1 || echo "unknown")
        echo ""
        echo -e "  Cluster ${BOLD}$cluster — $CNAME${RESET} (version: $ITERATION)"
        printf "  [k]eep or [b]ack up? "
        read -r CLUSTER_CHOICE
        case "$CLUSTER_CHOICE" in
          b|B)
            BACKUP="$TRAINING_DIR/modernization_manifest_${cluster}_backup_$(date +%Y%m%d_%H%M%S).md"
            cp "$MANIFEST" "$BACKUP"
            rm "$MANIFEST"
            success "Cluster $cluster backed up and removed"
            ;;
          *)
            success "Cluster $cluster kept"
            ;;
        esac
      done
      ;;
    f|F)
      STAMP="$(date +%Y%m%d_%H%M%S)"
      for cluster in $EXISTING_CLUSTERS; do
        MANIFEST="$TRAINING_DIR/modernization_manifest_${cluster}.md"
        cp "$MANIFEST" "$TRAINING_DIR/modernization_manifest_${cluster}_backup_${STAMP}.md"
        rm "$MANIFEST"
      done
      if [ "$SKILL_AUDIT_EXISTS" = true ]; then
        cp "$TRAINING_DIR/skill_audit.md" "$TRAINING_DIR/skill_audit_backup_${STAMP}.md"
        rm "$TRAINING_DIR/skill_audit.md"
      fi
      success "All manifests backed up — fresh start"
      ITERATION_RUN=false
      EXISTING_CLUSTERS=""
      ;;
    a|A|*)
      info "Aborted. No changes made."
      exit 0
      ;;
  esac
else
  info "No existing manifests — first run"
fi

# -----------------------------------------------------------------------------
# 5. Install shared ref files into .claude/refs/
# -----------------------------------------------------------------------------
header "5. Installing shared reference files"

for f in $REF_FILES; do
  DEST="$REFS_DIR/$f"
  cp "$SCRIPT_DIR/$f" "$DEST"
  success "Installed: .claude/refs/$f"
done

# -----------------------------------------------------------------------------
# 6. Install skills into .claude/skills/
# -----------------------------------------------------------------------------
header "6. Installing Claude Code skills"

cp "$SCRIPT_DIR/skill_interviewer.md" "$SKILL_INTERVIEWER/SKILL.md"
success "Installed: aipa-training-interviewer/SKILL.md"

cp "$SCRIPT_DIR/skill_generator.md"   "$SKILL_GENERATOR/SKILL.md"
success "Installed: aipa-training-generator/SKILL.md"

cp "$SCRIPT_DIR/skill_judge.md"       "$SKILL_JUDGE/SKILL.md"
success "Installed: aipa-training-judge/SKILL.md"

# -----------------------------------------------------------------------------
# 7. Deploy reference docs to /training/
# -----------------------------------------------------------------------------
header "7. Deploying training reference docs"

for f in $TRAINING_FILES; do
  cp "$SCRIPT_DIR/$f" "$TRAINING_DIR/$f"
  success "Deployed: training/$f"
done

# -----------------------------------------------------------------------------
# 8. Create /training/.gitignore
# -----------------------------------------------------------------------------
GITIGNORE="$TRAINING_DIR/.gitignore"
if [ ! -f "$GITIGNORE" ]; then
  cat > "$GITIGNORE" << 'EOF'
# Generated training datasets — do not commit
training_data_*.csv
training_data_*.jsonl
annotated_dataset_*.csv

# Review queues and judge reports
review_queue_*.md
judge_report_*.md

# Manifest and audit backups
modernization_manifest_*_backup_*.md
skill_audit_backup_*.md

# Commit these (version these as source of truth):
# skill_audit.md
# modernization_manifest_A.md (and _B _C _D _E)
# manifest_delta_A.md (and _B _C _D _E)
# methodology.md
# skill_clusters.md
EOF
  success "Created training/.gitignore"
fi

# -----------------------------------------------------------------------------
# 9. Summary and next steps
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}=================================================${RESET}"
echo -e "${GREEN}${BOLD}  Setup complete${RESET}"
echo -e "${BOLD}=================================================${RESET}"
echo ""
echo -e "  ${BOLD}Shared ref files:${RESET}"
for f in $REF_FILES; do
  echo -e "    ${GREEN}✓${RESET} .claude/refs/$f"
done
echo ""
echo -e "  ${BOLD}Claude Code skills:${RESET}"
echo -e "    ${GREEN}✓${RESET} .claude/skills/aipa-training-interviewer/"
echo -e "    ${GREEN}✓${RESET} .claude/skills/aipa-training-generator/"
echo -e "    ${GREEN}✓${RESET} .claude/skills/aipa-training-judge/"
echo ""
echo -e "  ${BOLD}Training reference docs:${RESET}"
for f in $TRAINING_FILES; do
  echo -e "    ${GREEN}✓${RESET} training/$f"
done

RETAINED=""
for cluster in A B C D E; do
  if [ -f "$TRAINING_DIR/modernization_manifest_${cluster}.md" ]; then
    RETAINED="$RETAINED $cluster"
  fi
done

if [ -n "$RETAINED" ]; then
  echo ""
  echo -e "  ${BOLD}Existing manifests retained:${RESET}"
  for cluster in $RETAINED; do
    echo -e "    ${CYAN}✓${RESET} Cluster $cluster — $(cluster_name "$cluster")"
  done
fi

echo ""

if [ "$ITERATION_RUN" = true ] && [ -n "$RETAINED" ]; then
  echo -e "  ${BOLD}This is an iteration run.${RESET}"
  echo ""
  echo -e "  ${BOLD}Next steps:${RESET}"
  echo ""
  echo -e "  1. Update ${CYAN}training/manifest_delta_{cluster}.md${RESET} with lessons"
  echo -e "     from the previous review cycle (if not already done)"
  echo ""
  echo -e "  2. In Claude Code, invoke the Interviewer skill:"
  echo -e "     ${CYAN}use the aipa-training-interviewer skill${RESET}"
  echo ""
  echo -e "  3. For each in-scope cluster, invoke the Generator skill:"
  echo -e "     ${CYAN}use the aipa-training-generator skill${RESET} (specify cluster)"
  echo ""
  echo -e "  4. For each cluster, invoke the Judge skill:"
  echo -e "     ${CYAN}use the aipa-training-judge skill${RESET} (specify cluster)"
else
  echo -e "  ${BOLD}This is a first run.${RESET}"
  echo ""
  echo -e "  ${BOLD}Next steps:${RESET}"
  echo ""
  echo -e "  1. In Claude Code, invoke the Interviewer skill:"
  echo -e "     ${CYAN}use the aipa-training-interviewer skill${RESET}"
  echo ""
  echo -e "     It will analyse the codebase, conduct the skill audit,"
  echo -e "     and produce manifests in ${CYAN}training/${RESET}"
  echo ""
  echo -e "  2. Review and approve all manifests"
  echo ""
  echo -e "  3. Invoke the Generator skill for each in-scope cluster:"
  echo -e "     ${CYAN}use the aipa-training-generator skill${RESET}"
  echo ""
  echo -e "  4. Invoke the Judge skill for each cluster:"
  echo -e "     ${CYAN}use the aipa-training-judge skill${RESET}"
  echo ""
  echo -e "  5. Review ${CYAN}training/review_queue_{cluster}_{date}.md${RESET},"
  echo -e "     produce ${CYAN}training/manifest_delta_{cluster}.md${RESET},"
  echo -e "     and re-run from step 1 until pass rate >= 90%% per cluster"
  echo ""
  echo -e "  See ${CYAN}training/methodology.md${RESET} for the full pipeline overview."
fi

echo ""

if [ "$ENV_OK" = false ]; then
  echo -e "  ${YELLOW}⚠  ANTHROPIC_API_KEY not detected. Ensure you are running${RESET}"
  echo -e "  ${YELLOW}   within an active Claude Code session before invoking skills.${RESET}"
  echo ""
fi
