#!/bin/bash
# =============================================================================
# Batch training script: Corgi dynamic-detail experiments
#
# NOTE: Experiments A/C/D were re-created after fixing a parser bug where all
# trainer.* params (decouple_training_enabled, decouple_dynamic_n_split,
# sam2_masks_dir, etc.) were silently ignored — every previous run used
# TrainerConfig defaults.  These configs now actually exercise the intended
# parameter changes.
#
# Exp B (lower global grad threshold) is excluded — caused GPU OOM.
#
# Usage:
#   bash scripts/batch_train_corgi.sh           # run all (baseline A C D AC)
#   bash scripts/batch_train_corgi.sh A C       # run only A and C
#   bash scripts/batch_train_corgi.sh baseline  # run only baseline
#
# Logs  → logs/corgi_exp_<name>.log
# Results → /media/data/freetimegs/res/corgi/exp_<name>/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
declare -A CONFIGS=(
    [baseline]="config/corgi_edge_optim.yaml"
    [A]="config/corgi_exp_A_more_splits.yaml"
    [C]="config/corgi_exp_C_stronger_boost.yaml"
    [D]="config/corgi_exp_D_extended_dynamic.yaml"
    [AC]="config/corgi_AC_aggressive.yaml"
)

declare -A DESCRIPTIONS=(
    [baseline]="Baseline — corgi_edge_optim (reference)"
    [A]="Exp A — More dynamic splits: n_split 3→4, densify_mult 0.5→0.4 (2.5× grad boost)"
    [C]="Exp C — Stronger SAM2 boost: max_boost 4→6, ramp 15k→20k, power 3→2"
    [D]="Exp D — Extended Phase 2: dynamic_end 25k→30k, densify_until 30k"
    [AC]="Exp AC — Aggressive fusion: n_split=5, mult=0.33, boost=8.0 (best of A+C)"
)

# Default run order (B omitted — causes GPU OOM)
DEFAULT_ORDER=(baseline A C D AC)

# ---------------------------------------------------------------------------
# Parse arguments: which experiments to run
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    TO_RUN=("${DEFAULT_ORDER[@]}")
else
    TO_RUN=("$@")
fi

for key in "${TO_RUN[@]}"; do
    if [ -z "${CONFIGS[$key]+x}" ]; then
        echo "[ERROR] Unknown experiment: '$key'"
        echo "  Valid: ${!CONFIGS[*]}"
        exit 1
    fi
    if [ ! -f "${CONFIGS[$key]}" ]; then
        echo "[ERROR] Config not found: ${CONFIGS[$key]}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    echo "[WARN] nvidia-smi not found — cannot confirm GPU availability"
else
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader
    echo ""
fi

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
TOTAL=${#TO_RUN[@]}
DONE=0
FAILED=()

for key in "${TO_RUN[@]}"; do
    DONE=$((DONE + 1))
    CONFIG="${CONFIGS[$key]}"
    DESC="${DESCRIPTIONS[$key]}"
    LOG="$LOG_DIR/corgi_exp_${key}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$DONE/$TOTAL] $DESC"
    echo "  Config : $CONFIG"
    echo "  Log    : $LOG"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    START_TS=$SECONDS

    if python train.py --config "$CONFIG" 2>&1 | tee "$LOG"; then
        ELAPSED=$((SECONDS - START_TS))
        echo ""
        echo "  [OK] Finished in $(($ELAPSED / 60))m $(($ELAPSED % 60))s"
    else
        ELAPSED=$((SECONDS - START_TS))
        echo ""
        echo "  [FAIL] Experiment '$key' failed after $(($ELAPSED / 60))m $(($ELAPSED % 60))s"
        echo "         See: $LOG"
        FAILED+=("$key")
        # Continue to next experiment instead of aborting
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BATCH COMPLETE"
echo "  Ran    : ${TO_RUN[*]}"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Status : All experiments succeeded"
else
    echo "  Failed : ${FAILED[*]}"
fi
echo "  Logs   : $LOG_DIR/corgi_exp_*.log"
echo "  Results: /media/data/freetimegs/res/corgi/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
