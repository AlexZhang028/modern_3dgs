#!/bin/bash
# =============================================================================
# Batch training script: Corgi SharpTimeGS baseline ablations
#
# Three configs isolating the effect of lambda_n and SAM2 loss boost,
# using pure SharpTimeGS (no phase decoupling, no color correction).
# Motion blur penalty is excluded — it is not part of the original paper.
#
#   vanilla   — full SharpTimeGS (paper Eq.7: L_scale + L_opacity + L_n + L_t)
#   no_ln     — same but lambda_n=0: measures L_n overhead (iter/s) and quality cost
#   sam2      — vanilla + SAM2 loss boost, no phase decoupling
#
# Usage:
#   bash scripts/batch_train_corgi_baseline.sh              # run all three
#   bash scripts/batch_train_corgi_baseline.sh vanilla      # single experiment
#   bash scripts/batch_train_corgi_baseline.sh no_ln sam2   # subset
#
# Logs    → logs/corgi_baseline_<name>.log
# Results → /media/data/freetimegs/res/corgi/baseline_*/
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
    [vanilla]="config/corgi_baseline_vanilla.yaml"
    [no_ln]="config/corgi_baseline_no_ln.yaml"
    [sam2]="config/corgi_baseline_sam2.yaml"
)

declare -A DESCRIPTIONS=(
    [vanilla]="Vanilla SharpTimeGS — paper Eq.7 losses, no extras (reference)"
    [no_ln]="Ablation: no L_n — removes second render pass, tests overhead vs quality"
    [sam2]="Ablation: +SAM2 loss boost — no phase decoupling, tests boost-only effect"
)

DEFAULT_ORDER=(vanilla no_ln sam2)

# ---------------------------------------------------------------------------
# Parse arguments
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
    LOG="$LOG_DIR/corgi_baseline_${key}.log"

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
        echo "  [FAIL] '$key' failed after $(($ELAPSED / 60))m $(($ELAPSED % 60))s — see $LOG"
        FAILED+=("$key")
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
echo "  Logs   : $LOG_DIR/corgi_baseline_*.log"
echo "  Results: /media/data/freetimegs/res/corgi/baseline_*/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
