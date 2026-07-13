#!/bin/bash
# =============================================================================
# GenSIE — Experiment Runner (Phase 1 + 2 + 3)
# =============================================================================
# Runs all evaluation phases sequentially. Each model runs through ALL 4
# pipelines × 3 datasets (starter, dev, test) before moving to the next.
#
# Execution order:
#   Phase 1 — google/gemma-4-e4b (Small tier)
#   Phase 2 — qwen3-14b (Medium tier)
#   Phase 3 — llama-3.2-3b-instruct (Tiny tier) [if enabled]
#   Phase 4 — Complexity analysis           [if enabled]
#
# Each individual run SKIPS if the result JSON already exists.
#
# Usage:
#   ./scripts/run_experiments.sh                           # phases 1+2
#   ./scripts/run_experiments.sh --with-llama              # phases 1+2+3
#   ./scripts/run_experiments.sh --with-llama --with-complexity  # all phases
#   ./scripts/run_experiments.sh --dry-run                 # preview only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GENSIE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$GENSIE_ROOT/results"

# --- Config ----------------------------------------------------------------
WITH_LLAMA=false
WITH_COMPLEXITY=false
DRY_RUN=false

# --- Colors ----------------------------------------------------------------
echo_blue()   { echo -e "\033[34m$*\033[0m"; }
echo_green()  { echo -e "\033[32m$*\033[0m"; }
echo_yellow() { echo -e "\033[33m$*\033[0m"; }
echo_red()    { echo -e "\033[31m$*\033[0m"; }

# --- Run one model through all pipelines × datasets -------------------------
run_model() {
    local model_id="$1" phase_name="$2"
    local dry_flag=""
    $DRY_RUN && dry_flag="--dry-run"

    echo ""
    echo_blue "============================================================"
    echo_blue "  PHASE $phase_name: $model_id"
    echo_blue "============================================================"
    echo ""

    bash "$SCRIPT_DIR/run_all_evaluations.sh" $dry_flag --model "$model_id" 2>&1
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        echo_red "  PHASE $phase_name FAILED (exit=$exit_code)"
        return $exit_code
    fi

    echo_green "  PHASE $phase_name COMPLETE"
    return 0
}

# --- Complexity analysis ----------------------------------------------------
run_complexity() {
    echo ""
    echo_blue "============================================================"
    echo_blue "  PHASE 4: COMPLEXITY ANALYSIS"
    echo_blue "============================================================"
    echo ""
    echo_yellow "[TODO] Complexity analysis not yet implemented."
    echo_yellow "  Stratify extraction results by complexity level (L1-L9)."
    echo ""
}

# --- Print summary of completed results ------------------------------------
print_summary() {
    echo ""
    echo_green "============================================================"
    echo_green "  EXPERIMENTS SUMMARY"
    echo_green "============================================================"
    echo ""

    local total=0 completed=0

    MODELS=("google_gemma-4-e4b-it" "qwen/qwen3-14b")
    $WITH_LLAMA && MODELS+=("llama-3.2-3b-instruct")
    DATASETS=("starter" "dev" "test")
    PIPELINES=("baseline" "extraction" "hybrid_cot" "adaptive")

    for dataset in "${DATASETS[@]}"; do
        echo "--- ${dataset^^} ---"
        printf "%-15s %-30s %-8s\n" "Pipeline" "Model" "F1"
        echo "-----------------------------------------------------------------"
        for pipeline in "${PIPELINES[@]}"; do
            for model in "${MODELS[@]}"; do
                total=$((total + 1))
                model_short=$(echo "$model" | tr '/' '_')
                out="$RESULTS_DIR/seesml_${pipeline}_${model_short}_${dataset}.json"
                if [[ -f "$out" ]]; then
                    completed=$((completed + 1))
                    f1=$(python3 -c "import json; d=json.load(open('$out')); print(f\"{d['metrics']['f1']:.4f}\")" 2>/dev/null || echo "—")
                    printf "%-15s %-30s %-8s\n" "$pipeline" "$model" "$f1"
                fi
            done
        done
        echo ""
    done

    echo "Completed: $completed / $total runs"
    echo ""
}

# --- Main ------------------------------------------------------------------
main() {
    mkdir -p "$RESULTS_DIR"

    echo_blue "============================================================"
    echo_blue "  GenSIE Experiments"
    echo_blue "  Date: $(date)"
    echo_blue "  Llama: $WITH_LLAMA"
    echo_blue "  Complexity: $WITH_COMPLEXITY"
    echo_blue "  Dry-run: $DRY_RUN"
    echo_blue "============================================================"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo_yellow "  DRY-RUN mode — no actual evaluations will run"
        echo ""
        run_model "google/gemma-4-e4b" "1 (gemma, dry)" || true
        run_model "qwen3-14b" "2 (qwen, dry)" || true
        $WITH_LLAMA && run_model "llama-3.2-3b-instruct" "3 (llama, dry)" || true
        $WITH_COMPLEXITY && run_complexity
        print_summary
        return 0
    fi

    # Phase 1: Gemma
    run_model "google_gemma-4-e4b-it" "1 (gemma)"

    # Phase 2: Qwen
    run_model "qwen/qwen3-14b" "2 (qwen)"

    # Phase 3: Llama (optional)
    if $WITH_LLAMA; then
        run_model "llama-3.2-3b-instruct" "3 (llama)"
    fi

    # Phase 4: Complexity analysis (optional)
    if $WITH_COMPLEXITY; then
        run_complexity
    fi

    print_summary

    echo ""
    echo_green "============================================================"
    echo_green "  ALL PHASES COMPLETE"
    echo_green "============================================================"
    echo "Results: $RESULTS_DIR"
}

# --- Parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-llama)      WITH_LLAMA=true; shift ;;
        --with-complexity) WITH_COMPLEXITY=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        *)                 echo "Usage: $0 [--with-llama] [--with-complexity] [--dry-run]"; exit 1 ;;
    esac
done

main
