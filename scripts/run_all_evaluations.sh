#!/bin/bash
# =============================================================================
# GenSIE — Evaluation Runner
# =============================================================================
# Runs 4 pipelines × 3 models × 3 datasets (starter, dev, test) = 36 runs.
#
# Each run saves a result file named:
#   results/seesml_{pipeline}_{model_short}_{dataset}.json
#
# If that file already exists, the run is SKIPPED. Delete the file to re-run.
#
# Usage:
#   ./scripts/run_all_evaluations.sh                         # run everything
#   ./scripts/run_all_evaluations.sh --dataset dev           # only one dataset
#   ./scripts/run_all_evaluations.sh --model qwen3-14b       # only one model
#   ./scripts/run_all_evaluations.sh --dry-run               # just print what would run
#   ./scripts/run_all_evaluations.sh --pipeline baseline     # only one pipeline
#
# Requires: gensie server on http://localhost:8000 (starts it if not running)
# =============================================================================

set -euo pipefail

GENSIE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$GENSIE_ROOT/results"
PYTHON="$GENSIE_ROOT/.venv/bin/python"
SERVER_URL="http://localhost:8000"
REQUEST_TIMEOUT=300
TIME_BUDGET=60

DATASETS=("starter" "dev" "test")
DATASET_PATHS=(
  "$GENSIE_ROOT/data/starter"
  "$GENSIE_ROOT/data/dev"
  "$GENSIE_ROOT/data/test"
)

PIPELINES=("baseline" "extraction" "hybrid_cot" "adaptive")
MODELS=(
  "llama-3.2-3b-instruct"   # Tiny
  "google_gemma-4-e4b-it"   # Small
  "qwen/qwen3-14b"          # Medium
)

SERVER_AUTO_START=true
SERVER_PORT=8000
SERVER_LOG="/tmp/gensie_server_eval.log"

# --- Helpers ---------------------------------------------------------------
echo_blue()   { echo -e "\033[34m$*\033[0m"; }
echo_green()  { echo -e "\033[32m$*\033[0m"; }
echo_yellow() { echo -e "\033[33m$*\033[0m"; }
echo_red()    { echo -e "\033[31m$*\033[0m"; }

# --- Output filename for a given combination -------------------------------
output_file() {
    local dataset="$1" pipeline="$2" model="$3"
    local model_short
    model_short=$(echo "$model" | tr '/' '_')
    echo "$RESULTS_DIR/seesml_${pipeline}_${model_short}_${dataset}.json"
}

# --- Server management ----------------------------------------------------
start_server() {
    if curl -sf "$SERVER_URL/info" > /dev/null 2>&1; then
        echo_green "Server already running at $SERVER_URL"
        return 0
    fi

    if [[ "$SERVER_AUTO_START" != "true" ]]; then
        echo_red "Server not running. Start it manually or set SERVER_AUTO_START=true"
        return 1
    fi

    echo_blue "Starting gensie server on port $SERVER_PORT..."
    export PYTHONPATH="$GENSIE_ROOT/src"
    export PARTICIPANT_PATH="gensie.baseline.OfficialParticipant"
    # Use LM Studio from env or default to localhost:1234
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:1234/v1}"
    export OPENAI_API_KEY="sk-dummy"

    nohup "$PYTHON" -m gensie.cli serve \
        --host 0.0.0.0 --port "$SERVER_PORT" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    for i in $(seq 1 30); do
        sleep 1
        if curl -sf "$SERVER_URL/info" > /dev/null 2>&1; then
            echo_green "Server ready (PID $SERVER_PID)"
            return 0
        fi
    done

    echo_red "Server failed to start. Check $SERVER_LOG"
    return 1
}

# --- Run one evaluation ---------------------------------------------------
run_eval() {
    local dataset="$1" dataset_path="$2" pipeline="$3" model="$4"
    local out
    out=$(output_file "$dataset" "$pipeline" "$model")

    # Skip only if output file exists, has content, and contains valid JSON with metrics.
    # If the file is empty, corrupt, or missing metrics → delete and re-run.
    if [[ -f "$out" ]]; then
        if [[ ! -s "$out" ]]; then
            echo_yellow "[EMPTY] $out — deleting and re-running"
            rm -f "$out"
        else
            local f1
            f1=$("$PYTHON" -c "import json; d=json.load(open('$out')); print(d['metrics']['f1'])" 2>/dev/null || echo "")
            if [[ -n "$f1" ]]; then
                echo_green "[SKIP] $dataset / $pipeline / $model → $out (F1=$f1)"
                return 0
            else
                echo_yellow "[CORRUPT] $out — cannot parse, deleting and re-running"
                rm -f "$out"
            fi
        fi
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo_yellow "[DRY]  $dataset / $pipeline / $model → $out"
        return 0
    fi

    echo ""
    echo_blue "============================================================"
    echo_blue "  RUN:     $dataset / $pipeline / $model"
    echo_blue "  OUTPUT:  $out"
    echo_blue "  START:   $(date)"
    echo_blue "============================================================"
    echo ""

    export PYTHONPATH="$GENSIE_ROOT/src"
    set +e
    "$PYTHON" -m gensie.cli eval \
        --data "$dataset_path" \
        --url "$SERVER_URL" \
        --pipeline "$pipeline" \
        --model "$model" \
        --output "$out" \
        --time-budget-s "$TIME_BUDGET" \
        --request-timeout-s "$REQUEST_TIMEOUT" \
        2>&1
    local exit_code=$?
    set -e

    if [[ $exit_code -ne 0 ]]; then
        echo_red "  FAILED: $dataset / $pipeline / $model (exit=$exit_code)"
        return $exit_code
    fi

    local f1
    f1=$("$PYTHON" -c "import json; d=json.load(open('$out')); print(d['metrics']['f1'])" 2>/dev/null || echo "?")
    echo_green "  DONE: $dataset / $pipeline / $model → F1=$f1"
    return 0
}

# --- Summary from result files -------------------------------------------
print_summary() {
    echo ""
    echo_green "============================================================"
    echo_green "  RESULTS SUMMARY"
    echo_green "============================================================"
    echo ""

    for dataset in "${DATASETS[@]}"; do
        # Skip if no results for this dataset
        first_out=$(output_file "$dataset" "${PIPELINES[0]}" "${MODELS[0]}")
        dir_exists=false
        for f in "$RESULTS_DIR"/seesml_*_"$dataset".json; do
            [[ -f "$f" ]] && dir_exists=true && break
        done
        [[ "$dir_exists" != "true" ]] && continue

        echo "--- ${dataset^^} ---"
        printf "%-15s %-30s %-8s\n" "Pipeline" "Model" "F1"
        echo "-----------------------------------------------------------------"
        for pipeline in "${PIPELINES[@]}"; do
            for model in "${MODELS[@]}"; do
                out=$(output_file "$dataset" "$pipeline" "$model")
                if [[ -f "$out" ]]; then
                    f1=$("$PYTHON" -c "import json; d=json.load(open('$out')); print(f\"{d['metrics']['f1']:.4f}\")" 2>/dev/null || echo "—")
                    printf "%-15s %-30s %-8s\n" "$pipeline" "$model" "$f1"
                fi
            done
        done
        echo ""
    done

    # Count completed
    local completed=0
    for dataset in "${DATASETS[@]}"; do
        for pipeline in "${PIPELINES[@]}"; do
            for model in "${MODELS[@]}"; do
                out=$(output_file "$dataset" "$pipeline" "$model")
                [[ -f "$out" ]] && completed=$((completed + 1))
            done
        done
    done
    echo "Completed: $completed / 36 runs"
    echo ""
}

# --- Main -----------------------------------------------------------------
main() {
    mkdir -p "$RESULTS_DIR"

    if ! start_server; then
        [[ "$DRY_RUN" != "true" ]] && { echo_red "Cannot proceed without server."; exit 1; }
    fi

    TOTAL_RUNS=$(( ${#DATASETS[@]} * ${#PIPELINES[@]} * ${#MODELS[@]} ))
    RUN_NUM=0

    for di in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$di]}"
        dataset_path="${DATASET_PATHS[$di]}"

        [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]] && continue

        for pipeline in "${PIPELINES[@]}"; do
            [[ -n "$FILTER_PIPELINE" && "$pipeline" != "$FILTER_PIPELINE" ]] && continue

            for model in "${MODELS[@]}"; do
                [[ -n "$FILTER_MODEL" && "$model" != "$FILTER_MODEL" ]] && continue
                RUN_NUM=$((RUN_NUM + 1))
                echo ""
                echo_yellow "--- Run $RUN_NUM / $TOTAL_RUNS: $dataset / $pipeline / $model ---"
                run_eval "$dataset" "$dataset_path" "$pipeline" "$model" || true
                sleep 2
            done
        done
    done

    print_summary
    echo_blue "Results: $RESULTS_DIR"
}

# --- Parse args -----------------------------------------------------------
DRY_RUN=false
FILTER_DATASET=""
FILTER_PIPELINE=""
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --dataset)    FILTER_DATASET="$2"; shift 2 ;;
        --pipeline)   FILTER_PIPELINE="$2"; shift 2 ;;
        --model)      FILTER_MODEL="$2"; shift 2 ;;
        *)            echo "Usage: $0 [--dataset starter|dev|test] [--pipeline name] [--model name] [--dry-run]"; exit 1 ;;
    esac
done

main
