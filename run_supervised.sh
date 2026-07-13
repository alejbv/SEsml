#!/bin/bash
# =============================================================================
# GenSIE — Supervised Experiment Runner
# =============================================================================
# 
# Starts (if not running):
#   1. LM Studio inference server
#   2. GenSIE evaluation server
# Then launches the experiments.
#
# All processes survive SSH disconnects. Run this script, log out, go to sleep,
# come back and check results. Works with or without tmux.
#
# Usage:
#   bash run_supervised.sh                           # phases 1+2 (gemma + qwen)
#   bash run_supervised.sh --with-llama              # phases 1+2+3
#   bash run_supervised.sh --with-llama --with-complexity  # all phases
#   bash run_supervised.sh --tmux                    # run INSIDE a tmux session
#
# If --tmux is given, the script wraps itself in a tmux session so you can
# safely disconnect (Ctrl+B, d) and reattach later (tmux attach -t gensie).
# =============================================================================

set -euo pipefail

# --- Config ----------------------------------------------------------------
LMSTUDIO_PORT=1234
GENSIE_PORT=8000
CONTEXT_LENGTH=32768
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
LMSTUDIO_LOG="$LOG_DIR/lms_server.log"
GENSIE_LOG="$LOG_DIR/gensie_server.log"
EXPERIMENT_LOG="$LOG_DIR/experiments.log"
TMUX_SESSION="gensie"

# Models used in experiments (in order)
BASE_MODELS=("google_gemma-4-e4b-it" "qwen/qwen3-14b")
LLAMA_MODEL="llama-3.2-3b-instruct"

# --- Colors ----------------------------------------------------------------
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
BLUE='\033[34m'
CYAN='\033[36m'
NC='\033[0m'

echo_blue()   { echo -e "${BLUE}$*${NC}"; }
echo_green()  { echo -e "${GREEN}$*${NC}"; }
echo_yellow() { echo -e "${YELLOW}$*${NC}"; }
echo_red()    { echo -e "${RED}$*${NC}"; }
echo_cyan()   { echo -e "${CYAN}$*${NC}"; }

# --- Helpers ---------------------------------------------------------------
check_port() {
    local port=$1
    ss -tlnp 2>/dev/null | grep -q ":$port " || nc -z localhost "$port" 2>/dev/null
}

wait_for_port() {
    local port=$1 name=$2 timeout=${3:-120}
    local waited=0
    echo -n "  Waiting for $name on port $port..."
    while ! check_port "$port"; do
        if [ $waited -ge $timeout ]; then
            echo -e " ${RED}TIMEOUT after ${timeout}s${NC}"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    echo -e " ${GREEN}READY (${waited}s)${NC}"
    return 0
}

# --- Load models with proper context length ---------------------------------
load_models() {
    local models=("${BASE_MODELS[@]}")
    local with_llama=false
    for arg in "$@"; do [ "$arg" = "--with-llama" ] && with_llama=true; done
    $with_llama && models+=("$LLAMA_MODEL")

    echo_cyan "  Loading models with ${CONTEXT_LENGTH} context..."
    echo ""
    for model in "${models[@]}"; do
        echo -n "    $model → "
        if lms ls 2>/dev/null | grep -qi "$model"; then
            # Already loaded — unload first to apply new context length
            lms unload "$model" 2>/dev/null || true
        fi
        if lms load "$model" --context-length "$CONTEXT_LENGTH" > /dev/null 2>>"$LMSTUDIO_LOG"; then
            echo -e "${GREEN}loaded${NC}"
        else
            echo -e "${YELLOW}⚠️  load failed — check $LMSTUDIO_LOG${NC}"
            continue
        fi
        
        # Verify context length by sending a prompt that exceeds 4096 tokens.
        # If the model is still loaded with default n_ctx=4096, this will
        # return a 400 error. If it accepts the prompt, context is correct.
        echo -n "    Verifying context length..."
        if python3 -c "
import urllib.request, json, sys
data = json.dumps({
    'model': '$model',
    'messages': [{'role': 'user', 'content': 'test ' * 2500}],
    'max_tokens': 1
}).encode()
req = urllib.request.Request('http://localhost:$LMSTUDIO_PORT/v1/chat/completions', data=data, headers={'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req, timeout=30)
    r = json.loads(resp.read())
    sys.exit(0 if r.get('choices') else 1)
except Exception as e:
    print(f'FAILED: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
            echo -e " ${GREEN}✅ ${CONTEXT_LENGTH} ctx verified${NC}"
        else
            echo -e " ${RED}❌ Context test FAILED — model loaded with default n_ctx=4096${NC}"
            echo_red "     Run: lms unload \"$model\" && lms load \"$model\" --context-length $CONTEXT_LENGTH"
            return 1
        fi
    done
    echo ""
    echo_green "  ✅ Models ready"
}

# --- TMUX wrapper: if --tmux and not already inside tmux, wrap -------------
maybe_tmux() {
    for arg in "$@"; do
        if [ "$arg" = "--tmux" ]; then
            if [ -z "${TMUX:-}" ]; then
                echo_cyan "🔄 Wrapping in tmux session '$TMUX_SESSION'..."
                echo_cyan "   After starting, press Ctrl+B then d to detach."
                echo_cyan "   Reattach: tmux attach -t $TMUX_SESSION"
                echo ""
                # Strip --tmux from args and re-exec inside tmux
                ARGS=()
                for a in "$@"; do [ "$a" != "--tmux" ] && ARGS+=("$a"); done
                exec tmux new-session -s "$TMUX_SESSION" \
                    "cd '$PROJECT_DIR' && bash '$0' ${ARGS[*]}; echo; echo_red 'Script finished — press Ctrl+D to exit'; bash"
            fi
        fi
    done
}

# --- Main ------------------------------------------------------------------
main() {
    mkdir -p "$LOG_DIR"

    echo_blue "============================================================"
    echo_blue "  GenSIE Supervised Runner"
    echo_blue "  $(date)"
    echo_blue "  Project: $PROJECT_DIR"
    echo_blue "  Logs:    $LOG_DIR"
    echo_blue "============================================================"
    echo ""

    # ---- Step 1: LM Studio (fresh restart) --------------------------------
    echo_yellow "[1/5] Restarting LM Studio (port $LMSTUDIO_PORT)..."
    
    # ---- Force-restart LM Studio daemon ----
    # The old daemon may have loaded models with default n_ctx=4096.
    # A simple `lms server stop` + `lms server start` often reconnects to
    # the same process. We use SIGKILL to guarantee a clean slate.
    
    echo_cyan "  Force-stopping ALL LM Studio processes..."
    lms server stop 2>/dev/null || true
    sleep 1
    pkill -9 -f "lms" 2>/dev/null || true
    sleep 2
    
    # Verify port is actually free
    if check_port "$LMSTUDIO_PORT"; then
        echo_red "  ⚠️  Port $LMSTUDIO_PORT still in use after SIGKILL — waiting..."
        sleep 5
        if check_port "$LMSTUDIO_PORT"; then
            echo_red "  ❌ Cannot free port $LMSTUDIO_PORT. Something else is using it."
            ss -tlnp 2>/dev/null | grep ":$LMSTUDIO_PORT " || true
            exit 1
        fi
    fi
    
    echo_cyan "  🚀 Starting LM Studio fresh..."
    nohup lms server start --bind 0.0.0.0 > "$LMSTUDIO_LOG" 2>&1 &
    echo_cyan "  (lms server start daemonizes — check log: $LMSTUDIO_LOG)"

    if ! wait_for_port "$LMSTUDIO_PORT" "LM Studio" 120; then
        echo_red "  ❌ LM Studio failed to start. Check:"
        echo_red "     cat $LMSTUDIO_LOG"
        exit 1
    fi

    # Verify LM Studio API is actually responding
    if curl -sf "http://localhost:$LMSTUDIO_PORT/v1/models" > /dev/null 2>&1; then
        echo_green "  ✅ LM Studio API responding at :$LMSTUDIO_PORT"
    else
        echo_red "  ⚠️  Port $LMSTUDIO_PORT open but API not responding."
        echo_red "     Check LM Studio: are models loaded? Is the server started?"
        exit 1
    fi

    # ---- Step 2: GenSIE evaluation server ----------------------------------
    echo ""
    echo_yellow "[2/5] Checking GenSIE evaluation server (port $GENSIE_PORT)..."
    if check_port "$GENSIE_PORT"; then
        echo_green "  ✅ GenSIE server already running"
    else
        echo_cyan "  🚀 Starting GenSIE server..."
        export PYTHONPATH="$PROJECT_DIR/src"
        export PARTICIPANT_PATH="gensie.baseline.OfficialParticipant"
        export OPENAI_BASE_URL="http://localhost:$LMSTUDIO_PORT/v1"
        export OPENAI_API_KEY="sk-dummy"

        cd "$PROJECT_DIR"
        nohup "$PROJECT_DIR/.venv/bin/python" -m gensie.cli serve \
            --host 0.0.0.0 --port "$GENSIE_PORT" \
            > "$GENSIE_LOG" 2>&1 &
        echo_cyan "  PID: $!"
        echo_cyan "  Log: $GENSIE_LOG"

        if ! wait_for_port "$GENSIE_PORT" "Gensie server" 60; then
            echo_red "  ❌ GenSIE server failed to start. Check:"
            echo_red "     cat $GENSIE_LOG"
            exit 1
        fi
    fi

    # ---- Step 3: Verify both servers are healthy ---------------------------
    echo ""
    echo_yellow "[3/5] Verifying both servers..."
    echo ""
    printf "  %-25s %-10s %s\n" "Service" "Port" "Status"
    echo "  ---------------------------------------------------"

    # LM Studio health
    if curl -sf "http://localhost:$LMSTUDIO_PORT/v1/models" > /dev/null 2>&1; then
        printf "  ${GREEN}%-25s${NC} ${CYAN}%-10s${NC} ${GREEN}✅ OK${NC}\n" "LM Studio" ":$LMSTUDIO_PORT"
    else
        printf "  ${RED}%-25s${NC} ${CYAN}%-10s${NC} ${RED}❌ DOWN${NC}\n" "LM Studio" ":$LMSTUDIO_PORT"
        exit 1
    fi

    # GenSIE health
    if curl -sf "http://localhost:$GENSIE_PORT/info" > /dev/null 2>&1; then
        printf "  ${GREEN}%-25s${NC} ${CYAN}%-10s${NC} ${GREEN}✅ OK${NC}\n" "GenSIE Server" ":$GENSIE_PORT"
    else
        printf "  ${RED}%-25s${NC} ${CYAN}%-10s${NC} ${RED}❌ DOWN${NC}\n" "GenSIE Server" ":$GENSIE_PORT"
        # Try alternative health endpoint
        if curl -sf "http://localhost:$GENSIE_PORT/" > /dev/null 2>&1; then
            printf "  ${GREEN}%-25s${NC} ${CYAN}%-10s${NC} ${GREEN}✅ OK (root)${NC}\n" "GenSIE Server" ":$GENSIE_PORT"
        else
            exit 1
        fi
    fi

    # ---- Step 4: Load models with 32K context ------------------------------
    echo ""
    echo_yellow "[4/5] Loading models with ${CONTEXT_LENGTH} context..."
    if ! load_models "$@"; then
        echo_red "  ❌ Model loading failed (likely context length issue)."
        echo_red "     Fix and re-run. NOT launching experiments."
        exit 1
    fi

    # ---- Step 5: Launch experiments ----------------------------------------
    echo ""
    echo_yellow "[5/5] Launching experiments..."
    echo ""

    # Remove --tmux from forwarded args (already handled)
    ARGS=()
    for a in "$@"; do [ "$a" != "--tmux" ] && ARGS+=("$a"); done

    cd "$PROJECT_DIR"
    echo_cyan "  Command: bash scripts/run_experiments.sh ${ARGS[*]}"
    echo_cyan "  Log:     $EXPERIMENT_LOG"
    echo ""

    # Run experiments — this blocks until finished
    bash scripts/run_experiments.sh "${ARGS[@]}" 2>&1 | tee "$EXPERIMENT_LOG"
    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo_green "============================================================"
        echo_green "  ✅ ALL EXPERIMENTS COMPLETE"
        echo_green "============================================================"
    else
        echo_red "============================================================"
        echo_red "  ❌ EXPERIMENTS FAILED (exit=$exit_code)"
        echo_red "============================================================"
    fi

    echo ""
    echo_cyan "  Results: $PROJECT_DIR/results/"
    echo_cyan "  Logs:    $LOG_DIR"
    echo ""

    return $exit_code
}

# --- Entry point -----------------------------------------------------------
maybe_tmux "$@"
main "$@"
