#!/bin/bash
# =============================================================================
# GenSIE — SEsml: Setup & Experiment Runner
# =============================================================================
# Run this script on a NEW machine to set up and launch experiments.
#
# Prerequisites:
#   - Python 3.13 (or as specified in .python-version)
#   - uv (https://docs.astral.sh/uv/)
#   - LM Studio running on localhost:8080 with models loaded
#
# Usage:
#   bash setup_and_run.sh                          # phases 1+2 (gemma + qwen)
#   bash setup_and_run.sh --with-llama             # phases 1+2+3
#   bash setup_and_run.sh --with-llama --with-complexity  # all phases
#
# Quick start:
#   1. Install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   2. Edit .env with your LM Studio URL
#   3. bash setup_and_run.sh
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "  GenSIE SEsml — Setup & Experiment Runner"
echo "============================================================"
echo ""

# --- 1. Check prerequisites ------------------------------------------------
echo "▶ [1/5] Checking prerequisites..."

if ! command -v uv &> /dev/null; then
    echo "  ❌ 'uv' not found. Install it first:"
    echo "     curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  ✅ uv $(uv --version)"

PYTHON_VERSION=$(python3 --version 2>/dev/null || echo "none")
REQUIRED=$(cat .python-version)
echo "  📋 Required Python: $REQUIRED | Current: $PYTHON_VERSION"

# --- 2. Install dependencies -----------------------------------------------
echo ""
echo "▶ [2/5] Installing dependencies (uv sync)..."
uv sync
echo "  ✅ Dependencies installed"

# --- 3. Check .env configuration -------------------------------------------
echo ""
echo "▶ [3/5] Checking environment configuration..."

if [[ ! -f .env ]]; then
    echo "  ⚠️  No .env found. Creating template..."
    cat > .env << 'ENVEOF'
# LM Studio (or OpenAI-compatible) endpoint
OPENAI_BASE_URL=http://localhost:8080/v1
# API key (dummy for local LM Studio)
OPENAI_API_KEY=sk-dummy
ENVEOF
    echo "  📝 Created .env — edit OPENAI_BASE_URL if LM Studio is on a different host"
else
    echo "  ✅ .env found"
    echo "  📋 Config:"
    cat .env | sed 's/^/    /'
fi

source .env 2>/dev/null || true

# --- 4. Check LM Studio connection -----------------------------------------
echo ""
echo "▶ [4/5] Checking LM Studio connection..."

if curl -sf "${OPENAI_BASE_URL:-http://localhost:8080/v1}/models" > /dev/null 2>&1; then
    echo "  ✅ LM Studio is running at ${OPENAI_BASE_URL:-http://localhost:8080/v1}"
    echo "  📋 Available models:"
    curl -sf "${OPENAI_BASE_URL:-http://localhost:8080/v1}/models" | python3 -m json.tool 2>/dev/null | grep '"id"' | sed 's/^/      /' || echo "      (could not list models)"
else
    echo "  ⚠️  LM Studio not reachable at ${OPENAI_BASE_URL:-http://localhost:8080/v1}"
    echo "  🔧 Please:"
    echo "     1. Start LM Studio"
    echo "     2. Load the required models"
    echo "     3. Start the inference server (Settings → Developer → Start Server)"
    echo "     4. Re-run this script"
    echo ""
    echo "  ℹ️  If LM Studio is on a different machine, set OPENAI_BASE_URL in .env"
    echo ""
    echo "  Continue anyway? (y/N)"
    read -r answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        exit 1
    fi
fi

# --- 5. Launch experiments -------------------------------------------------
echo ""
echo "▶ [5/5] Launching experiments..."

# Forward any CLI flags to the experiment runner
exec bash scripts/run_experiments.sh "$@"
