#!/bin/bash
# Setup the sandbox Docker image with Python + project dependencies
set -e

echo "=== Building sandbox with Python runtime ==="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not available, installing Python packages system-wide"
    apt-get update -qq && apt-get install -y -qq python3-pip 2>&1 | tail -3
    python3 -m pip install --quiet openai fastembed httpx pytest 2>&1 | tail -5
    echo "=== System packages installed ==="
    exit 0
fi

echo "Docker available, building extended sandbox image..."

# Create a more complete Dockerfile
cat > /tmp/sandbox-ext.Dockerfile << 'DOCKERFILE'
FROM gensie-sandbox:latest

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --quiet openai fastembed httpx pytest 2>&1 | tail -5

WORKDIR /project
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
DOCKERFILE

docker build -t gensie-sandbox:latest -f /tmp/sandbox-ext.Dockerfile .
echo "=== Extended sandbox image built ==="
