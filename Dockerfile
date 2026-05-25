# Use a slim Python image
FROM python:3.13-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml README.md ./

# Copy source code
COPY src/ ./src/

# Install the project with all dependencies (system-wide, production only)
# Note: uv.lock is optional — omitting it lets uv resolve at build time
RUN uv pip install --system -e .

# Pre-download fastembed model for offline inference (no internet at runtime)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Expose the FastAPI port
EXPOSE 8000

# Set default environment variables for the agent
ENV PARTICIPANT_PATH="gensie.baseline.OfficialParticipant"
ENV OPENAI_BASE_URL=""
ENV OPENAI_API_KEY="sk-dummy"

# Run the server via the CLI
ENTRYPOINT ["gensie"]
