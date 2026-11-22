# ============================================================
# Multi-stage Dockerfile for RAG Semantic Cache API
# ============================================================

# Stage 1: Base image with Python
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies (gosu for entrypoint to drop root)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libmagic1 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY --chown=appuser:appuser . .

# Make entrypoint executable (chown mounted dirs so appuser can write logs)
RUN chmod +x /app/entrypoint.sh

# Create necessary directories
RUN mkdir -p data/documents data/vector_db data/cache logs && \
    chown -R appuser:appuser data logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run as root so entrypoint can chown; entrypoint then runs app as appuser
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]