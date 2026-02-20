# CORP - Crypto Options Research Platform
# Docker Image

FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e "."

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 corp && \
    chown -R corp:corp /app
USER corp

# Create necessary directories
RUN mkdir -p logs data/cache

# Expose ports (for Jupyter and API if needed)
EXPOSE 8888 8000

# Default command
CMD ["python", "-c", "print('CORP Platform Ready. Run your strategy or jupyter lab.')"]
