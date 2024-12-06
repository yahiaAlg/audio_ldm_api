# Use Python slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    HF_HOME=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    HF_DATASETS_CACHE=/app/cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir numpy==1.23.5 && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory and set permissions
RUN mkdir -p /app/cache && \
    chmod -R 777 /app/cache && \
    chown -R 1000:1000 /app/cache

# Run as non-root user with UID 1000
USER 1000

# Expose port
EXPOSE 8000

# Start script
CMD ["python3", "start.py"]