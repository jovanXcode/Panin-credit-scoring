FROM python:3.12.4-slim

# Set the working directory
WORKDIR /api

# Install system dependencies in a single step
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
COPY ./docker/api/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# Set up necessary directories
RUN mkdir -p /opt/artifacts /api/dataset && chmod -R 777 /opt/artifacts /api/dataset

# Copy application files (grouped to minimize layers)
COPY ./src /api/src
COPY ./alembic.ini /api/
COPY ./migration /api/migration
