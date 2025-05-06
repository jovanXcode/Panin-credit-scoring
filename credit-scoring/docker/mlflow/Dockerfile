FROM python:3.12-slim-bullseye

# Install required system dependencies in a single step
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies efficiently
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# Set up artifacts directory with correct permissions
RUN mkdir -p /opt/artifacts && chmod 777 /opt/artifacts

# CMD Removed (Handled by docker-compose)
