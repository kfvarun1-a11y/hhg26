FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install python dependencies cleanly
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir --root-user-action=ignore pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy application source code
COPY . .

# Expose default port
EXPOSE 8000

# Run FastAPI via Uvicorn with dynamic Render $PORT support
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
