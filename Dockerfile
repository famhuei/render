FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV TORCH_HOME=/app/cache
ENV PORT=10000
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory
RUN mkdir -p /app/cache

# Copy application code
COPY . .

# Expose the port explicitly
EXPOSE 10000

# Use gunicorn with uvicorn worker
CMD gunicorn api_server:app --bind 0.0.0.0:10000 --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 120 --keep-alive 5 --log-level info