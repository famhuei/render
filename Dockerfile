FROM python:3.9-slim

WORKDIR /app

# Set environment variables for memory management
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV TORCH_HOME=/app/cache
ENV PORT=10000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory
RUN mkdir -p /app/cache

COPY . .

# Expose the port explicitly
EXPOSE 10000

# Use uvicorn directly with optimized settings
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1", "--limit-concurrency", "1", "--timeout-keep-alive", "30"]