FROM python:3.9-slim

WORKDIR /app

# Set environment variables for memory management
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV TORCH_HOME=/app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory
RUN mkdir -p /app/cache

COPY . .

# Expose the port explicitly
EXPOSE 10000

# Use gunicorn for production
CMD ["gunicorn", "api_server:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:10000", "--timeout", "120"]