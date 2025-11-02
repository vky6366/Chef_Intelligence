FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory
ENV TRANSFORMERS_CACHE=/app/models/cache
ENV HF_HOME=/app/models/cache
RUN mkdir -p /app/models/cache

# Pre-download TinyLlama model during build (this will take time but only once!)
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
import os; \
cache_dir = os.getenv('TRANSFORMERS_CACHE', '/app/models/cache'); \
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'; \
print('📥 Downloading TinyLlama tokenizer...'); \
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir); \
print('📥 Downloading TinyLlama model (2.2GB, this may take several minutes)...'); \
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir); \
print('✅ Model cached successfully!'); \
"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw_recipes data/processed_chunks data/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application with proper module path
WORKDIR /app
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
