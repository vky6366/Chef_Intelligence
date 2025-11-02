# Deployment Guide - Chef Intelligence Method 1

## Local Development

### Prerequisites
- Python 3.11.7
- pip
- Virtual environment (recommended)

### Setup
```bash
# Clone repository
git clone <your-repo-url>
cd chef_intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup project
make setup

# Configure .env file
# Add your TINY_LLAMA_API_KEY

# Add recipe data
# Place your recipes in data/raw_recipes/recipes.txt

# Build index
make index

# Run server
make run
```

## Docker Deployment

```bash
# Build docker-compose
docker-compose up --build 

# Run container
docker run -p 8000:8000 \
  -e TINYLLAMA_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  chef-intelligence:method1
```

## Docker Compose

```bash
# Create .env file with OPENAI_API_KEY
echo "TINYLLAMA_API_KEY=your-key" > .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Logs
```bash
# View logs
tail -f data/logs/chef_intelligence_*.log

# With Docker
docker logs -f <container-id>

## Performance Tuning

- Adjust `TOP_K_RETRIEVAL` in config for speed vs accuracy
- Increase `LLM_MAX_TOKENS` for longer responses
- Use connection pooling for concurrent requests
- Cache frequent queries (future enhancement)

## Security

- Never commit `.env` file
- Use environment variables for secrets
- Enable HTTPS in production
- Implement rate limiting
- Add API authentication (future enhancement)
