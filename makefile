.PHONY: install setup run test clean help

help:
	@echo "Chef Intelligence - Method 1"
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make setup      - Setup project (create directories, .env)"
	@echo "  make index      - Build BM25 index"
	@echo "  make run        - Run FastAPI server"
	@echo "  make test       - Run all tests"
	@echo "  make pipeline   - Run end-to-end pipeline test"
	@echo "  make analyze    - Analyze performance"
	@echo "  make clean      - Clean generated files"

install:
	pip install -r requirements.txt

setup:
	mkdir -p data/raw_recipes data/processed_chunks data/memory_store data/logs
	cp .env.example .env
	@echo "✓ Setup complete. Please edit .env with your API keys"

index:
	python scripts/build_index.py

run:
	python app/main.py

test:
	pytest tests/ -v

pipeline:
	python scripts/run_pipeline.py

analyze:
	python scripts/analyse_performance.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
