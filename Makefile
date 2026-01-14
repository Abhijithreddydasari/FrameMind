# FrameMind Development Commands
# Usage: make <command>

.PHONY: help install dev test lint format clean docker-up docker-down docker-logs worker api

# Default target
help:
	@echo "FrameMind Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make dev          Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make api          Run API server locally"
	@echo "  make worker       Run ARQ worker locally"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code with ruff"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up    Start all services"
	@echo "  make docker-down  Stop all services"
	@echo "  make docker-logs  View logs"
	@echo "  make docker-build Build images"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        Clean generated files"
	@echo "  make models       Download ML models"

# ============ Setup ============

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install || true

# ============ Development ============

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

worker:
	arq src.workers.pipeline.WorkerSettings --watch src

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

# ============ Docker ============

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-rebuild:
	docker compose -f docker/docker-compose.yml build --no-cache

docker-shell-api:
	docker compose -f docker/docker-compose.yml exec api /bin/bash

docker-shell-worker:
	docker compose -f docker/docker-compose.yml exec worker /bin/bash

# Debug mode with Redis Commander
docker-debug:
	docker compose -f docker/docker-compose.yml --profile debug up -d

# ============ Utilities ============

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true

models:
	python scripts/download_models.py

# Create data directories
init:
	mkdir -p data/videos data/frames data/embeddings

# Redis CLI
redis-cli:
	docker compose -f docker/docker-compose.yml exec redis redis-cli

# Database migrations (when using Postgres)
migrate:
	alembic upgrade head

migrate-create:
	@read -p "Migration name: " name; alembic revision --autogenerate -m "$$name"
