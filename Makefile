# Makefile for Enhanced RL Portfolio Optimization

.PHONY: help install install-dev test lint format docker-build docker-up docker-down clean

help:
	@echo "Enhanced RL Portfolio Optimization - Available Commands:"
	@echo ""
	@echo "  make install       Install dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linting"
	@echo "  make format        Format code"
	@echo ""
	@echo "  make docker-build  Build Docker images"
	@echo "  make docker-up     Start Docker services"
	@echo "  make docker-down   Stop Docker services"
	@echo "  make docker-logs   View Docker logs"
	@echo ""
	@echo "  make train         Train RL agents"
	@echo "  make evaluate      Evaluate strategies"
	@echo "  make api           Start API server"
	@echo ""
	@echo "  make clean         Clean temporary files"

# Installation
install:
	pip install -r requirements.txt
	pip install -r requirements-prod.txt

install-dev: install
	pip install pytest pytest-cov black flake8 mypy

# Testing
test:
	pytest tests/ -v --cov=code --cov-report=html

lint:
	flake8 code/ production/ --max-line-length=100
	mypy code/ production/

format:
	black code/ production/ tests/

# Docker Commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services started!"
	@echo "API: http://localhost:8000"
	@echo "Jupyter: http://localhost:8888"
	@echo "Grafana: http://localhost:3000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

# Training & Evaluation
train:
	python code/train.py

evaluate:
	python code/evaluate.py

figures:
	python code/figure_generation.py

# Analysis Commands
analyze-costs:
	python code/transaction_cost_analysis.py

analyze-reward:
	python code/reward_ablation.py

analyze-regime:
	python code/regime_analysis.py

# API Commands
api:
	uvicorn production.api:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	uvicorn production.api:app --host 0.0.0.0 --port 8000 --workers 4

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage

clean-data:
	rm -rf data/*
	rm -rf models/*
	rm -rf results/*

# Database Commands
db-init:
	docker exec -it rl-portfolio-db psql -U portfolio_user -d portfolio_db -f /docker-entrypoint-initdb.d/init.sql

db-backup:
	docker exec rl-portfolio-db pg_dump -U portfolio_user portfolio_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore:
	@read -p "Enter backup file name: " file; \
	docker exec -i rl-portfolio-db psql -U portfolio_user portfolio_db < $$file

# Development Commands
notebook:
	jupyter notebook notebooks/

shell:
	docker exec -it rl-portfolio-api bash

# All-in-one commands
setup: install docker-build docker-up
	@echo "Setup complete! Run 'make train' to train models."

full-pipeline: train evaluate figures
	@echo "Full pipeline complete! Check results/ directory."

# Help for specific features
help-transaction-costs:
	@echo "Transaction Cost Analysis:"
	@echo "  1. Run: make analyze-costs"
	@echo "  2. Check results/transaction_cost_analysis/"
	@echo "  3. View notebook: notebooks/transaction_cost_analysis.ipynb"

help-regime:
	@echo "Market Regime Analysis:"
	@echo "  1. Run: make analyze-regime"
	@echo "  2. Check results/regime_analysis/"
	@echo "  3. View notebook: notebooks/regime_analysis.ipynb"
