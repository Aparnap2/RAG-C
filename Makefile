.PHONY: install install-dev test test-unit test-integration lint format clean build publish docs-serve docs-build docs-deploy

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pip install pytest pytest-asyncio pytest-cov black isort mypy mkdocs mkdocs-material mkdocstrings[python]

install-all:
	pip install -e ".[all]"

# Testing
test:
	pytest

test-unit:
	pytest tests/unit/ -m "not slow"

test-integration:
	pytest tests/integration/ -m "not slow"

test-coverage:
	pytest --cov=RAG-C --cov-report=html --cov-report=term

# Code quality
lint:
	black --check RAG-C tests
	isort --check-only RAG-C tests
	mypy RAG-C

format:
	black RAG-C tests
	isort RAG-C tests

# Documentation
docs-serve:
	mkdocs serve

docs-build:
	mkdocs build

docs-deploy:
	mkdocs gh-deploy

# Build and publish
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Development
dev-setup: install-dev
	pre-commit install

# Docker
docker-build:
	docker build -t uni-rag:latest .

docker-run:
	docker run -p 8000:8000 uni-rag:latest