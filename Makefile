.PHONY: install test lint typecheck format docker-up docker-down clean build-rust

PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -e ".[dev,stats,data]"

install-all:
	$(PIP) install -e ".[all]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=quantcore --cov-report=html --cov-report=term

test-integration:
	$(PYTHON) -m pytest tests/ -v -m integration --tb=short

lint:
	$(PYTHON) -m ruff check quantcore/ tests/

format:
	$(PYTHON) -m ruff check --fix quantcore/ tests/
	$(PYTHON) -m ruff format quantcore/ tests/

typecheck:
	$(PYTHON) -m mypy quantcore/

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

build-rust:
	cd rust_core && maturin develop --release
