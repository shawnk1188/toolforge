# ============================================================
# ToolForge Makefile
# ============================================================
# Why a Makefile?
#   Every production project needs a "cheat sheet" of common commands.
#   New contributors run `make setup` and they're ready.
#   CI runs `make test && make lint` — same commands as local dev.
# ============================================================

.PHONY: help setup install lint format test test-unit test-integration \
        eval-baseline specs clean build-container run-container

# Default target — show help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Setup & Installation
# ============================================================

PIP_INDEX := --index-url https://repository.walmart.com/repository/pypi-proxy/simple/ --trusted-host repository.walmart.com

setup: ## Full project setup (create venv, install all deps)
	python3.12 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install $(PIP_INDEX) setuptools wheel
	.venv/bin/pip install $(PIP_INDEX) pyyaml pydantic rich typer pytest pytest-cov ruff
	.venv/bin/pip install --no-build-isolation --no-deps -e .
	@echo "\n✅ Setup complete. Activate with: source .venv/bin/activate"

install: ## Install core + dev dependencies (assumes venv is active)
	pip install $(PIP_INDEX) pyyaml pydantic rich typer pytest pytest-cov ruff

install-train: ## Install training dependencies
	pip install $(PIP_INDEX) peft trl bitsandbytes accelerate wandb scipy transformers tokenizers datasets

install-mlx: ## Install MLX dependencies (Apple Silicon)
	pip install $(PIP_INDEX) mlx mlx-lm

# ============================================================
# Code Quality
# ============================================================

lint: ## Run linter (ruff)
	ruff check src/ tests/

format: ## Auto-format code (ruff)
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run mypy type checking
	mypy src/toolforge/

# ============================================================
# Testing
# ============================================================

test: ## Run all tests
	pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests (requires model downloads)
	pytest tests/integration/ -v --tb=short -m integration

test-cov: ## Run tests with coverage
	pytest tests/ -v --tb=short --cov=toolforge --cov-report=term-missing

# ============================================================
# Evaluation & Specs
# ============================================================

specs: ## Validate all behavioral specs are well-formed
	python -m toolforge.eval.specs validate

eval-baseline: ## Run behavioral specs against base model (before fine-tuning)
	python -m toolforge.eval.harness --config configs/specs/ --stage baseline

eval-sft: ## Run behavioral specs after SFT
	python -m toolforge.eval.harness --config configs/specs/ --stage sft

eval-dpo: ## Run behavioral specs after DPO
	python -m toolforge.eval.harness --config configs/specs/ --stage dpo

eval-all: ## Run specs against all stages and generate comparison report
	python -m toolforge.eval.harness --config configs/specs/ --stage all

# ============================================================
# Data
# ============================================================

data-download: ## Download raw datasets from HuggingFace
	python -m toolforge.data.download

data-prepare: ## Process raw data into training format
	python -m toolforge.data.prepare --config configs/data/default.yaml

data-validate: ## Validate processed dataset quality
	python -m toolforge.data.validate --config configs/data/default.yaml

data-stats: ## Print dataset statistics
	python -m toolforge.data.prepare --stats-only

# ============================================================
# Training
# ============================================================

train-sft: ## Run supervised fine-tuning
	python -m toolforge.training.sft --config configs/training/sft.yaml

train-dpo: ## Run DPO preference tuning
	python -m toolforge.training.dpo --config configs/training/dpo.yaml

# ============================================================
# Serving
# ============================================================

serve: ## Start inference API server
	uvicorn toolforge.serving.api:app --host 0.0.0.0 --port 8000 --reload

# ============================================================
# Containers (Podman)
# ============================================================

build-container: ## Build container image with Podman
	podman build -t toolforge:latest -f Containerfile .

run-container: ## Run container with Podman
	podman run --rm -it \
		-v ./data:/app/data:z \
		-v ./artifacts:/app/artifacts:z \
		-v ./configs:/app/configs:ro,z \
		-p 8000:8000 \
		toolforge:latest

# ============================================================
# Cleanup
# ============================================================

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned build artifacts"

clean-all: clean ## Remove everything including venv and data
	rm -rf .venv/ data/raw/ data/processed/ data/eval/ artifacts/
	@echo "⚠️  Cleaned all data and virtual environment"
