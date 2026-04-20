# ToolForge

**Spec-Driven Fine-Tuning for Precision Tool-Calling**

> Fine-tune a Llama 3.2 3B model to achieve 95%+ tool-calling accuracy on a 50-tool
> schema — matching GPT-4o's accuracy at 1/50th the inference cost. Built with
> spec-driven development: 6 behavioral specifications defined before training,
> with a full eval harness that runs as a CI gate.

## Why This Project Exists

In 2026, every company building agentic AI systems hits the same wall: base models
hallucinate tool names, generate wrong argument types, and call tools when they shouldn't.

**The enterprise reality:**

| Scenario | Solution | Trade-off |
|---|---|---|
| Proprietary models (GPT-4o, Claude) | Tool-calling works out-of-box | $$$ at scale, latency, data privacy |
| Open-source models (Llama, Mistral) | Need fine-tuning for tool-calling quality | Engineering investment, but 10-100x cheaper at scale |

ToolForge targets **Tier 2 inference** — the 80% of agentic traffic that needs fast,
accurate tool routing at minimal cost. Fine-tuning a 3B model for this layer delivers
the best cost-quality-latency trade-off for production deployments.

## Approach: Spec-Driven Development

Instead of the typical "train and hope" approach, ToolForge defines success **before** training:

```
Traditional:  Train → Evaluate → "I guess it's better?"
Spec-Driven:  Define specs → Measure baseline → Train → Verify specs pass
```

### Behavioral Specs

| Spec | Metric | Threshold | Priority | What It Measures |
|------|--------|-----------|----------|------------------|
| `tool_selection` | exact_match | 0.95 | Critical | Picks the correct tool name |
| `argument_accuracy` | json_schema_match | 0.90 | Critical | Extracts correct argument values & types |
| `hallucination_resistance` | tool_exists_check | 0.99 | Critical | Never invents nonexistent tools |
| `relevance_detection` | correct_refusal_rate | 0.92 | High | Knows when NOT to call a tool |
| `multi_tool_sequencing` | sequence_exact_match | 0.85 | High | Correct multi-step tool ordering |
| `error_recovery` | graceful_error_rate | 0.88 | Medium | Handles tool errors gracefully |

## Project Stages

### Stage 1: Foundation & Specs ← **CURRENT**

**Goal:** Production-grade project scaffold with behavioral specs and eval harness.

**What was built:**
- Project structure with clean separation: `configs/` (YAML) · `src/` (code) · `tests/` (pytest)
- `pyproject.toml` with optional dependency groups (`[train]`, `[mlx]`, `[serve]`, `[dev]`)
- 6 behavioral spec YAML files with documented thresholds and rationale
- Eval harness: spec loader (Pydantic-validated), metric functions, model-agnostic runner
- 45 unit tests covering spec validation, all 6 metrics, and edge cases
- CLI entry point (`toolforge eval specs`, `toolforge eval run`, etc.)
- Containerfile (Podman) with multi-stage build
- Makefile with all common operations

**Key design decisions:**
- **Pydantic for spec validation** — catch invalid YAML at load time, not at evaluation time
  (same "validate at boundary" pattern used in the Lumina project)
- **Metric registry pattern** — adding a new metric requires only: write the function, add to enum,
  register in `METRIC_REGISTRY`. No framework changes needed.
- **Model-agnostic harness** — the harness accepts any `ModelFn` callable. Whether we evaluate
  a HuggingFace model, MLX model, or Ollama API, the harness doesn't change.
- **Python 3.12** — chosen for ML library compatibility over bleeding-edge 3.14

**How to verify:**
```bash
# Setup
make setup  # or: python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# Run tests (45 should pass)
make test

# Validate specs
toolforge eval specs
```

### Stage 2: Data Engineering (Next)

**Goal:** Download, curate, and format tool-calling datasets for SFT training.

- Download open-source datasets (Glaive Function Calling v2, BFCL, Gorilla)
- Data validation pipeline with schema enforcement
- Train/val/test split with stratification by tool type
- Format conversion to ChatML/Llama chat template
- Generate evaluation datasets for each behavioral spec

### Stage 3: Baseline Evaluation

**Goal:** Run all 6 specs against the base Llama 3.2 3B model — establish the "before" numbers.

- Load base model with HuggingFace transformers
- Implement the `ModelFn` adapter for HuggingFace inference
- Run full spec suite and publish baseline scores
- All 6 specs expected to **FAIL** (confirming fine-tuning is needed)

### Stage 4: Supervised Fine-Tuning (SFT)

**Goal:** QLoRA fine-tuning on the curated dataset, targeting the 3 critical specs.

- QLoRA configuration (4-bit quantization, LoRA rank/alpha tuning)
- Training with HuggingFace TRL `SFTTrainer` + PEFT
- Hyperparameter sweep tracked with W&B
- Checkpoint selection based on spec pass rate (not just loss)
- Expected: 4-5/6 specs pass after SFT

### Stage 5: Preference Tuning (DPO)

**Goal:** DPO training targeting the remaining failing specs.

- Generate preference pairs from SFT model failures
- DPO training on SFT checkpoint using TRL `DPOTrainer`
- Expected: 6/6 specs pass after DPO
- Full comparison report: Base → SFT → SFT+DPO

### Stage 6: Serving & Demo

**Goal:** Production inference API with interactive demo.

- Merge LoRA adapters into base model
- FastAPI inference endpoint with tool-calling schema
- Streamlit/Gradio interactive demo
- Model card with training curves and spec results
- Podman deployment

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.12 | ML ecosystem compatibility |
| Validation | Pydantic v2 | Schema enforcement at boundaries |
| Training | TRL + PEFT + bitsandbytes | Industry standard for QLoRA/DPO |
| Tracking | Weights & Biases | Training curves, hyperparameter comparison |
| CLI | Typer | Auto-generated help, type-safe arguments |
| Testing | pytest | Spec validation + metric correctness |
| Containers | Podman | Rootless, daemonless, OCI-compliant |
| Formatting | Ruff | Single tool replaces black + isort + flake8 |
| Base Model | Llama 3.2 3B | Fast iteration on M1 Pro, upgradeable to 7B |

## Project Structure

```
toolforge/
├── configs/               # All configuration (YAML, not code)
│   ├── specs/             # Behavioral specifications (6 specs)
│   ├── data/              # Dataset processing config
│   ├── training/          # SFT and DPO hyperparameters
│   └── serving/           # Inference server config
├── src/toolforge/         # Source code (pip-installable package)
│   ├── eval/              # Eval harness, metrics, spec runner
│   ├── data/              # Data download, validation, preparation
│   ├── training/          # SFT and DPO training loops
│   └── serving/           # FastAPI inference endpoint
├── tests/                 # pytest (unit + integration)
├── scripts/               # Runnable entry points
├── data/                  # Local data (gitignored)
├── artifacts/             # Models and reports (gitignored)
├── Containerfile          # Podman multi-stage build
├── Makefile               # Common operations
└── pyproject.toml         # Single source of truth for deps
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/shawnk1188/toolforge.git
cd toolforge
make setup

# Activate virtual environment
source .venv/bin/activate

# Validate specs are well-formed
toolforge eval specs

# Run all tests
make test

# See all available commands
toolforge --help
```

## License

MIT

## Author

Sushanth ([github.com/shawnk1188](https://github.com/shawnk1188))
