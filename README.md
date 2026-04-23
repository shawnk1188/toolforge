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

### Stage 1: Foundation & Specs ✅

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

### Stage 2: Data Engineering ✅

**Goal:** Download, curate, and format tool-calling datasets for SFT training.

**What was built:**
- **Canonical data format** (`schema.py`) — Pydantic models for `ToolCallingExample`, `ToolDefinition`, `ToolCall` with 4 example types (SINGLE_TOOL, MULTI_TOOL, NO_TOOL, ERROR_HANDLING). Model validators catch referencing nonexistent tools at parse time.
- **Download pipeline** (`download.py`) — Downloads from NousResearch/Hermes Function Calling v1 (3 configs: singleturn, multiturn, glaive). Hermes converter handles `<tool_call>` XML format. Brace-counting JSON parser for nested arguments.
- **Validation pipeline** (`validate.py`) — 5 composable quality checks (query length, tool definitions, tool count, argument types, empty names). Content-hash deduplication. Rich quality report tables.
- **Stratified splitting** (`prepare.py`) — Train/val/test split preserving type distribution. Synthesized error-handling examples from normal tool calls. Generates all 6 eval datasets for behavioral specs.
- **Chat template formatter** (`formatter.py`) — Llama 3.2 chat template with correct special tokens (BOS/EOS/EOT/header tokens). Training format (full sequence) and inference format (prompt-only). Token count estimation.
- **CLI wired up** — `toolforge data download`, `toolforge data prepare`, `toolforge data validate`, `toolforge data format`
- **224 unit tests** covering schema validation, download parsing, quality checks, stratified splitting, and chat template formatting
- **Data config** (`configs/data/default.yaml`) — all pipeline parameters centralized

**Dataset summary:**

| Source | Config | Raw | After Validation |
|--------|--------|-----|-----------------|
| Hermes singleturn | `func_calling_singleturn` | 1,893 | 973 |
| Hermes multiturn | `func_calling` | 1,893 | 973 |
| Hermes Glaive | `glaive_func_calling` | 5,209 | 1,294 |
| **Total** | | **8,995** | **3,240** |

**Split distribution:**

| Split | Examples | single_tool | multi_tool | no_tool |
|-------|----------|-------------|------------|---------|
| Train | 2,757 | 1,611 | 899 | 247 |
| Val | 323 | 189 | 105 | 29 |
| Test | 160 | 94 | 52 | 14 |

**Key design decisions:**
- **NousResearch/Hermes over Glaive** — Glaive's original dataset went behind auth in 2025. Hermes includes the same Glaive data plus higher-quality curated examples.
- **Aggressive validation** — 5,755 examples removed for short queries, deeply nested args, and duplicates. Quality > quantity for fine-tuning.
- **Synthesized error examples** — No open-source error-handling dataset exists. We create 200 error examples from normal tool calls by removing tool calls and adding error context to queries.
- **Brace-counting JSON parser** — Regex `\{.*?\}` fails on nested JSON. Our parser tracks brace depth and respects quoted strings — handles real-world Glaive/Hermes data correctly.

**How to verify:**
```bash
# Run the full data pipeline (downloads ~9K examples, produces 3.2K validated)
toolforge data prepare

# Validate a specific file
toolforge data validate data/processed/train.jsonl

# Format for Llama 3.2 training
toolforge data format data/processed/train.jsonl

# Run all 334 tests
make test
```

### Stage 3: Baseline Evaluation ✅

**Goal:** Run all 6 specs against the base Llama 3.2 3B model — establish the "before" numbers.

**What was built:**
- **MLX model adapter** (`eval/models.py`) — Apple Silicon native inference via mlx-lm. Loads Llama 3.2 3B 4-bit (~1.8GB) and runs at ~40 tokens/sec on M1 Pro. Adapter Pattern: pluggable backends (MLX, Ollama, Dummy) behind uniform `ModelFn` interface.
- **Prompt builder** — Constructs Llama 3.2 chat template prompts from eval dataset fields (user query + system prompt + tool schema). Separate from Stage 2's formatter because it works with raw dicts rather than Pydantic models.
- **Output parser** — Extracts structured tool calls from raw model text using multi-strategy parsing: direct JSON → JSON extraction (brace counting) → JSON array → text fallback. Normalizes OpenAI-style, our format, and function_call wrapper formats.
- **CLI wired up** — `toolforge eval run --backend mlx` runs the full spec suite
- **264 unit tests** (40 new for model adapters, parsers, and factory)

**Baseline Results (Llama 3.2 3B Instruct, 4-bit quantized, zero-shot):**

| Spec | Score | Threshold | Status | Analysis |
|------|-------|-----------|--------|----------|
| `tool_selection` | **0.920** | 0.95 | ❌ FAIL | Close! Picks right tool 92% of the time but misses edge cases |
| `argument_accuracy` | **0.560** | 0.90 | ❌ FAIL | Main weakness — often gets argument values/types wrong |
| `hallucination_resistance` | **1.000** | 0.99 | ✅ PASS | Perfect! Never invents nonexistent tools |
| `relevance_detection` | **0.256** | 0.92 | ❌ FAIL | Calls tools when it shouldn't — poor refusal behavior |
| `multi_tool_sequencing` | **0.000** | 0.85 | ❌ FAIL | Cannot produce JSON arrays — needs format training |
| `error_recovery` | **0.000** | 0.88 | ❌ FAIL | No graceful error handling — needs behavioral training |

**Result: 1/6 specs pass.** This confirms fine-tuning is needed and identifies exactly WHERE:
- **Already strong:** Tool selection (92%), hallucination resistance (100%)
- **Needs SFT:** Argument accuracy (56%), relevance detection (26%)
- **Needs format training:** Multi-tool (0%), error recovery (0%)

**Key design decisions:**
- **MLX over HuggingFace transformers** — 10x faster on Apple Silicon (unified memory vs CPU fallback). 50 samples × 6 specs completed in 15 minutes.
- **4-bit quantized model** — Same model we'll fine-tune with QLoRA. Baseline measures the EXACT starting point.
- **temperature=0.0** — Deterministic greedy decoding for reproducible baselines. No sampling randomness.
- **Adapter Pattern** — Adding a new backend (e.g., vLLM for cloud) requires only implementing `load()` and `generate()`.

**How to verify:**
```bash
# Run baseline with 50 samples per spec (~15 min on M1 Pro)
toolforge eval run --backend mlx --max-samples 50

# Quick smoke test with 5 samples (~2 min)
toolforge eval run --backend mlx --max-samples 5

# Test with dummy adapter (instant, all specs fail)
toolforge eval run --backend dummy --max-samples 5

# Run all 264 tests
make test
```

### Stage 4: Supervised Fine-Tuning (SFT) ✅ ← **CURRENT**

**Goal:** LoRA fine-tuning on the curated dataset using MLX native training on Apple Silicon.

**What was built:**
- **MLX format converter** (`data/mlx_format.py`) — Converts canonical ToolCallingExample format to OpenAI chat format for MLX's ChatDataset. The tokenizer's `apply_chat_template` handles Llama 3.2 special tokens automatically — more correct than manual template formatting.
- **SFT training module** (`training/sft.py`) — `SFTConfig` dataclass loaded from YAML, cosine decay LR schedule with warmup, gradient checkpointing for Apple Silicon memory safety, SSL proxy patching for corporate environments.
- **Training config** (`configs/training/sft.yaml`) — Fully documented hyperparameters with rationale for every choice.
- **LoRA adapter evaluation** — MLX adapter extends `MLXModelAdapter` with `adapter_path` parameter. Eval harness loads base model + LoRA weights for spec evaluation.
- **334 unit tests** (70 new for MLX format conversion and SFT config)
- **CLI wired up** — `toolforge train sft` with `--iters`, `--batch-size`, `--lr`, `--skip-data-prep` overrides

**Training Details:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Framework | MLX LoRA (Apple Silicon native) | 10x faster than CPU HuggingFace on M1/M2/M3 |
| Model | Llama 3.2 3B Instruct 4-bit | 1.8GB base + 7MB LoRA adapters |
| LoRA rank | 8 | Good quality/speed trade-off for 3B model |
| LoRA scale (alpha) | 20.0 | Amplifies LoRA signal (alpha/rank = 2.5) |
| LR schedule | Cosine decay (1e-5 → 0) | Warmup 50 steps, prevents early instability |
| Iterations | 1000 | Full convergence on 2.7K training examples |
| Batch size | 2 | Memory-safe for 32GB unified memory |
| Gradient checkpointing | ✅ | Essential for Metal GPU memory management |
| Peak memory | 7.6 GB | Comfortable on 32GB M1 Pro |
| Training time | ~5.7 hours | On M1 Pro 32GB |

**Training Curve (val loss):**
```
Iter    1: 1.616  (starting point)
Iter  100: 0.056  (rapid convergence)
Iter  300: 0.038  (best checkpoint)
Iter  500: 0.041  (stable plateau)
Iter 1000: 0.084  (final — 95% reduction from start)
```

**SFT Results vs Baseline (n=50):**

| Spec | Baseline | SFT | Threshold | Status | Δ |
|------|----------|-----|-----------|--------|---|
| `tool_selection` | 0.920 | **0.960** | 0.95 | ✅ PASS | +4.3% |
| `hallucination_resistance` | 1.000 | **1.000** | 0.99 | ✅ PASS | maintained |
| `argument_accuracy` | 0.560 | **0.800** | 0.90 | ❌ FAIL | **+42.9%** |
| `relevance_detection` | 0.256 | **0.186** | 0.92 | ❌ FAIL | -27.3% |
| `multi_tool_sequencing` | 0.000 | **0.000** | 0.85 | ❌ FAIL | — |
| `error_recovery` | 0.000 | **0.060** | 0.88 | ❌ FAIL | +6.0% |

**Result: 2/6 specs pass** (up from 1/6 baseline). SFT pushed tool_selection over its 0.95 threshold and boosted argument_accuracy by 43% (0.560 → 0.800). Error recovery showed its first signs of life (0% → 6%). The remaining specs need preference tuning (DPO) where the model learns from its own mistakes.

**Analysis — what SFT can and can't do:**
- **SFT excels at:** Learning output FORMAT (JSON tool calls), improving tool NAME selection, getting argument values closer to correct. argument_accuracy jumped 43% purely from seeing correct examples.
- **SFT struggles with:** BEHAVIORAL changes (knowing when NOT to call tools, graceful error recovery). Relevance detection actually regressed (-27%) — the model became MORE eager to call tools after training on tool-calling examples.
- **Multi-tool sequencing at 0%:** The model still can't produce JSON arrays. This is a format issue that DPO with targeted preference pairs can address.
- **Error recovery at 6%:** First signs of learning. The synthesized error examples in training data are starting to take effect, but the model needs contrastive training to fully learn this behavior.

**Key design decisions:**
- **MLX native LoRA over QLoRA+bitsandbytes** — bitsandbytes requires CUDA GPUs. MLX LoRA runs natively on Apple Silicon unified memory with comparable quality.
- **Gradient checkpointing + cache clearing** — Metal GPU memory management is less granular than CUDA. Without these, training OOMs at 32GB. With them, peak stays at 7.6GB.
- **mask_prompt: true** — Only compute loss on assistant responses, not on system/user messages. The model learns to GENERATE tool calls, not to parrot prompts.
- **Cosine decay with warmup** — Warmup prevents early instability when LoRA weights are random. Cosine decay provides smooth convergence.
- **5 checkpoint saves** — Every 200 iterations. Allows selecting the best checkpoint (iter 300 had lowest val loss) if the final checkpoint overfits.

**How to verify:**
```bash
# Run SFT training (full: ~6 hours on M1 Pro, quick test: ~5 min)
toolforge train sft                          # full 1000 iterations
toolforge train sft --iters 10              # quick 10-iteration test

# Evaluate SFT checkpoint
toolforge eval run --backend mlx --adapter-path artifacts/sft/adapters --max-samples 20

# Run all 334 tests
make test
```

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
| Training | MLX LoRA (Apple Silicon native) | 10x faster than CPU on M1/M2/M3 |
| Tracking | Built-in loss logging | Training curves via mlx-lm reports |
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

# Run all 334 tests
make test

# See all available commands
toolforge --help
```

## License

MIT

## Author

Sushanth ([github.com/shawnk1188](https://github.com/shawnk1188))
