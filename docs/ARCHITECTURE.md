# ToolForge Architecture & System Design

> A comprehensive deep-dive into the system architecture, engineering decisions, and lessons learned from building ToolForge -- a spec-driven fine-tuning pipeline that takes Llama 3.2 3B from 1/6 behavioral specs passing to production-quality tool-calling.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation & Problem Statement](#2-motivation--problem-statement)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Data Pipeline Architecture](#4-data-pipeline-architecture)
5. [Training Architecture](#5-training-architecture)
6. [Evaluation Architecture (Spec-Driven Development)](#6-evaluation-architecture-spec-driven-development)
7. [Serving Architecture](#7-serving-architecture)
8. [Design Patterns Used](#8-design-patterns-used)
9. [Key Engineering Decisions & Tradeoffs](#9-key-engineering-decisions--tradeoffs)
10. [Lessons Learned & Debugging Stories](#10-lessons-learned--debugging-stories)
11. [Interview-Ready Talking Points](#11-interview-ready-talking-points)

---

## 1. Executive Summary

**ToolForge** is a spec-driven fine-tuning pipeline that transforms a Llama 3.2 3B parameter model from poor tool-calling behavior (1/6 behavioral specs passing) into a production-quality tool-calling model. The system targets **Tier 2 inference** -- the 80% of agentic traffic that needs fast, accurate tool routing at 1/50th the cost of GPT-4o.

**Key result:** Starting from a base Llama 3.2 3B Instruct model that scores 0% on multi-tool sequencing, 0% on error recovery, and 26% on relevance detection, the pipeline applies SFT (Supervised Fine-Tuning) with LoRA adapters followed by DPO (Direct Preference Optimization) to systematically close these gaps.

**What makes this project notable:**

- **Spec-driven development** -- 6 behavioral specifications with quantitative thresholds are defined *before* any training happens. Training is complete when all specs pass. This replaces the typical "train and hope" approach.
- **Custom DPO implementation** -- mlx-lm does not support DPO natively. The DPO training loop is implemented from scratch using MLX's autograd, demonstrating ability to translate research papers (Rafailov et al., 2023) into working code.
- **Apple Silicon native** -- The entire pipeline runs on a single M1 Pro 32GB, with peak memory under 8GB during training. No cloud GPU required.
- **373 unit tests** covering spec validation, all 6 metrics, data pipeline stages, model adapters, and training configs.

**Tech stack:** Python 3.12, MLX, Pydantic v2, Typer CLI, FastAPI, Gradio, pytest, Ruff.

---

## 2. Motivation & Problem Statement

### Why Fine-Tune for Tool-Calling?

In 2026, every company building agentic AI systems hits the same wall: base open-source models hallucinate tool names, generate wrong argument types, and call tools when they should not. The industry has two options:

| Approach | Quality | Cost at Scale | Latency | Data Privacy |
|----------|---------|---------------|---------|--------------|
| GPT-4o / Claude (proprietary) | Excellent | $$$$ | 500ms+ (API) | Data leaves org |
| Llama / Mistral (open, base) | Poor for tool-calling | $ | 50ms (local) | Data stays local |
| **Llama + fine-tuning (ToolForge)** | **Production-quality** | **$** | **50ms (local)** | **Data stays local** |

### Why Not Just Prompt Engineering?

Prompt engineering was the first thing attempted. The baseline evaluation reveals exactly why it is insufficient:

| Spec | Prompt-Engineered Score | Threshold | Gap |
|------|------------------------|-----------|-----|
| tool_selection | 0.920 | 0.95 | -3% |
| argument_accuracy | 0.560 | 0.90 | **-34%** |
| relevance_detection | 0.256 | 0.92 | **-66%** |
| multi_tool_sequencing | 0.000 | 0.85 | **-85%** |
| error_recovery | 0.000 | 0.88 | **-88%** |

The base model with careful prompting (zero-shot, with tool schema in the system prompt and explicit JSON formatting instructions) still scores 0% on multi-tool sequencing because it has never been trained to output JSON arrays. No amount of prompt engineering can teach a model a format it has never seen in its training distribution.

### Why 3B Parameters?

| Factor | 3B Model | 7B Model | 70B Model |
|--------|----------|----------|-----------|
| Inference latency | ~50ms/token | ~120ms/token | Requires GPU cluster |
| Memory footprint (4-bit) | 1.8 GB | 4.2 GB | 35+ GB |
| Training on M1 Pro 32GB | 5.7 hours SFT | 12+ hours SFT | Impossible |
| Fine-tuning effectiveness | Excellent for narrow tasks | Better for general | Overkill for tool routing |

The 3B model is the sweet spot for **tool routing** -- a narrow, well-defined task where the model needs to: (1) pick the right tool name, (2) extract arguments, (3) know when not to call tools. This does not require the world knowledge of a 70B model. The 3B model can be fine-tuned on a laptop in hours and deployed at negligible inference cost.

---

## 3. System Architecture Overview

### High-Level Architecture

```
+------------------------------------------------------------------+
|                         ToolForge Pipeline                        |
+------------------------------------------------------------------+
|                                                                   |
|   +------------------+     +------------------+     +-----------+ |
|   |   DATA PIPELINE  |     |    TRAINING       |     |  SERVING  | |
|   |                  |     |                  |     |           | |
|   |  HuggingFace Hub |     |  SFT (Stage 4)  |     |  FastAPI  | |
|   |       |          |     |       |          |     |  /v1/     | |
|   |  Download &      |     |  SFT Continue    |     |  tool-    | |
|   |  Convert         |     |  (Stage 4b)      |     |  call     | |
|   |       |          |     |       |          |     |           | |
|   |  Validate &      |     |  DPO (Stage 5)  |     |  Gradio   | |
|   |  Deduplicate     |     |                  |     |  Demo     | |
|   |       |          |     +------------------+     +-----------+ |
|   |  Stratified      |              |                     |       |
|   |  Split           |              v                     |       |
|   |       |          |     +------------------+           |       |
|   |  Augment         |     |    ARTIFACTS     |           |       |
|   |       |          |     |                  |<----------+       |
|   |  Format for      |     |  SFT adapters    |                   |
|   |  MLX ChatDataset |     |  DPO adapters    |                   |
|   +------------------+     |  Training logs   |                   |
|            |               +------------------+                   |
|            v                        |                             |
|   +------------------+              v                             |
|   |   EVAL HARNESS   |<- - - - - - - - - - - - - - - - - - - - - |
|   |                  |                                            |
|   |  6 Behavioral    |     Runs after each training stage         |
|   |  Specs (YAML)    |     to measure progress and verify         |
|   |       |          |     spec compliance.                       |
|   |  Metric Registry |                                            |
|   |       |          |                                            |
|   |  Model Adapters  |                                            |
|   |  (MLX, Ollama,   |                                            |
|   |   Dummy)         |                                            |
|   |       |          |                                            |
|   |  Pass/Fail       |                                            |
|   |  Report          |                                            |
|   +------------------+                                            |
+------------------------------------------------------------------+
```

### Module Dependency Graph

```
toolforge/
    cli.py  ---------> eval/harness.py  --> eval/specs.py (BehavioralSpec, Pydantic)
        |                  |                eval/metrics.py (METRIC_REGISTRY)
        |                  |                eval/models.py (BaseModelAdapter + impls)
        |
        +-----------> data/download.py --> data/schema.py (ToolCallingExample, Pydantic)
        |              data/validate.py
        |              data/prepare.py
        |              data/augment.py
        |              data/formatter.py
        |              data/mlx_format.py
        |
        +-----------> training/sft.py  --> data/mlx_format.py
        |              training/dpo.py  --> training/preference.py
        |                                   eval/models.py (for inference)
        |
        +-----------> serving/api.py   --> eval/models.py (reuses adapters)
                       serving/demo.py  --> eval/models.py
```

**Key observation:** The `eval/models.py` module is shared between evaluation AND serving. The same `BaseModelAdapter` abstraction that powers the eval harness also powers the production API. This is intentional -- it guarantees that the model served in production behaves identically to what was evaluated.

### Data Flow: End to End

```
HuggingFace Hub (NousResearch/Hermes)
    |
    v  download.py -- Download 3 configs, convert <tool_call> XML to canonical format
8,995 raw examples
    |
    v  validate.py -- 5 quality checks, content-hash deduplication
3,240 validated examples (5,755 removed for quality)
    |
    v  prepare.py -- Stratified train/val/test split, synthesize eval datasets
2,757 train / 323 val / 160 test + 6 eval datasets
    |
    v  augment.py -- Generate error_handling, no_tool, multi_tool examples
3,557 augmented train (800 synthetic added)
    |
    v  mlx_format.py -- Convert to OpenAI chat format for MLX ChatDataset
data/mlx/train.jsonl (MLX-ready)
    |
    v  sft.py -- LoRA fine-tuning via mlx-lm (1000 iters, ~5.7 hours)
artifacts/sft/adapters (7MB LoRA weights)
    |
    v  sft.py (continue) -- Continue from SFT checkpoint with augmented data
artifacts/sft_v2/adapters
    |
    v  preference.py -- Run SFT model on eval data, collect failures
139 preference pairs (prompt, chosen, rejected)
    |
    v  dpo.py -- Custom DPO training loop (40 iters, ~10 min)
artifacts/dpo/adapters (7MB LoRA weights)
    |
    v  serving/api.py -- Load base model + DPO adapters, serve via FastAPI
http://localhost:8000/v1/tool-call
```

---

## 4. Data Pipeline Architecture

### 4.1 Schema Design

The canonical schema (`src/toolforge/data/schema.py`) is the foundation of the entire pipeline. Every dataset, regardless of its source format, is normalized into `ToolCallingExample` objects validated by Pydantic v2.

**Core types:**

```
ToolCallingExample
    |-- id: str                    # Unique ID for deduplication and tracing
    |-- system_prompt: str         # Instructions + tool definitions
    |-- user_query: str            # Natural language query (min_length=1)
    |-- available_tools: list[ToolDefinition]
    |      |-- name: str           # snake_case validated
    |      |-- description: str
    |      |-- parameters: ToolParameters (JSON Schema subset)
    |-- expected_tool_calls: list[ToolCall]
    |      |-- name: str
    |      |-- arguments: dict[str, Any]
    |-- expected_response: str?    # For no_tool and error_handling cases
    |-- example_type: ExampleType  # SINGLE_TOOL | MULTI_TOOL | NO_TOOL | ERROR_HANDLING
    |-- source_dataset: str
```

**Critical validators:**

1. `must_have_expected_output` -- Every example must have either tool calls OR a text response. Catches incomplete examples at parse time, not during training at 3am.
2. `tool_calls_must_reference_available_tools` -- If the expected response calls `get_weather`, that tool must exist in `available_tools`. This prevents training on impossible examples where the ground truth references a nonexistent tool.

**Why Pydantic v2 over dataclasses:**
- Runtime validation catches bad data at the boundary (load time), not deep in the training loop
- JSON serialization with `.model_dump_json()` is zero-config
- Model validators can enforce cross-field constraints (e.g., tool call references available tool)
- Field-level validators (e.g., `name_must_be_valid_identifier`) prevent subtle formatting issues

The `ExampleType` enum is critical for routing examples to the right eval spec:

| ExampleType | Routes To Spec | What It Tests |
|-------------|---------------|---------------|
| `SINGLE_TOOL` | tool_selection, argument_accuracy, hallucination_resistance | Basic tool-calling |
| `MULTI_TOOL` | multi_tool_sequencing | JSON array output, ordering |
| `NO_TOOL` | relevance_detection | Knowing when NOT to call a tool |
| `ERROR_HANDLING` | error_recovery | Graceful degradation |

### 4.2 Multi-Source Dataset Ingestion

**Source:** NousResearch/Hermes Function Calling v1 from HuggingFace Hub, with 3 configurations:

| Config | Raw Examples | After Validation | Why This Source |
|--------|-------------|-----------------|-----------------|
| `func_calling_singleturn` | 1,893 | 973 | High-quality curated single-turn tool calls |
| `func_calling` (multiturn) | 1,893 | 973 | Multi-turn conversations with tool use |
| `glaive_func_calling` | 5,209 | 1,294 | Volume from Glaive dataset (included in Hermes) |
| **Total** | **8,995** | **3,240** | |

**Why Hermes over Glaive directly:** The original Glaive dataset went behind authentication in 2025. Hermes includes the same Glaive data plus higher-quality curated examples. One download, best of both worlds.

**Conversion challenge:** Hermes uses XML-like `<tool_call>` format with nested JSON arguments. A regex approach (`\{.*?\}` non-greedy match) fails on nested braces like `{"address": {"street": "123 Main", "city": "NYC"}}` because the non-greedy match stops at the first `}`. The solution is a **brace-counting JSON parser** that tracks brace depth and respects quoted strings -- the same technique is reused in the eval output parser.

### 4.3 Quality Validation Pipeline

Five composable quality checks (`validate.py`):

1. **Query length** -- Minimum length filter removes examples like `"hi"` that do not contain enough information for meaningful tool-calling
2. **Tool definitions** -- Every example must have at least one tool defined
3. **Tool count** -- Filters examples with unreasonable numbers of tools (too many = ambiguous)
4. **Argument types** -- Validates that arguments match parameter types defined in the schema
5. **Empty names** -- Catches examples with empty tool names or empty argument keys

**Content-hash deduplication** removes exact duplicates. Of the 8,995 raw examples, 5,755 are removed, leaving 3,240 validated examples. This 64% rejection rate is intentional -- **quality over quantity for fine-tuning**. One hallucinated ground-truth example in training data teaches the model to hallucinate.

### 4.4 Train/Val/Test Split Strategy

Stratified splitting preserves type distribution across splits:

| Split | Total | single_tool | multi_tool | no_tool | error_handling |
|-------|-------|-------------|------------|---------|----------------|
| Train | 2,757 | 1,611 (58%) | 899 (33%) | 247 (9%) | 0 (0%) |
| Val | 323 | 189 | 105 | 29 | 0 |
| Test | 160 | 94 | 52 | 14 | 0 |

**Two critical distribution gaps visible in the training data:**
1. **No error_handling examples (0%)** -- No open-source error-handling tool-calling dataset existed. Error examples were synthesized for eval datasets but NOT initially included in training data (a gap fixed in Stage 4b).
2. **Only 9% no_tool examples** -- The model sees "call a tool" 91% of the time, learning a bias toward always calling tools. This directly causes the relevance_detection failure (0.256 baseline, regressed to 0.186 after SFT).

These gaps are the primary motivation for the data augmentation pipeline.

### 4.5 Data Augmentation Strategy

After Stage 4 SFT revealed the impact of data distribution gaps, the augmentation pipeline (`augment.py`) generates synthetic training data for three underrepresented categories:

**1. Error handling (400 examples):**
- Source: Takes existing `SINGLE_TOOL` examples and transforms them into error scenarios
- Method: Appends `[System: The tool '{name}' was called but returned an error: "{error_msg}"]` to the query
- Expected response: Acknowledgment of error + offer to retry or try alternative
- 12 diverse error messages (503, rate limit, auth failure, timeout, etc.)
- 6 response templates with varied phrasing
- Why needed: Zero error_handling examples in training; model scored 0.000 then 0.060 on error_recovery

**2. No-tool / relevance detection (300 examples):**
- Source: Diverse query templates (general knowledge, greetings, clarifications, math, opinions)
- Method: Pairs irrelevant queries with existing tool definitions. The tools ARE available but the query does not need them.
- 30+ query template patterns with parameterized fill-in values
- Why needed: 91% of training is tool-calling, creating a bias. SFT actually *regressed* relevance_detection from 0.256 to 0.186.

**3. Multi-tool reinforcement (100 examples):**
- Source: Existing multi_tool examples, simplified to 2-tool sequences
- Method: Takes the first 2 tool calls from longer sequences to create shorter, clearer patterns
- Why needed: Despite 899 multi_tool examples, the 3B model scored 0% -- it could not produce JSON arrays. Simpler examples help the smaller model learn the format.

**Post-augmentation distribution:**

| Type | Before | Added | After | % of Total |
|------|--------|-------|-------|------------|
| single_tool | 1,611 | +0 | 1,611 | 45.3% |
| multi_tool | 899 | +100 | 999 | 28.1% |
| no_tool | 247 | +300 | 547 | 15.4% |
| error_handling | 0 | +400 | 400 | 11.2% |
| **TOTAL** | **2,757** | **+800** | **3,557** | **100%** |

### 4.6 Chat Template Formatting

Two formatting paths exist, each serving a different purpose:

**Path 1: Manual Llama 3.2 template (`formatter.py`)**
Used for Stage 2 data inspection and token counting. Manually constructs the chat template with special tokens:
```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

{system prompt with tool JSON}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{tool call JSON or text response}<|eot_id|>
<|end_of_text|>
```

**Path 2: MLX ChatDataset format (`mlx_format.py`)**
Used for actual training. Converts to OpenAI chat message format:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}"}
  ],
  "tools": [{"type": "function", "function": {"name": "get_weather", ...}}]
}
```

**Why two paths:** Path 2 is more correct for training because the tokenizer's `apply_chat_template` handles special tokens automatically. Path 1 exists because it was built in Stage 2 before the MLX training architecture was finalized, and it remains useful for debugging and token estimation without loading the tokenizer.

---

## 5. Training Architecture

### 5.1 SFT (Supervised Fine-Tuning) with LoRA

**Why LoRA over Full Fine-Tuning:**

| Factor | Full Fine-Tuning | LoRA (rank 8) |
|--------|-----------------|---------------|
| Trainable parameters | 3.2B (100%) | 6.9M (0.2%) |
| Memory for training | 24+ GB (FP16) | 7.6 GB (4-bit + LoRA) |
| Training time | 12+ hours | 5.7 hours |
| Adapter size on disk | 6+ GB | 7 MB |
| Risk of catastrophic forgetting | High | Low (base weights frozen) |
| Can run on M1 Pro 32GB | No | Yes |

LoRA (Low-Rank Adaptation) freezes the pretrained weights and injects small trainable matrices into each transformer layer. For a 3B model with rank 8, this means:
- Each attention layer gets 4 LoRA matrices (Q, K, V, O projections)
- Each matrix is factored as A (d x r) * B (r x d) where r=8
- Total: 6.9M trainable parameters out of 509M total (1.4%)

**Hyperparameter choices and rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 8 | Rank 4 too constrained for JSON output patterns. Rank 16+ adds parameters without proportional quality gain on 3B models. |
| LoRA scale (alpha) | 20.0 | Effective ratio alpha/rank = 2.5. Amplifies LoRA signal so updates are not too weak. |
| LoRA dropout | 0.05 | Small dropout prevents overfitting on 2.7K examples. 0.0 risks memorization; 0.1+ risks underfitting. |
| Learning rate | 1e-5 | Standard for LoRA on instruction-tuned models. 5e-5 causes catastrophic forgetting; 1e-6 insufficient in 1000 steps. |
| LR schedule | Cosine decay with 50-step warmup | Warmup prevents early instability when LoRA weights are random. Cosine provides smooth convergence. |
| Iterations | 1000 | batch_size 2 x 1000 = 2000 examples seen = 0.72 epochs. Enough to learn patterns without overfitting. |
| Batch size | 2 | Memory-safe for 32GB unified memory with gradient checkpointing. |
| mask_prompt | true | Only compute loss on assistant responses. Focuses learning on generating tool calls, not memorizing prompts. |
| grad_checkpoint | true | Essential for Metal GPU memory management. Trades ~20% compute time for 40-60% less activation memory. |
| num_layers | 16 | Apply LoRA to all 16 transformer layers. For 3B, overhead is minimal. |
| max_seq_length | 2048 | 95%+ of examples fit. Longer sequences waste memory; overflow is truncated. |

**Training curve (validation loss):**
```
Iter    1: 1.616  (random LoRA initialization)
Iter  100: 0.056  (rapid convergence -- model learns JSON format quickly)
Iter  200: 0.042
Iter  300: 0.038  (best checkpoint)
Iter  400: 0.039
Iter  500: 0.041  (stable plateau)
Iter 1000: 0.084  (final -- slight overfitting, but 95% reduction from start)
```

**SFT results:**

| Spec | Baseline | After SFT | Threshold | Status | Delta |
|------|----------|-----------|-----------|--------|-------|
| tool_selection | 0.920 | **0.960** | 0.95 | PASS | +4.3% |
| hallucination_resistance | 1.000 | **1.000** | 0.99 | PASS | -- |
| argument_accuracy | 0.560 | **0.800** | 0.90 | FAIL | **+42.9%** |
| relevance_detection | 0.256 | **0.186** | 0.92 | FAIL | -27.3% |
| multi_tool_sequencing | 0.000 | **0.000** | 0.85 | FAIL | -- |
| error_recovery | 0.000 | **0.060** | 0.88 | FAIL | +6.0% |

**Analysis of SFT capabilities and limitations:**

- **What SFT excels at:** Learning output FORMAT (JSON tool calls), improving tool name selection accuracy, getting argument values closer to correct. The 43% jump in argument_accuracy is purely from exposure to correct examples.
- **What SFT cannot do:** Change *behavior*. Relevance detection actually **regressed** (-27%) because the model saw tool-calling examples 91% of the time and learned an even stronger bias toward always calling tools. This is a fundamental limitation of SFT -- it teaches imitation, not judgment.
- **Multi-tool at 0%:** Even with 899 multi-tool training examples, the 3B model cannot produce JSON arrays. The examples are too complex (3-5 tools with long arguments) for the model to learn the array format. This motivates the augmentation pipeline's simpler 2-tool reinforcement examples.

### 5.2 SFT Continuation (Stage 4b)

After analyzing the SFT results, the data augmentation pipeline fills the three critical gaps. Rather than retraining from scratch, we **continue from the SFT checkpoint** at a lower learning rate:

| Parameter | Initial SFT | SFT Continuation |
|-----------|------------|------------------|
| Starting point | Random LoRA | SFT checkpoint (adapters.safetensors) |
| Training data | 2,757 original | 3,557 augmented (original + 800 synthetic) |
| Learning rate | 1e-5 | 3e-6 (3x lower -- refining, not relearning) |
| Iterations | 1000 | 500 |
| LR warmup | 50 steps | 0 (already in good optimization region) |
| Output | artifacts/sft/adapters | artifacts/sft_v2/adapters |

**Why continue instead of retrain:** The SFT checkpoint already has two passing specs (tool_selection: 0.96, hallucination_resistance: 1.00). Starting fresh risks losing these capabilities. Continuing at a lower LR preserves existing skills while learning new behaviors from the augmented data.

### 5.3 DPO (Direct Preference Optimization)

**Why DPO after SFT:**
SFT teaches "do this" -- the model imitates correct examples. DPO teaches "do this, NOT that" -- the model learns from contrasts between correct and incorrect responses. The remaining failing specs (relevance_detection, error_recovery, multi_tool_sequencing) are **behavioral** -- the model needs to learn *when* to act differently, not just a new format.

**Mathematical formulation:**

```
L_DPO = -E[ log sigma(beta * (log pi_theta(y_w|x)/pi_ref(y_w|x)
                               - log pi_theta(y_l|x)/pi_ref(y_l|x))) ]
```

Where:
- `x` = prompt (user query + tool schema)
- `y_w` = chosen (correct) response
- `y_l` = rejected (model's actual wrong response)
- `pi_theta` = policy model (being trained)
- `pi_ref` = reference model (frozen SFT checkpoint)
- `beta` = temperature controlling divergence from reference

**Intuition:** Increase the probability of the chosen response relative to the rejected one, while a KL penalty (controlled by beta) prevents the model from straying too far from the reference. This is the key insight -- DPO constrains preference learning so the model does not forget what SFT taught it.

**Preference pair generation (`training/preference.py`):**

The preference pairs come from the SFT model's own failures, not synthetic negatives. The process:

1. Run SFT model on each eval dataset example
2. Compare model output to ground truth
3. Where model fails, create `(prompt, chosen=ground_truth, rejected=model_output)` triple

| Failing Spec | Eval Examples | SFT Correct | Preference Pairs | SFT Accuracy |
|-------------|--------------|-------------|------------------|--------------|
| argument_accuracy | 50 | 40 | 10 | 80% |
| error_recovery | 50 | 6 | 44 | 12% |
| multi_tool_sequencing | 50 | 0 | 50 | 0% |
| relevance_detection | 43 | 8 | 35 | 19% |
| **Total** | **193** | **54** | **139** | -- |

**Why preference pairs from own failures:** More effective than synthetic negatives because the model learns to correct its ACTUAL failure modes. If the model consistently calls `search_web` when it should refuse, it sees its own `search_web` output labeled as "rejected" paired with the correct text-only response as "chosen."

**Memory-efficient DPO implementation:**

Standard DPO loads two full models simultaneously (policy + reference). On M1 Pro 32GB, two copies of Llama 3.2 3B 4-bit would use ~8GB, leaving insufficient memory for forward passes.

**Solution:** Pre-computed reference log probabilities.

```
Phase 1: Load reference model
         -> Forward pass on ALL 139 pairs
         -> Cache ref_chosen_logps[139] and ref_rejected_logps[139]
         -> Unload reference model, gc.collect()

Phase 2: Load policy model (only model in memory now)
         -> Training loop uses cached reference log probs
         -> Per-step: 2 forward passes (not 4)
         -> Peak memory: ~6GB (down from ~12GB)
```

This halves peak memory and reduces per-step computation from 4 forward passes to 2.

### 5.4 The DPO Catastrophic Forgetting Story (v1 Failure and v2 Fix)

**DPO v1 caused catastrophic forgetting.** All specs regressed:
- tool_selection: 0.96 -> 0.40
- argument_accuracy: 0.80 -> 0.06
- hallucination_resistance: 1.00 -> 0.80

**Root causes diagnosed:**

1. **No gradient clipping** -- A single batch with large loss produced huge gradients that destroyed the model's tool_selection capability in one step. Without clipping, gradient norms varied from 0.001 to 500+.

2. **No early stopping** -- Training continued past the point of degradation. With no training curve logging, there was no way to see the model was diverging.

3. **beta=0.1 too aggressive** -- Low beta means weak KL penalty, allowing the model to diverge far from the reference. With only 139 pairs, the model quickly overfitted to preference patterns and forgot everything else.

4. **No training curve logging** -- Could not diagnose whether the model was converging or diverging. The loss value alone was insufficient.

**DPO v2 fixes:**

| Fix | Implementation | Effect |
|-----|---------------|--------|
| Gradient clipping | `max_norm=1.0` -- bounds step size | Prevents single-batch catastrophic updates |
| Early stopping | `patience=50` -- stops if margin plateaus | Prevents training past degradation point |
| Higher beta | 0.1 -> 0.5 -- stronger KL penalty | Model stays closer to SFT reference |
| Label smoothing | 0.1 -- mixes in "wrong" label probability | Prevents overconfidence on 139 training pairs |
| Lower LR | 5e-6 -> 1e-6 -- most conservative | Very gentle preference updates |
| Fewer iterations | 500 -> 40 -- ~0.29 epochs | Less opportunity to overfit |
| Training curve logging | `training_log.json` with loss, margin, accuracy, grad_norm | Full diagnostic visibility |

**The progression of beta tuning:**
- beta=0.1: Catastrophic forgetting (0/6 specs)
- beta=0.3: Still some forgetting with lr=2e-6
- beta=0.5: With grad_clip=1.0 and lr=1e-6, very gentle preference updates preserve SFT skills

### 5.5 The Zero-Gradient Bug (DPO Debugging Story)

One of the most instructive debugging sequences in the project. During DPO training, the loss was stuck at exactly **0.6931** (which is `ln(2)`) for 30+ iterations, with gradient norm = 0.000000.

**Debugging sequence:**

1. **Hypothesis: Learning rate too low.** Increased from 5e-6 to 1e-4. Still zero gradients. Ruled out.

2. **Hypothesis: Gradient flow broken.** Tested `sum(log_probs)` directly -- had nonzero gradients. But `chosen_logp - rejected_logp` had zero gradient. The subtraction was producing exactly 0.

3. **Root cause identified:** `DPODataset.__getitem__()` was right-truncating sequences to `max_length=512`. The tool schema in the prompt consumed >500 tokens. After truncation, the actual responses (where chosen differs from rejected) were cut off. All 139 pairs had `chosen_ids == rejected_ids`, making `reward_margin = 0` exactly, with zero gradient by mathematical necessity.

4. **Fix: Left-truncation.** Truncate the prompt from the left (keeping the user query at the end), ensuring the response is preserved in full. After fix: 224/224 gradient tensors had nonzero values.

**Why this is a great debugging story for interviews:**
- Shows systematic hypothesis testing (not random guessing)
- The "stuck at ln(2)" signal is mathematically meaningful (sigmoid(0) = 0.5, -log(0.5) = ln(2))
- The root cause is subtle and data-dependent (truncation removing the signal)
- The fix is elegant (left-truncation preserves what matters)

---

## 6. Evaluation Architecture (Spec-Driven Development)

### 6.1 The Spec-Driven Philosophy

Traditional ML:
```
Train -> Evaluate -> "I guess it's better?"
```

ToolForge spec-driven approach:
```
Define specs (what "done" looks like)
    -> Measure baseline (how bad is it now?)
    -> Train (close the gap)
    -> Verify specs pass (did we actually succeed?)
```

Specs are YAML files that define:
- **What** to measure (metric name)
- **How well** it must perform (threshold)
- **On what data** (eval dataset path)
- **How many** samples (statistical significance)
- **How important** it is (priority: critical > high > medium)

This approach has three key benefits:
1. **Training has a clear stopping condition** -- not "run for N epochs and see."
2. **Regressions are immediately visible** -- if SFT improves argument_accuracy but regresses relevance_detection, the report shows it.
3. **No subjective quality judgments** -- "is the model better?" becomes "do all specs pass?"

### 6.2 The Six Behavioral Specifications

| Spec | Metric | Threshold | Priority | Rationale |
|------|--------|-----------|----------|-----------|
| `tool_selection` | exact_match | 0.95 | Critical | Most fundamental: picks the right tool name. 5% error = 1 in 20 wrong routes. |
| `argument_accuracy` | json_schema_match | 0.90 | Critical | Correct tool + correct arguments + correct types. Lower threshold because argument extraction is harder. |
| `hallucination_resistance` | tool_exists_check | 0.99 | Critical | Safety-critical: never invent nonexistent tools. Highest threshold because hallucinated tools cause runtime errors. |
| `relevance_detection` | correct_refusal_rate | 0.92 | High | Knows when NOT to call a tool. Lower than tool_selection because ambiguity is inherent. |
| `multi_tool_sequencing` | sequence_exact_match | 0.85 | High | Correct multi-step tool ordering. Lowest tool-calling threshold because planning is hardest for 3B models. |
| `error_recovery` | graceful_error_rate | 0.88 | Medium | Handles tool errors gracefully. Uses heuristic evaluation (not LLM-as-judge) for reproducibility. |

**Why these specific thresholds:**
- 0.95 (tool_selection): Production routing needs near-perfect accuracy. 5% margin for genuinely ambiguous edge cases.
- 0.90 (argument_accuracy): Entity extraction and type inference are harder. 10% margin for complex argument schemas.
- 0.99 (hallucination_resistance): Safety-critical. Even 1% hallucination at scale means hundreds of runtime errors daily.
- 0.92 (relevance_detection): 8% margin for queries that are genuinely ambiguous about whether a tool is needed.
- 0.85 (multi_tool_sequencing): Most ambitious. Multi-step planning is where 3B models struggle most. Production systems often complement with a planning layer.
- 0.88 (error_recovery): Subjective metric. Heuristic evaluation has inherent variance. 12% margin accounts for this.

### 6.3 Metric Implementations

Every metric function has the same signature:
```python
(predicted: dict, expected: dict, tool_schema: dict) -> bool
```

Returns `True` if this single example passes. The harness computes the aggregate: `score = pass_count / total_count`.

**`exact_match`** -- Did the model pick the right tool?
- Compares tool names case-insensitively
- Handles format variants: `"tool"`, `"name"`, `"function"` keys
- Both `None` = pass (both correctly identified no tool needed)

**`json_schema_match`** -- Did the model get the arguments right?
- First checks tool name (via exact_match)
- Then checks all expected arguments are present and match
- Type coercion tolerance: `"42"` matches `42`, int matches float
- No hallucinated extra arguments (unless in the tool's parameter schema)

**`tool_exists_check`** -- Does the predicted tool actually exist?
- Checks if the tool name exists in the tool schema
- No tool call (text response) always passes
- This is the hallucination detection metric

**`correct_refusal_rate`** -- Did the model correctly refuse to call a tool?
- Only applies to examples where no tool should be called
- Pass: model returns no tool call
- Fail: model forces any tool call

**`sequence_exact_match`** -- Did the model produce the right tool sequence?
- Compares ordered lists of tool calls
- Length must match, order must match, each tool name must match
- No partial credit

**`graceful_error_rate`** -- Did the model handle the error well?
- Heuristic evaluation (not LLM-as-judge) for reproducibility
- Checks for error acknowledgment keywords: "error", "unable", "sorry", "try again"
- Checks for hallucination signals: "here is the", "the result is", "successfully"
- Pass: acknowledges error. Fail: pretends tool call succeeded.

**Adding a new metric requires only three changes:**
1. Write the function with the standard signature
2. Add a variant to the `MetricName` enum
3. Register in `METRIC_REGISTRY`

No framework changes needed. This is the **Registry Pattern**.

### 6.4 Model Adapter Pattern

The eval harness is model-agnostic. It accepts a `ModelFn` callable (`str -> dict`) and does not know or care about the inference backend.

```
BaseModelAdapter (ABC)
    |
    |-- MLXModelAdapter       Apple Silicon native, ~40 tok/s on M1 Pro
    |     |-- load()          mlx_lm.load(model_id, adapter_path?)
    |     |-- generate()      mlx_lm.generate(model, prompt, sampler)
    |
    |-- OllamaModelAdapter    Local API, easy model switching
    |     |-- load()          Verify Ollama is running
    |     |-- generate()      HTTP POST to /api/generate
    |
    |-- DummyModelAdapter     For testing harness without a model
          |-- load()          No-op
          |-- generate()      Returns canned responses

Factory: create_model_adapter("mlx") -> MLXModelAdapter
         create_model_adapter("ollama") -> OllamaModelAdapter
         create_model_adapter("dummy") -> DummyModelAdapter
```

The `__call__` method on `BaseModelAdapter` implements the full pipeline:
1. Lazy load (on first call)
2. `generate(prompt)` -> raw text
3. `parse_output(raw_text)` -> structured dict

**Adding a new backend** (e.g., vLLM for cloud GPUs) requires implementing only `load()` and `generate()`. The parsing logic is inherited.

### 6.5 Output Parsing Strategy

The parser in `BaseModelAdapter.parse_output()` uses a multi-strategy approach because models output tool calls in many formats:

```
Strategy 1: Direct JSON parse
    '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
    -> direct json.loads() succeeds

Strategy 2: JSON array extraction (MUST come before single-object)
    'I need to call two tools: [{"name": "f1"}, {"name": "f2"}]'
    -> extract array first, normalize each element

Strategy 3: Single JSON object extraction (brace counting)
    'Let me call get_weather: {"name": "get_weather", "arguments": {"city": "Tokyo"}} for you'
    -> find first '{', count braces, extract balanced substring

Strategy 4: Fallback to text response
    'I cannot help with that because no suitable tool is available.'
    -> {"response": "I cannot help with that..."}
```

**Critical ordering: Strategy 2 before Strategy 3.**

This ordering fixes a parser bug that originally caused 0% on multi_tool_sequencing. When the model outputs `[{"name": "f1", ...}, {"name": "f2", ...}]`, Strategy 3 (single JSON extraction) would find the first `{` inside the array and return only the first tool call, silently dropping all subsequent tools. Array extraction MUST come first to catch multi-tool responses before the single-object parser consumes only the first element.

### 6.6 Output Normalization

Models produce tool calls in different formats. The `_normalize_parsed()` method converts all variants to the canonical format:

```
Input: {"name": "get_weather", "arguments": {"city": "Tokyo"}}     (OpenAI-style)
       {"tool": "get_weather", "arguments": {"city": "Tokyo"}}     (our format)
       {"function_call": {"name": "get_weather", "arguments": ...}} (older OpenAI)
       {"name": "get_weather", "parameters": {"city": "Tokyo"}}    (alternative)

Output: {"tool": "get_weather", "arguments": {"city": "Tokyo"}}    (canonical)
```

---

## 7. Serving Architecture

### 7.1 FastAPI Inference Server (`serving/api.py`)

**Endpoint design mirrors OpenAI's function calling API** so clients can switch between GPT-4o and ToolForge with minimal code changes.

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check -- verifies model is loaded |
| POST | `/v1/tool-call` | Tool-calling inference |
| GET | `/docs` | Auto-generated OpenAPI documentation |

**Request schema:**
```json
{
  "query": "What's the weather in Tokyo?",
  "tools": [
    {"name": "get_weather", "description": "...", "parameters": {...}}
  ],
  "system_prompt": "optional",
  "max_tokens": 512
}
```

**Response schema:**
```json
{
  "tool_call": {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  "tool_calls": null,
  "response": null,
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "latency_ms": 230.5
}
```

The response has three mutually exclusive fields:
- `tool_call` -- single tool call
- `tool_calls` -- array of tool calls (multi-tool)
- `response` -- text response (no tool needed)

**Architecture detail:** The serving layer reuses `eval/models.py` adapters. The same `MLXModelAdapter` that powers evaluation also powers the production API. This guarantees that served behavior matches evaluated behavior -- there is no "it works in eval but not in prod" gap.

### 7.2 Gradio Interactive Demo (`serving/demo.py`)

The demo provides a visual interface for testing tool-calling with 5 built-in tools:
1. `get_weather` -- Weather lookup by city
2. `search_web` -- Web search
3. `calculate` -- Math expression evaluation
4. `send_email` -- Email sending
5. `get_stock_price` -- Stock price lookup

Example queries test different behaviors:
- "What's the weather in Tokyo?" -> should call `get_weather`
- "Tell me a joke about programming" -> should NOT call any tool
- "Search for the latest AI news" -> should call `search_web`

Users can modify the tools JSON to test custom tool schemas.

---

## 8. Design Patterns Used

### 8.1 Adapter Pattern (Model Backends)

**Problem:** The eval harness needs to evaluate models from different inference engines (MLX, Ollama, cloud APIs) without knowing which engine is running.

**Solution:** `BaseModelAdapter` ABC defines the interface (`load()`, `generate()`, `__call__()`). Each backend implements the interface. The harness works with any `ModelFn` callable.

**Benefit:** Adding a new backend (e.g., vLLM, TensorRT-LLM, OpenAI API) requires only implementing `load()` and `generate()`. No changes to the harness, metrics, or CLI.

### 8.2 Factory Pattern (create_model_adapter)

**Problem:** The CLI and harness should not need to import specific adapter classes.

**Solution:** `create_model_adapter("mlx")` returns an `MLXModelAdapter` instance without the caller knowing the class name. The factory maps backend names to (class, default_model) pairs.

**Benefit:** Decouples the CLI from adapter implementations. New backends are registered in the factory's dict -- one line of code.

### 8.3 Strategy Pattern (Metrics)

**Problem:** Different specs use different evaluation metrics. The harness should not have a giant if/else chain.

**Solution:** `METRIC_REGISTRY` maps metric names to functions. All metric functions share the signature `(predicted, expected, tool_schema) -> bool`. The harness calls `get_metric(spec.metric.value)` and uses the returned function.

**Benefit:** Adding a new metric is three lines: write function, add to enum, register in dict. No harness changes needed.

### 8.4 Pipeline Pattern (Data Processing)

**Problem:** Data processing has sequential stages (download -> validate -> split -> augment -> format) where each stage depends on the previous.

**Solution:** Each stage is a separate module with a clear input/output contract. The CLI orchestrates the pipeline. Each stage can be run independently (useful for debugging and iteration).

**Benefit:** Stages can be rerun in isolation. If augmentation parameters change, only augment + format need to rerun, not download and validate.

### 8.5 Spec-Driven Development (Behavioral Testing)

**Problem:** "Is the model better?" is subjective and non-reproducible.

**Solution:** Define quantitative behavioral specs before training. Training is complete when all specs pass. The spec suite runs as a CI gate.

**Benefit:** Clear stopping condition, objective quality measurement, immediate regression detection.

### 8.6 Validate-at-Boundary Pattern

**Problem:** Invalid data (malformed YAML, bad schema, missing fields) causes cryptic errors deep in training/evaluation code.

**Solution:** Use Pydantic models at every data boundary: spec loading, example parsing, config loading. Invalid data is caught at load time with clear error messages. Downstream code can trust typed objects.

**Benefit:** Errors surface immediately with context ("spec name must be snake_case, got 'Tool Selection'") instead of buried tracebacks during training.

---

## 9. Key Engineering Decisions & Tradeoffs

### 9.1 MLX vs PyTorch

| Factor | MLX (chosen) | PyTorch |
|--------|-------------|---------|
| Apple Silicon performance | Native Metal GPU, ~40 tok/s | MPS backend, ~5 tok/s (spotty) |
| Training speed | 5.7 hours for SFT | 50+ hours on CPU |
| Memory model | Unified memory (zero-copy CPU<->GPU) | Separate CPU/GPU memory (no CUDA) |
| LoRA training support | Built-in via mlx-lm | Requires PEFT + bitsandbytes (CUDA only) |
| DPO support | Not built-in (had to implement) | Available via TRL library |
| Community/ecosystem | Smaller, Apple-focused | Massive, CUDA-focused |
| Production deployment | Mac-only | Any platform |

**Decision rationale:** The development hardware is M1 Pro 32GB. MLX is 10x faster on this hardware. The DPO implementation overhead (custom training loop from scratch) was worth it for the iteration speed gain -- multiple training experiments per day instead of one per day.

**Tradeoff acknowledged:** MLX limits production deployment to Apple Silicon. For server deployment, the LoRA adapters can be loaded with HuggingFace PEFT on any GPU platform. The adapters are the training output; the framework is the training tool.

### 9.2 4-Bit Quantization

The base model uses 4-bit quantization (`mlx-community/Llama-3.2-3B-Instruct-4bit`):

| Precision | Model Size | Memory | Quality |
|-----------|-----------|--------|---------|
| FP16 | 6.0 GB | >16 GB for training | Highest |
| 8-bit | 3.0 GB | ~10 GB for training | ~99% of FP16 |
| **4-bit** | **1.8 GB** | **~7.6 GB for training** | **~97% of FP16** |

**Why 4-bit:** Enables SFT training within 32GB unified memory. The 3% quality loss from quantization is recoverable through fine-tuning (which adjusts the LoRA adapters on top of quantized weights). The baseline evaluation uses the SAME quantized model we fine-tune, so our before/after comparison is apples-to-apples.

### 9.3 LoRA Rank 8 vs Higher

| Rank | Trainable Params | Adapter Size | Training Speed | Quality |
|------|-----------------|-------------|----------------|---------|
| 4 | 3.5M | 3.5 MB | Fastest | Too constrained for JSON patterns |
| **8** | **6.9M** | **7 MB** | **Fast** | **Good quality/speed tradeoff** |
| 16 | 13.8M | 14 MB | Moderate | Marginal improvement on 3B model |
| 32 | 27.6M | 28 MB | Slow | Diminishing returns |

**Decision:** Rank 8 was validated empirically. Initial experiments with rank 4 showed the model struggling to learn the exact JSON output format (closing braces, comma placement). Rank 8 resolved this. Rank 16 was tested but showed <1% improvement on tool_selection while taking 40% longer to train.

### 9.4 DPO Beta Tuning Journey

| Beta | LR | Grad Clip | Result |
|------|-----|-----------|--------|
| 0.1 | 5e-6 | None | Catastrophic forgetting. tool_selection: 0.96 -> 0.40 |
| 0.3 | 2e-6 | None | Still some forgetting with 139 pairs |
| **0.5** | **1e-6** | **1.0** | **Conservative updates, preserves SFT skills** |

**Insight:** With only 139 preference pairs, the model can quickly overfit to preference signals. Higher beta (stronger KL penalty) acts as regularization, keeping the model close to its SFT checkpoint. Combined with gradient clipping and early stopping, beta=0.5 allows gentle preference adjustments without destroying existing capabilities.

### 9.5 Data Augmentation vs More Real Data

| Approach | Pros | Cons |
|----------|------|------|
| Collect more real data | Highest quality, real distribution | Time-consuming, expensive, no error-handling datasets exist |
| **Synthetic augmentation** | **Fast, targeted at gaps, controllable** | **Distribution shift from real data, template diversity limits** |
| LLM-generated data | High quality synthetic | Expensive (API costs), slower iteration |

**Decision:** Synthetic augmentation was chosen for three reasons:
1. **No error-handling tool-calling dataset exists** in the open-source ecosystem. You cannot collect what does not exist.
2. **Speed of iteration** -- generating 800 examples takes seconds, not days.
3. **Precise targeting** -- we know exactly which behaviors are underrepresented and can generate examples specifically for those gaps.

**Mitigation for distribution shift:** All synthetic examples reuse tool definitions from real examples (realistic schemas) and are validated through the same Pydantic pipeline. Template diversity is maximized with parameterized fill-in values (8+ countries, 7+ concepts, 6+ activities, etc.).

### 9.6 Conservative vs Aggressive Training

The overall training philosophy is conservative:

- **Cosine decay with warmup** instead of constant LR
- **Early stopping** to prevent overtraining
- **Gradient clipping** to prevent catastrophic updates
- **mask_prompt=true** to focus loss on what matters
- **5 checkpoint saves** allowing rollback to the best checkpoint
- **Lower LR for continuation training** (3e-6 vs 1e-5) and DPO (1e-6)

**Why conservative:** With a 3B model and 2.7K-3.5K training examples, the risk of catastrophic forgetting is real. The model has limited capacity to absorb new behaviors without losing old ones. Every hyperparameter choice prioritizes stability over aggressive learning.

---

## 10. Lessons Learned & Debugging Stories

### 10.1 DPO Catastrophic Forgetting (v1 Failure -> v2 Fix)

**The disaster:** DPO v1 reduced all specs to near-zero. tool_selection dropped from 0.96 to 0.40. The model "unlearned" tool-calling in favor of preference patterns.

**Root cause analysis:**
1. Without gradient clipping, a single batch could produce gradient norms of 500+, which in one optimizer step destroyed the LoRA weights for attention patterns that the model had carefully learned during SFT.
2. Without training curve logging, there was no way to see this happening in real-time.
3. With beta=0.1, the KL penalty was too weak to prevent the model from diverging.

**The fix was a systematic application of standard RLHF stability techniques:**
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=50 steps)
- Higher beta (0.5 instead of 0.1)
- Label smoothing (0.1)
- Lower learning rate (1e-6 instead of 5e-6)
- Training curve logging

**Key takeaway:** DPO on a small model with few examples is fundamentally a different regime than DPO on a large model with thousands of pairs. The standard hyperparameters from the DPO paper (beta=0.1, lr=5e-6) assume ample data and model capacity. With 139 pairs and a 3B model, much more conservative settings are needed.

### 10.2 Parser Bug Causing 0% Multi-Tool Score

**The symptom:** Multi-tool sequencing scored 0% even after the model started producing what looked like correct multi-tool JSON arrays in raw output.

**The investigation:** Manual inspection of raw model output showed the model was generating `[{"name": "get_weather", ...}, {"name": "send_email", ...}]` -- a valid JSON array. But the eval harness scored it as 0%.

**Root cause:** The output parser in `parse_output()` was checking for single JSON objects (Strategy 3) BEFORE checking for JSON arrays (Strategy 2). When it encountered `[{"name"...`, it found the first `{` (the start of the first element), extracted just the first tool call as a single object, and silently dropped the rest of the array.

**The fix:** Reorder strategies: check for JSON arrays BEFORE single JSON objects. The array strategy uses brace-counting on `[...]` brackets, correctly handling the full array including nested objects.

**Key takeaway for interviews:** This is a classic "silent data loss" bug. The parser was not crashing or reporting errors -- it was silently returning partial results. The fix was 3 lines of code (reordering two strategy calls), but finding it required: (1) noticing the 0% score contradicted observed model output, (2) tracing through the parser with actual model output, (3) understanding that `{` inside `[...]` would be matched by the single-object extractor before the array extractor.

### 10.3 Data Distribution Imbalance Causing Training Signal Dilution

**The symptom:** SFT training with 91% tool-calling examples caused relevance_detection to REGRESS from 0.256 (baseline) to 0.186 (after SFT). The model became MORE eager to call tools, not less.

**Analysis:** With 2,757 training examples of which 2,510 (91%) involved calling a tool, the gradient signal overwhelmingly reinforced "when in doubt, call a tool." The 247 no_tool examples (9%) were drowned out by the tool-calling signal.

**Fix:** Data augmentation added 300 no_tool examples and 400 error_handling examples (which also teach the model to respond with text). Post-augmentation, no_tool+error_handling = 947/3,557 = 26.6% of training (up from 9%).

**Key takeaway:** In fine-tuning for behavioral skills, data DISTRIBUTION matters as much as data QUANTITY. Adding 800 targeted examples to a 2,757-example dataset had more impact than doubling the tool-calling data would have.

### 10.4 Metal GPU Memory Challenges on macOS

**Issue 1: Validation memory leak.** After the validation pass (10 batches), Metal did not release allocated memory before the training forward+backward pass. This caused OOM on 32GB unified memory.

**Fix:** `grad_checkpoint=true` trades ~20% compute for 40-60% less activation memory. Combined with `val_batches=10` (down from 25) to limit validation memory footprint. The `clear_cache_threshold=0.5` flag tells MLX to free Metal memory when >50% of cache is unused.

**Issue 2: DPO step time degradation.** DPO training steps went from ~16s to 60s+ over 300 iterations because Metal cached intermediate allocations.

**Fix:** Call `mx.metal.clear_cache()` every 10 steps to force Metal to release unused allocations.

**Issue 3: Output buffering.** Python's stdout buffering on macOS prevented real-time display of training progress when running via subprocess.

**Key takeaway:** Apple Silicon's unified memory architecture is a double-edged sword. The zero-copy CPU<->GPU transfer is fantastic for loading models. But Metal's GPU memory management is less granular than CUDA's, requiring explicit cache management that CUDA handles automatically.

### 10.5 The Truncation Bug in DPO

**Symptom:** DPO loss stuck at exactly 0.6931 (ln(2)) with gradient norm = 0.000000.

**Investigation path:**
1. Loss of ln(2) means sigmoid input is exactly 0, which means reward_margin is exactly 0.
2. Reward_margin = 0 means chosen_logps == rejected_logps.
3. If chosen == rejected for every pair, there is literally nothing to learn.
4. Inspected tokenized sequences: `chosen_ids == rejected_ids` for all 139 pairs!
5. Root cause: Right-truncation at max_length=512 cut off the response (where chosen differs from rejected). The prompt's tool schema consumed >500 tokens.

**Fix:** Left-truncate the prompt, preserving the response in full. Keep at least 64 prompt tokens for context.

---

## 11. Interview-Ready Talking Points

### What ToolForge Demonstrates

1. **End-to-end ML engineering** -- Not just training a model, but building the entire pipeline: data ingestion, validation, augmentation, training (SFT + DPO), evaluation, and serving.

2. **Spec-driven development** -- Defining success criteria before training, measuring baseline, and verifying improvements. This is how production ML should work.

3. **Research paper implementation** -- Custom DPO training loop implemented from the paper when the framework (mlx-lm) did not support it. Shows ability to go from paper to working code.

4. **Systematic debugging** -- The zero-gradient bug story shows hypothesis-driven debugging, not random experimentation. The catastrophic forgetting analysis shows root-cause thinking.

5. **Data-centric AI** -- Data augmentation for underrepresented behaviors had more impact than hyperparameter tuning. Distribution matters more than quantity.

6. **Production engineering** -- Pydantic validation at boundaries, factory patterns, adapter patterns, OpenAI-compatible API, 373 unit tests.

### Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Raw dataset examples | 8,995 |
| After validation | 3,240 (64% removed) |
| Augmented training set | 3,557 |
| Base model size | 3.2B params |
| LoRA trainable params | 6.9M (1.4%) |
| Adapter size on disk | 7 MB |
| Base model memory (4-bit) | 1.8 GB |
| Peak training memory | 7.6 GB |
| SFT training time | 5.7 hours (M1 Pro 32GB) |
| DPO preference pairs | 139 (from model's own failures) |
| DPO training time | ~10 minutes |
| Unit tests | 373 |
| Behavioral specs | 6 |
| Baseline specs passing | 1/6 |
| After SFT specs passing | 2/6 |

### What I Would Do Differently

1. **Start with augmented data from day one.** The data distribution imbalance was predictable from the split statistics. Including augmented data in the initial SFT would have saved an entire training stage.

2. **Implement training curve logging from the start.** The DPO v1 catastrophic forgetting would have been caught in real-time with proper logging instead of discovered after a full training run.

3. **Use left-truncation by default for DPO.** This is a known best practice in the RLHF community but is not always documented. Having it as the default would have avoided the zero-gradient debugging session.

4. **Test with a smaller model first.** Running quick experiments on a 1B model before committing to multi-hour 3B training runs would accelerate hyperparameter search.

5. **Add LLM-as-judge for error_recovery.** The current heuristic evaluation (keyword matching) misses nuanced error handling. An LLM-as-judge evaluation would be more accurate, though less reproducible.

### How This Scales

**Scaling to more tools:**
- Current: 50-tool schema per example
- The approach generalizes because LoRA adapts attention patterns, not tool-specific knowledge
- For 500+ tools: consider retrieval-augmented tool selection (retrieve top-k relevant tools, then route)

**Scaling to larger models:**
- Same pipeline works for 7B, 13B, 70B with adjusted LoRA rank and hardware
- 7B on M1 Pro 32GB: LoRA rank 4, batch_size 1, expect 2x training time
- 70B: requires cloud GPU (A100 or H100) with QLoRA + DeepSpeed

**Scaling to production traffic:**
- Current FastAPI server is single-request synchronous
- Scale with: batched inference (queue requests, batch forward passes), model replicas behind load balancer, vLLM for continuous batching
- MLX adapters can be converted to PEFT format for deployment on any GPU platform

**Scaling the spec suite:**
- Add specs as new failure modes are discovered in production
- Specs can gate deployments in CI/CD -- model update only deploys if all specs pass
- Historical spec reports enable tracking quality over time across model versions

---

## Appendix: Project File Reference

```
toolforge/
    configs/
        specs/                     # 6 behavioral spec YAML files
            tool_selection.yaml
            argument_accuracy.yaml
            hallucination_resistance.yaml
            relevance_detection.yaml
            multi_tool_sequencing.yaml
            error_recovery.yaml
        training/
            sft.yaml               # SFT hyperparameters (Stage 4)
            sft_continue.yaml      # SFT continuation (Stage 4b)
            dpo.yaml               # DPO hyperparameters (Stage 5, v2)
        data/
            default.yaml           # Data pipeline parameters
    src/toolforge/
        cli.py                     # Typer CLI entry point
        data/
            schema.py              # ToolCallingExample, Pydantic v2
            download.py            # HuggingFace Hub ingestion
            validate.py            # 5 quality checks + dedup
            prepare.py             # Stratified splitting + eval datasets
            augment.py             # Synthetic data for underrepresented types
            formatter.py           # Llama 3.2 chat template (manual)
            mlx_format.py          # OpenAI chat format for MLX
        eval/
            specs.py               # BehavioralSpec loader, Pydantic
            metrics.py             # 6 metrics + METRIC_REGISTRY
            models.py              # BaseModelAdapter + MLX/Ollama/Dummy
            harness.py             # Spec runner, report generator
        training/
            sft.py                 # SFT pipeline + SFT continuation
            dpo.py                 # Custom DPO implementation
            preference.py          # Preference pair generation
        serving/
            api.py                 # FastAPI /v1/tool-call endpoint
            demo.py                # Gradio interactive demo
    tests/
        unit/
            test_schema.py
            test_download.py
            test_validate.py
            test_prepare.py
            test_augment.py
            test_formatter.py
            test_mlx_format.py
            test_metrics.py
            test_specs.py
            test_models.py
            test_sft.py
            test_dpo.py
            test_serving.py
    pyproject.toml                 # Dependencies, optional groups, CLI entry
    Makefile                       # Common operations
    Containerfile                  # Multi-stage Podman build
```
