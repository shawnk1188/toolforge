"""
DPO (Direct Preference Optimization) training for MLX.

WHY DPO OVER MORE SFT:
  SFT teaches "do this" — the model imitates correct examples.
  DPO teaches "do this, NOT that" — the model learns from contrasts.

  After SFT, our model:
    - Gets tool calls RIGHT most of the time (format learned)
    - Still calls tools when it SHOULDN'T (relevance_detection: 0.186)
    - Can't handle errors gracefully (error_recovery: 0.060)
    - Can't produce multi-tool sequences (multi_tool_sequencing: 0.000)

  These are BEHAVIORAL issues. The model needs to see examples of what
  NOT to do (its own mistakes) paired with what TO do (ground truth).
  That's exactly what DPO provides.

THE DPO LOSS:
  L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

  Where:
    x    = prompt
    y_w  = chosen (correct) response
    y_l  = rejected (model's wrong) response
    π_θ  = policy model (being trained)
    π_ref = reference model (frozen SFT checkpoint)
    β    = temperature controlling how much to diverge from reference

  Intuition: increase probability of chosen response relative to rejected,
  while not straying too far from the reference model.

IMPLEMENTATION:
  Since mlx-lm doesn't have DPO built in, we implement the training loop
  from scratch using MLX's autograd. The key pieces:
    1. compute_log_probs() — forward pass to get per-token log probabilities
    2. dpo_loss() — the DPO objective combining policy and reference log probs
    3. train_step() — gradient computation and optimizer update
    4. training loop — iterate over preference pairs with validation
"""

from __future__ import annotations

import json
import math
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from rich.console import Console

console = Console()


# ============================================================
# DPO Configuration
# ============================================================


@dataclass
class DPOConfig:
    """
    DPO training configuration.

    WHY THESE DEFAULTS:
      beta=0.1: Standard DPO temperature. Lower = more aggressive preference
        learning. Higher = more conservative (stay closer to reference).
        0.1 is the sweet spot from the original DPO paper.

      learning_rate=5e-6: Half of SFT LR. DPO is a fine-tuning-on-fine-tuning
        step — too high and you forget SFT skills, too low and preferences
        don't take effect. 5e-6 is standard for DPO on top of SFT.

      iters=500: DPO converges faster than SFT because:
        1. The model already knows the format (from SFT)
        2. Preference pairs are more information-dense than SFT examples
        3. We have fewer examples (only model failures)
    """

    # Model
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    sft_adapter_path: str = "artifacts/sft/adapters"
    adapter_path: str = "artifacts/dpo/adapters"

    # DPO hyperparameters
    beta: float = 0.1
    label_smoothing: float = 0.0  # 0 = standard DPO, >0 = conservative DPO

    # Training
    iters: int = 500
    batch_size: int = 1  # DPO pairs are longer (chosen+rejected), so smaller batch
    learning_rate: float = 5e-6
    max_seq_length: int = 2048
    grad_checkpoint: bool = True
    seed: int = 42

    # LoRA
    lora_rank: int = 8
    lora_dropout: float = 0.05
    lora_scale: float = 20.0
    num_layers: int = 16

    # Stability
    grad_clip_norm: float = 1.0  # Max gradient norm (0 = no clipping)
    early_stopping_patience: int = 50  # Stop if margin doesn't improve for N steps (0 = disabled)

    # Logging
    steps_per_report: int = 10
    steps_per_eval: int = 100
    save_every: int = 100

    # Data
    data_path: str = "data/dpo/train.jsonl"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DPOConfig":
        """Load config from YAML file, filling defaults for missing fields."""
        import yaml

        path = Path(path)
        if not path.exists():
            console.print(f"[yellow]Config not found at {path}, using defaults[/yellow]")
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Map YAML fields to dataclass fields
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if hasattr(cls, k):
                kwargs[k] = v

        # Handle nested lora_parameters
        lora_params = raw.get("lora_parameters", {})
        if "rank" in lora_params:
            kwargs["lora_rank"] = lora_params["rank"]
        if "dropout" in lora_params:
            kwargs["lora_dropout"] = lora_params["dropout"]
        if "scale" in lora_params:
            kwargs["lora_scale"] = lora_params["scale"]

        return cls(**kwargs)


# ============================================================
# Log Probability Computation
# ============================================================


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: mx.array,
    target_ids: mx.array,
    lengths: mx.array,
) -> mx.array:
    """
    Compute the sum of log probabilities for target tokens.

    WHY SUM NOT MEAN:
      DPO uses the total log probability of the sequence, not the average.
      This is because DPO's implicit reward is based on how much MORE likely
      the policy makes the chosen vs rejected — and that's a ratio of
      total probabilities.

    Args:
        model: The language model
        input_ids: Input token IDs [batch, seq_len]
        target_ids: Target token IDs to compute log probs for [batch, seq_len]
        lengths: Length of each sequence (for masking padding)

    Returns:
        Sum of log probs for each example in the batch [batch]
    """
    # Forward pass — get logits
    logits = model(input_ids)  # [batch, seq_len, vocab_size]

    # Shift: predict next token from current position
    # logits[:, :-1] predicts target[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_targets = target_ids[:, 1:]

    # Log softmax to get log probabilities
    log_probs = nn.log_softmax(shift_logits, axis=-1)

    # Gather log probs for the actual target tokens
    # shape: [batch, seq_len-1]
    target_log_probs = mx.take_along_axis(
        log_probs,
        shift_targets[:, :, None],
        axis=-1,
    ).squeeze(-1)

    # Create mask for valid (non-padding) positions
    # lengths tells us how many real tokens each sequence has
    seq_len = shift_targets.shape[1]
    positions = mx.arange(seq_len)[None, :]  # [1, seq_len]
    mask = positions < (lengths[:, None] - 1)  # -1 because we shifted

    # Sum log probs over valid positions (masked)
    masked_log_probs = target_log_probs * mask
    return mx.sum(masked_log_probs, axis=-1)  # [batch]


# ============================================================
# DPO Loss
# ============================================================


def dpo_loss(
    policy_chosen_logps: mx.array,
    policy_rejected_logps: mx.array,
    ref_chosen_logps: mx.array,
    ref_rejected_logps: mx.array,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[mx.array, dict[str, float]]:
    """
    Compute the DPO loss.

    THE MATH:
      L = -log σ(β · (log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))))
        = -log σ(β · (chosen_rewards - rejected_rewards))

      where rewards = log π_θ(y|x) - log π_ref(y|x)

    With label smoothing (conservative DPO):
      L = -((1-ε)·log σ(β·Δ) + ε·log σ(-β·Δ))

    Args:
        policy_chosen_logps: π_θ(y_w|x) log probs
        policy_rejected_logps: π_θ(y_l|x) log probs
        ref_chosen_logps: π_ref(y_w|x) log probs
        ref_rejected_logps: π_ref(y_l|x) log probs
        beta: Temperature parameter
        label_smoothing: Conservative DPO parameter (0 = standard)

    Returns:
        (loss, metrics_dict) where metrics has rewards and margins
    """
    # Compute implicit rewards
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # Reward margin (positive = model prefers chosen, which is what we want)
    reward_margin = chosen_rewards - rejected_rewards

    # DPO loss
    if label_smoothing > 0:
        # Conservative DPO: smooth the labels
        loss = -(
            (1 - label_smoothing) * nn.log_sigmoid(reward_margin)
            + label_smoothing * nn.log_sigmoid(-reward_margin)
        )
    else:
        # Standard DPO
        loss = -nn.log_sigmoid(reward_margin)

    loss = mx.mean(loss)

    # Metrics for logging
    metrics = {
        "loss": loss.item(),
        "chosen_reward": mx.mean(chosen_rewards).item(),
        "rejected_reward": mx.mean(rejected_rewards).item(),
        "reward_margin": mx.mean(reward_margin).item(),
        "accuracy": mx.mean(reward_margin > 0).item(),
    }

    return loss, metrics


# ============================================================
# Dataset
# ============================================================


class DPODataset:
    """
    Dataset for DPO preference pairs.

    WHY CUSTOM DATASET:
      MLX's built-in ChatDataset doesn't support preference pairs.
      DPO needs (prompt, chosen, rejected) triples, not single messages.
      We handle tokenization of both chosen and rejected sequences.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = self._load(path)
        console.print(f"  Loaded {len(self.pairs)} preference pairs from {path}")

    def _load(self, path: str | Path) -> list[dict]:
        """Load preference pairs from JSONL."""
        pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """
        Tokenize a preference pair.

        Returns dict with:
          - chosen_ids: token IDs for prompt + chosen response
          - rejected_ids: token IDs for prompt + rejected response
          - chosen_length: number of non-padding tokens in chosen
          - rejected_length: number of non-padding tokens in rejected

        TRUNCATION STRATEGY:
          The response (chosen/rejected) is what differs between pairs —
          it MUST be preserved in full. If the total exceeds max_length,
          we truncate the PROMPT from the left (keeping the end, which
          has the user query — the most important part).

          Previous bug: right-truncation of the full sequence cut off
          the response, making chosen == rejected for all pairs.
        """
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        # Tokenize prompt and responses SEPARATELY
        prompt_tokens = self.tokenizer.encode(prompt)
        chosen_resp_tokens = self.tokenizer.encode(chosen)
        rejected_resp_tokens = self.tokenizer.encode(rejected)

        # Truncate prompt from the LEFT if needed to fit the response
        # Keep at least 64 prompt tokens for context
        min_prompt_tokens = 64
        max_resp_len = max(len(chosen_resp_tokens), len(rejected_resp_tokens))
        max_prompt_len = max(self.max_length - max_resp_len, min_prompt_tokens)

        if len(prompt_tokens) > max_prompt_len:
            # Truncate from the left — keep the end (user query is there)
            prompt_tokens = prompt_tokens[-max_prompt_len:]

        # Build full sequences
        chosen_tokens = prompt_tokens + chosen_resp_tokens
        rejected_tokens = prompt_tokens + rejected_resp_tokens

        # Final safety truncation (in case response itself exceeds max_length)
        chosen_tokens = chosen_tokens[:self.max_length]
        rejected_tokens = rejected_tokens[:self.max_length]

        return {
            "chosen_ids": mx.array(chosen_tokens),
            "rejected_ids": mx.array(rejected_tokens),
            "chosen_length": len(chosen_tokens),
            "rejected_length": len(rejected_tokens),
        }


def collate_dpo_batch(
    batch: list[dict[str, Any]],
) -> dict[str, mx.array]:
    """
    Collate a batch of preference pairs with padding.

    WHY PADDING:
      Chosen and rejected sequences have different lengths. We pad both
      to the max length in the batch so we can batch them through the model.
    """
    max_chosen = max(item["chosen_length"] for item in batch)
    max_rejected = max(item["rejected_length"] for item in batch)

    # Pad to max length in batch
    chosen_ids = []
    rejected_ids = []
    chosen_lengths = []
    rejected_lengths = []

    pad_id = 0  # Most tokenizers use 0 as pad

    for item in batch:
        c = item["chosen_ids"].tolist()
        r = item["rejected_ids"].tolist()

        chosen_ids.append(c + [pad_id] * (max_chosen - len(c)))
        rejected_ids.append(r + [pad_id] * (max_rejected - len(r)))
        chosen_lengths.append(item["chosen_length"])
        rejected_lengths.append(item["rejected_length"])

    return {
        "chosen_ids": mx.array(chosen_ids),
        "rejected_ids": mx.array(rejected_ids),
        "chosen_lengths": mx.array(chosen_lengths),
        "rejected_lengths": mx.array(rejected_lengths),
    }


# ============================================================
# Training Loop
# ============================================================


def run_dpo_training(
    config: DPOConfig | None = None,
    config_path: str | Path | None = None,
    iters: int | None = None,
    skip_pair_gen: bool = False,
) -> Path:
    """
    Run the full DPO training pipeline.

    Steps:
      1. (Optional) Generate preference pairs from SFT failures
      2. Load reference model (frozen SFT) + policy model (trainable SFT copy)
      3. DPO training loop
      4. Save adapters

    Args:
        config: DPO configuration (or loaded from config_path)
        config_path: Path to YAML config file
        iters: Override number of iterations
        skip_pair_gen: Skip preference pair generation (use existing data)

    Returns:
        Path to the saved adapter directory
    """
    from rich.panel import Panel

    # Load config
    if config is None:
        if config_path:
            config = DPOConfig.from_yaml(config_path)
        else:
            config = DPOConfig()

    if iters is not None:
        config.iters = iters

    console.print(Panel(
        "[bold]ToolForge DPO Training Pipeline[/bold]\n"
        f"Config: {config_path or 'defaults'}",
        title="Stage 5",
        border_style="magenta",
    ))

    # Step 1: Generate preference pairs (optional)
    if not skip_pair_gen:
        console.print("\n[bold]Step 1/3: Generating preference pairs[/bold]")
        from toolforge.training.preference import generate_preference_pairs
        stats = generate_preference_pairs(
            adapter_path=config.sft_adapter_path,
            output_path=config.data_path,
            model_id=config.model,
        )
        console.print(f"  Generated {stats['total_pairs']} pairs")
    else:
        console.print("\n[bold]Step 1/3:[/bold] Skipping pair generation (--skip-pair-gen)")

    # Verify data exists
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"DPO training data not found at {data_path}. "
            "Run without --skip-pair-gen to generate it."
        )

    # Step 2: Load models
    console.print("\n[bold]Step 2/3: Loading models[/bold]")

    # Patch SSL for proxy environments
    try:
        import httpx
        _orig_init = httpx.Client.__init__

        def _ssl_patched_init(self_client, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _orig_init(self_client, *args, **kwargs)

        httpx.Client.__init__ = _ssl_patched_init
    except ImportError:
        pass

    from mlx_lm import load as mlx_load
    # Reference model: base + SFT adapters (frozen, computes reference log probs)
    # Policy model: base + fresh LoRA (initialized from SFT, gets trained)
    #
    # WHY THIS SPLIT:
    #   DPO needs TWO models: a frozen reference (π_ref) and a trainable policy (π_θ).
    #   Both START as the SFT model, but only the policy gets updated.
    #   The reference provides a baseline for computing preference rewards.

    console.print(f"  Loading base model: {config.model}")
    console.print(f"  Loading SFT adapters: {config.sft_adapter_path}")

    # MEMORY-EFFICIENT DPO:
    # Loading 2 full models simultaneously would use ~8GB and leave no room for
    # forward passes on Apple Silicon. Instead:
    # 1. Load reference model → pre-compute ALL reference log probs → unload
    # 2. Load policy model → train using cached reference log probs
    #
    # This cuts peak memory from ~12GB to ~6GB.

    console.print(f"  Loading base model: {config.model}")
    console.print(f"  Loading SFT adapters: {config.sft_adapter_path}")

    # Load dataset first (need tokenizer)
    ref_model, tokenizer = mlx_load(
        config.model,
        adapter_path=config.sft_adapter_path,
    )
    ref_model.freeze()

    dataset = DPODataset(
        path=config.data_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )

    if len(dataset) == 0:
        raise ValueError("No preference pairs found. Cannot train DPO.")

    # Pre-compute ALL reference log probs
    console.print(f"\n  [cyan]Pre-computing reference log probs for {len(dataset)} pairs...[/cyan]")
    ref_chosen_logps = []
    ref_rejected_logps = []

    for i in range(len(dataset)):
        item = dataset[i]
        batch = collate_dpo_batch([item])

        chosen_lp = compute_sequence_log_probs(
            ref_model,
            batch["chosen_ids"],
            batch["chosen_ids"],
            batch["chosen_lengths"],
        )
        rejected_lp = compute_sequence_log_probs(
            ref_model,
            batch["rejected_ids"],
            batch["rejected_ids"],
            batch["rejected_lengths"],
        )
        mx.eval(chosen_lp, rejected_lp)
        ref_chosen_logps.append(chosen_lp.item())
        ref_rejected_logps.append(rejected_lp.item())

        if (i + 1) % 25 == 0:
            console.print(f"    [{i+1}/{len(dataset)}]")

    console.print(f"  [green]Reference log probs cached for {len(dataset)} pairs[/green]")

    # Unload reference model to free memory
    del ref_model
    import gc
    gc.collect()

    # Load policy model
    console.print(f"  Loading policy model...")
    policy_model, _ = mlx_load(
        config.model,
        adapter_path=config.sft_adapter_path,
    )

    # Freeze base weights, unfreeze LoRA only
    policy_model.freeze()
    for name, module in policy_model.named_modules():
        if "LoRA" in type(module).__name__:
            module.unfreeze()
            if hasattr(module, "linear"):
                module.linear.freeze()

    # Count trainable parameters
    def _count_params(params):
        total = 0
        if isinstance(params, mx.array):
            return params.size
        elif isinstance(params, dict):
            for v in params.values():
                total += _count_params(v)
        elif isinstance(params, list):
            for v in params:
                total += _count_params(v)
        return total

    total_params = _count_params(policy_model.parameters())
    trainable = _count_params(policy_model.trainable_parameters())
    console.print(f"  Trainable parameters: {trainable:,} / {total_params:,}")

    # Step 3: DPO Training
    console.print(f"\n[bold]Step 3/3: DPO Training[/bold]")
    console.print(f"  Iterations:      {config.iters}")
    console.print(f"  Batch size:      {config.batch_size}")
    console.print(f"  Learning rate:   {config.learning_rate}")
    console.print(f"  Beta:            {config.beta}")
    console.print(f"  Label smoothing: {config.label_smoothing}")
    console.print(f"  Grad clip norm:  {config.grad_clip_norm}")
    console.print(f"  Pairs:           {len(dataset)}")

    # Setup optimizer
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    mx.random.seed(config.seed)
    np.random.seed(config.seed)

    adapter_dir = Path(config.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Training curve log — save metrics for every report step
    # WHY: Previous DPO runs had no loss curves, making it impossible
    # to diagnose whether the model was converging or diverging.
    training_log: list[dict] = []

    start_time = time.time()
    indices = list(range(len(dataset)))

    # Track best reward margin for early stopping
    best_margin = float("-inf")
    steps_without_improvement = 0
    patience = config.early_stopping_patience

    console.print("\n  [cyan]Starting DPO training...[/cyan]")

    for step in range(1, config.iters + 1):
        # Sample a batch
        batch_idx = np.random.choice(indices, size=config.batch_size, replace=True)
        batch_items = [dataset[i] for i in batch_idx]
        batch = collate_dpo_batch(batch_items)

        # Get cached reference log probs for this batch
        batch_ref_chosen = mx.array([ref_chosen_logps[i] for i in batch_idx])
        batch_ref_rejected = mx.array([ref_rejected_logps[i] for i in batch_idx])

        # Compute loss and gradients (policy model only — 2 forward passes, not 4)
        is_report_step = (step % config.steps_per_report == 0)
        loss_val, grads, metrics = _train_step_cached(
            policy_model=policy_model,
            batch=batch,
            ref_chosen_logps=batch_ref_chosen,
            ref_rejected_logps=batch_ref_rejected,
            beta=config.beta,
            label_smoothing=config.label_smoothing,
            compute_metrics=is_report_step,
        )

        # Gradient clipping — prevents explosive updates that cause catastrophic forgetting
        # WHY: Without clipping, a single batch with large loss can produce huge gradients
        # that destroy the model's existing capabilities in one step.
        if config.grad_clip_norm > 0:
            grads = _clip_grad_norm(grads, config.grad_clip_norm)

        # Update policy model
        optimizer.update(policy_model, grads)
        mx.eval(policy_model.parameters(), optimizer.state)

        # Clear Metal GPU cache periodically to prevent memory leak
        # WHY: Metal doesn't release cached allocations automatically.
        # Without this, step time degrades from ~16s to 60s+ over 300 iterations.
        if step % 10 == 0:
            if hasattr(mx, 'clear_cache'):
                mx.clear_cache()
            elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()

        # Report
        if step % config.steps_per_report == 0:
            elapsed = time.time() - start_time

            # Compute gradient norm for debugging
            grad_norm = _compute_grad_norm(grads)

            margin = metrics.get('reward_margin', 0.0)
            acc = metrics.get('accuracy', 0.0)
            console.print(
                f"  Iter {step:4d}: loss={metrics['loss']:.4f}  "
                f"margin={margin:.4f}  "
                f"acc={acc:.3f}  "
                f"grad_norm={grad_norm:.6f}  "
                f"[{elapsed/60:.1f}min]"
            )

            # Log training curve
            step_log = {
                "step": step,
                "loss": metrics["loss"],
                "reward_margin": margin,
                "accuracy": acc,
                "grad_norm": grad_norm,
                "elapsed_min": elapsed / 60,
            }
            if "chosen_reward" in metrics:
                step_log["chosen_reward"] = metrics["chosen_reward"]
                step_log["rejected_reward"] = metrics["rejected_reward"]
            training_log.append(step_log)

            # Early stopping: if reward margin is declining steadily, stop
            # WHY: A declining margin means the model is unlearning its
            # preferences — continuing will only make it worse.
            if margin > best_margin:
                best_margin = margin
                steps_without_improvement = 0
            else:
                steps_without_improvement += config.steps_per_report

            if patience > 0 and steps_without_improvement >= patience:
                console.print(
                    f"\n  [yellow]Early stopping at iter {step}: "
                    f"no margin improvement for {patience} steps[/yellow]"
                )
                break

        # Save checkpoint
        if step % config.save_every == 0:
            _save_adapters(policy_model, adapter_dir, step, config)
            console.print(f"  [green]Saved checkpoint at iter {step}[/green]")

    # Save final adapters
    _save_adapters(policy_model, adapter_dir, step, config)

    elapsed = time.time() - start_time
    console.print(f"\n  [bold green]DPO training complete in {elapsed/60:.1f} minutes[/bold green]")

    # Save training curve log
    log_path = adapter_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    console.print(f"  Training log saved to {log_path}")

    # Save config
    _save_config(config, adapter_dir)

    console.print(Panel(
        f"[bold]DPO training complete![/bold]\n"
        f"Adapters saved to: {adapter_dir}\n"
        f"Training log: {log_path}\n"
        f"Best reward margin: {best_margin:.4f}\n\n"
        f"Next steps:\n"
        f"  1. Evaluate: toolforge eval run --backend mlx --adapter-path {adapter_dir}\n"
        f"  2. Compare: baseline → SFT → SFT+DPO",
        title="Done",
        border_style="green",
    ))

    return adapter_dir


def _train_step_cached(
    policy_model: nn.Module,
    batch: dict[str, mx.array],
    ref_chosen_logps: mx.array,
    ref_rejected_logps: mx.array,
    beta: float,
    label_smoothing: float,
    compute_metrics: bool = False,
) -> tuple[mx.array, Any, dict]:
    """
    DPO training step with PRE-COMPUTED reference log probs.

    WHY CACHED:
      Loading 2 full 3B models simultaneously would use ~8GB on Apple Silicon
      and leave insufficient memory for forward passes. Instead, we pre-compute
      all reference log probs once, unload the reference model, then train
      with only the policy model in memory.

      This reduces per-step computation from 4 forward passes to 2.

    PERFORMANCE:
      Only compute metrics when compute_metrics=True (at reporting steps).
      Previously, 2 extra forward passes ran EVERY step for metrics, causing
      a Metal GPU memory leak that slowed steps from 16s → 67s+.
    """
    chosen_ids = batch["chosen_ids"]
    rejected_ids = batch["rejected_ids"]
    chosen_lengths = batch["chosen_lengths"]
    rejected_lengths = batch["rejected_lengths"]

    def loss_fn(model):
        # Policy log probs (these get gradients via autograd)
        policy_chosen = compute_sequence_log_probs(
            model, chosen_ids, chosen_ids, chosen_lengths
        )
        policy_rejected = compute_sequence_log_probs(
            model, rejected_ids, rejected_ids, rejected_lengths
        )

        loss, _ = dpo_loss(
            policy_chosen, policy_rejected,
            ref_chosen_logps, ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )
        return loss

    # nn.value_and_grad(model, fn) returns a callable that computes (loss, grads)
    loss_and_grad_fn = nn.value_and_grad(policy_model, loss_fn)
    loss, grads = loss_and_grad_fn(policy_model)
    mx.eval(loss, grads)

    # Only compute metrics at reporting steps (saves 2 forward passes per step)
    metrics = {"loss": loss.item()}
    if compute_metrics:
        policy_chosen = compute_sequence_log_probs(
            policy_model, chosen_ids, chosen_ids, chosen_lengths
        )
        policy_rejected = compute_sequence_log_probs(
            policy_model, rejected_ids, rejected_ids, rejected_lengths
        )
        _, metrics = dpo_loss(
            policy_chosen, policy_rejected,
            ref_chosen_logps, ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )
        mx.eval(policy_chosen, policy_rejected)

    return loss, grads, metrics


def _compute_grad_norm(grads: Any) -> float:
    """Compute the L2 norm of a nested gradient tree."""
    total = 0.0

    def _accumulate(tree):
        nonlocal total
        if isinstance(tree, mx.array):
            total += mx.sum(tree * tree).item()
        elif isinstance(tree, dict):
            for v in tree.values():
                _accumulate(v)
        elif isinstance(tree, list):
            for v in tree:
                _accumulate(v)

    _accumulate(grads)
    return math.sqrt(total)


def _clip_grad_norm(grads: Any, max_norm: float) -> Any:
    """
    Clip gradient tree by global L2 norm.

    WHY GRADIENT CLIPPING:
      DPO with few examples (139 pairs) can produce wildly varying gradients
      between steps. A single batch with a large loss delta can produce a
      gradient that destroys the model's existing capabilities. Clipping
      bounds the step size, preventing catastrophic single-step forgetting.

      This is standard practice in RLHF/DPO training and was notably absent
      from the initial implementation, contributing to the 0/6 regression.

    Args:
        grads: Nested dict/list of mx.arrays (gradient tree)
        max_norm: Maximum allowed L2 norm

    Returns:
        Clipped gradient tree (same structure)
    """
    norm = _compute_grad_norm(grads)
    if norm <= max_norm or norm == 0:
        return grads

    scale = max_norm / norm

    def _scale_tree(tree):
        if isinstance(tree, mx.array):
            return tree * scale
        elif isinstance(tree, dict):
            return {k: _scale_tree(v) for k, v in tree.items()}
        elif isinstance(tree, list):
            return [_scale_tree(v) for v in tree]
        return tree

    return _scale_tree(grads)


def _save_adapters(
    model: nn.Module,
    adapter_dir: Path,
    step: int,
    config: DPOConfig | None = None,
) -> None:
    """
    Save LoRA adapter weights in safetensors format.

    WHY FLATTEN + SAFETENSORS:
      mlx's trainable_parameters() returns nested dicts/lists.
      mx.savez can't handle nested structures. We flatten the tree
      to dot-separated keys and save with mx.save_safetensors().

    ADAPTER CONFIG:
      Must be compatible with mlx-lm's `load()` function, which expects
      fields like `lora_parameters`, `num_layers`, and `model` to
      properly reconstruct the LoRA layers when loading.
    """
    # Flatten nested parameter tree to flat dict
    flat_weights = {}

    def _flatten(tree, prefix=""):
        if isinstance(tree, mx.array):
            flat_weights[prefix] = tree
        elif isinstance(tree, dict):
            for k, v in tree.items():
                _flatten(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(tree, list):
            for i, v in enumerate(tree):
                _flatten(v, f"{prefix}.{i}" if prefix else str(i))

    _flatten(model.trainable_parameters())

    # Save as safetensors (compatible with mlx-lm load_adapters)
    mx.save_safetensors(str(adapter_dir / f"{step:06d}_adapters.safetensors"), flat_weights)
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), flat_weights)

    # Save mlx-lm compatible adapter config
    # This format is required by mlx_lm.load() to properly apply LoRA layers
    lora_params = {"rank": 8, "dropout": 0.05, "scale": 20.0}
    num_layers = 16
    model_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    if config is not None:
        # DPOConfig stores LoRA params as separate fields, not a nested dict
        lora_params = {
            "rank": getattr(config, "lora_rank", 8),
            "dropout": getattr(config, "lora_dropout", 0.05),
            "scale": getattr(config, "lora_scale", 20.0),
        }
        num_layers = getattr(config, "num_layers", 16)
        model_id = getattr(config, "model", model_id)

    adapter_config = {
        "model": model_id,
        "adapter_path": str(adapter_dir),
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_params,
        "step": step,
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)


def _save_config(config: DPOConfig, adapter_dir: Path) -> None:
    """Save training config for reproducibility."""
    import yaml
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open(adapter_dir / "toolforge_dpo_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
