"""
Supervised Fine-Tuning (SFT) with MLX LoRA.

WHY THIS MODULE EXISTS:
  Wraps mlx-lm's LoRA training with ToolForge-specific orchestration:
    1. Data preparation (convert canonical format → MLX chat format)
    2. Config loading from YAML
    3. Training with progress reporting
    4. Post-training evaluation against behavioral specs

  You CAN run mlx_lm.lora directly with our YAML config, but this module
  adds the data conversion step and spec evaluation that make the
  workflow reproducible.

DESIGN:
  The training itself is delegated to mlx-lm's battle-tested LoRA trainer.
  We don't reimplement the training loop — that would be fragile and
  pointless. Instead, we orchestrate the pipeline around it.

HARDWARE:
  Designed for Apple Silicon (M1/M2/M3) with 16-32GB unified memory.
  MLX uses Metal for GPU acceleration and unified memory for zero-copy
  data transfer between CPU and GPU.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()

# Default config path
DEFAULT_CONFIG = "configs/training/sft.yaml"


@dataclass
class SFTConfig:
    """
    SFT training configuration.

    Loaded from YAML, with sensible defaults for each field.
    CLI arguments can override individual fields.
    """

    # Model
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    # Data
    data_dir: str = "data/mlx"
    processed_dir: str = "data/processed"

    # LoRA
    lora_rank: int = 8
    lora_dropout: float = 0.05
    lora_scale: float = 20.0

    # Training
    iters: int = 1000
    batch_size: int = 2
    learning_rate: float = 1e-5
    max_seq_length: int = 2048
    grad_checkpoint: bool = True
    mask_prompt: bool = True
    seed: int = 42
    num_layers: int = 16

    # LR Schedule (passed directly to mlx-lm as a dict)
    lr_schedule: dict | None = None

    # Checkpointing
    save_every: int = 200
    steps_per_report: int = 10
    steps_per_eval: int = 100
    val_batches: int = 10

    # Output
    adapter_path: str = "artifacts/sft/adapters"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        """Load config from YAML file, merging with defaults."""
        path = Path(path)
        if not path.exists():
            console.print(f"[yellow]Config not found: {path} — using defaults[/yellow]")
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Map YAML keys to dataclass fields
        lora_params = raw.get("lora_parameters", {})
        lr_schedule = raw.get("lr_schedule", {})

        return cls(
            model=raw.get("model", cls.model),
            data_dir=raw.get("data", cls.data_dir),
            lora_rank=lora_params.get("rank", cls.lora_rank),
            lora_dropout=lora_params.get("dropout", cls.lora_dropout),
            lora_scale=lora_params.get("scale", cls.lora_scale),
            iters=raw.get("iters", cls.iters),
            batch_size=raw.get("batch_size", cls.batch_size),
            learning_rate=raw.get("learning_rate", cls.learning_rate),
            max_seq_length=raw.get("max_seq_length", cls.max_seq_length),
            grad_checkpoint=raw.get("grad_checkpoint", cls.grad_checkpoint),
            mask_prompt=raw.get("mask_prompt", cls.mask_prompt),
            seed=raw.get("seed", cls.seed),
            num_layers=raw.get("num_layers", cls.num_layers),
            lr_schedule=raw.get("lr_schedule"),
            save_every=raw.get("save_every", cls.save_every),
            steps_per_report=raw.get("steps_per_report", cls.steps_per_report),
            steps_per_eval=raw.get("steps_per_eval", cls.steps_per_eval),
            val_batches=raw.get("val_batches", cls.val_batches),
            adapter_path=raw.get("adapter_path", cls.adapter_path),
        )

    def to_mlx_config(self) -> dict[str, Any]:
        """
        Convert to mlx-lm lora config format.

        WHY CONVERT:
          Our YAML is human-friendly with documented fields.
          mlx-lm expects a flat dict matching CONFIG_DEFAULTS.
          This method bridges the two formats.
        """
        return {
            "model": self.model,
            "data": self.data_dir,
            "train": True,
            "fine_tune_type": "lora",
            "seed": self.seed,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "iters": self.iters,
            "learning_rate": self.learning_rate,
            "steps_per_report": self.steps_per_report,
            "steps_per_eval": self.steps_per_eval,
            "save_every": self.save_every,
            "val_batches": self.val_batches,
            "max_seq_length": self.max_seq_length,
            "grad_checkpoint": self.grad_checkpoint,
            "mask_prompt": self.mask_prompt,
            "adapter_path": self.adapter_path,
            "lora_parameters": {
                "rank": self.lora_rank,
                "dropout": self.lora_dropout,
                "scale": self.lora_scale,
            },
            "lr_schedule": self.lr_schedule,
        }


def prepare_data(config: SFTConfig) -> dict[str, int]:
    """
    Step 1: Convert processed data to MLX training format.

    Converts ToolCallingExample JSONL files to OpenAI chat format
    JSONL files that mlx-lm's ChatDataset can read.
    """
    from toolforge.data.mlx_format import prepare_mlx_training_data

    console.print("\n[bold blue]Step 1/3: Prepare Training Data[/bold blue]")
    stats = prepare_mlx_training_data(
        processed_dir=config.processed_dir,
        output_dir=config.data_dir,
    )

    total = sum(stats.values())
    console.print(f"  [green]Total: {total} examples prepared[/green]")
    return stats


def run_training(config: SFTConfig) -> Path:
    """
    Step 2: Run MLX LoRA training.

    Delegates to mlx-lm's LoRA trainer, which handles:
      - Model loading (with 4-bit quantization)
      - LoRA layer injection
      - Training loop with gradient accumulation
      - Checkpoint saving
      - Validation loss evaluation
    """
    console.print("\n[bold blue]Step 2/3: LoRA Fine-Tuning[/bold blue]")

    # Patch SSL for corporate proxy
    try:
        import httpx

        _orig = httpx.Client.__init__

        def _patched(self, *a, **kw):
            kw.setdefault("verify", False)
            _orig(self, *a, **kw)

        httpx.Client.__init__ = _patched
    except ImportError:
        pass

    mlx_config = config.to_mlx_config()

    console.print(f"  Model:      {config.model}")
    console.print(f"  Data:       {config.data_dir}")
    console.print(f"  LoRA rank:  {config.lora_rank}")
    console.print(f"  Iterations: {config.iters}")
    console.print(f"  Batch size: {config.batch_size}")
    console.print(f"  LR:         {config.learning_rate}")
    console.print(f"  Output:     {config.adapter_path}")

    # Write MLX config to temp file for mlx_lm.lora
    adapter_dir = Path(config.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    mlx_config_path = adapter_dir / "training_config.yaml"
    with open(mlx_config_path, "w") as f:
        yaml.dump(mlx_config, f, default_flow_style=False)

    console.print(f"\n  [cyan]Starting training...[/cyan]")
    start_time = time.time()

    # Run mlx-lm LoRA training
    from mlx_lm import lora as mlx_lora

    # Build args namespace from config
    import types

    args = types.SimpleNamespace(**mlx_config)
    args.resume_adapter_file = None
    args.test = False
    args.test_batches = 500
    args.config = None
    args.optimizer = "adam"
    args.optimizer_config = {"adam": {}}
    args.report_to = None
    args.project_name = None
    args.hf_dataset = None
    # WHY 0.5: Tells MLX to free Metal memory when >50% of the cache is unused.
    # On Apple Silicon, Metal doesn't release GPU memory eagerly — this forces
    # periodic cleanup to prevent OOM after validation passes eat into headroom.
    args.clear_cache_threshold = 0.5
    args.grad_accumulation_steps = 1

    mlx_lora.run(args)

    elapsed = time.time() - start_time
    console.print(f"\n  [bold green]Training complete in {elapsed/60:.1f} minutes[/bold green]")

    # Save our config alongside the adapters for reproducibility
    our_config_path = adapter_dir / "toolforge_config.yaml"
    with open(our_config_path, "w") as f:
        yaml.dump(mlx_config, f, default_flow_style=False)

    return adapter_dir


def run_sft_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    iters: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    skip_data_prep: bool = False,
) -> Path:
    """
    Run the complete SFT pipeline: data prep → training.

    Args:
        config_path: Path to SFT YAML config
        iters: Override iterations (for quick experiments)
        batch_size: Override batch size
        learning_rate: Override learning rate
        skip_data_prep: Skip data conversion (if already done)

    Returns:
        Path to the adapter directory
    """
    console.print(
        Panel(
            "[bold]ToolForge SFT Training Pipeline[/bold]\n"
            f"Config: {config_path}",
            title="Stage 4",
            border_style="blue",
        )
    )

    # Load config with overrides
    config = SFTConfig.from_yaml(config_path)
    if iters is not None:
        config.iters = iters
    if batch_size is not None:
        config.batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate

    # Step 1: Data prep
    if not skip_data_prep:
        prepare_data(config)
    else:
        console.print("\n[dim]Skipping data preparation (--skip-data-prep)[/dim]")

    # Step 2: Training
    adapter_dir = run_training(config)

    console.print(
        Panel(
            f"[bold green]SFT training complete![/bold green]\n"
            f"Adapters saved to: {adapter_dir}\n\n"
            f"Next steps:\n"
            f"  1. Evaluate: toolforge eval run --backend mlx --model-path {adapter_dir}\n"
            f"  2. Compare: check artifacts/reports/ for baseline vs SFT scores",
            title="Done",
            border_style="green",
        )
    )

    return adapter_dir
