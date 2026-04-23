"""
ToolForge CLI — the single entry point for all operations.

Why Typer?
  - Auto-generates --help from function signatures and docstrings
  - Type-safe arguments with Python type hints
  - Subcommand groups (toolforge data download, toolforge eval run, etc.)
  - Same library used in your Lumina project — consistency across portfolio

Usage:
  toolforge --help           Show all commands
  toolforge eval run         Run behavioral specs against a model
  toolforge data download    Download datasets
  toolforge train sft        Run supervised fine-tuning
"""

import typer
from rich.console import Console

app = typer.Typer(
    name="toolforge",
    help="Spec-driven fine-tuning for precision tool-calling",
    no_args_is_help=True,
)

console = Console()

# ============================================================
# Subcommand groups — each maps to a module
# ============================================================

# --- Eval commands ---
eval_app = typer.Typer(help="Evaluation and spec management")
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    specs_dir: str = typer.Option("configs/specs", help="Path to specs directory"),
    stage: str = typer.Option("baseline", help="Evaluation stage: baseline, sft, dpo, all"),
    model_path: str = typer.Option(None, help="Path to model or HF model ID"),
    adapter_path: str = typer.Option(None, help="Path to LoRA adapter directory (for SFT/DPO eval)"),
    backend: str = typer.Option("mlx", help="Inference backend: mlx, ollama, dummy"),
    output: str = typer.Option("artifacts/reports", help="Output directory for reports"),
    max_samples: int = typer.Option(None, help="Limit samples per spec (for quick checks)"),
) -> None:
    """Run behavioral specs against a model and generate a pass/fail report."""
    from toolforge.eval.harness import run_all_specs, save_report
    from toolforge.eval.models import create_model_adapter

    # Create model adapter (optionally with LoRA adapters)
    adapter = create_model_adapter(
        backend=backend,
        model_id=model_path,
        adapter_path=adapter_path,
    )

    # Run all specs
    report = run_all_specs(
        specs_dir=specs_dir,
        model_fn=adapter,
        stage=stage,
        model_id=adapter.model_id,
        max_samples=max_samples,
    )

    # Save report
    save_report(report, output)


@eval_app.command("specs")
def eval_specs(
    specs_dir: str = typer.Option("configs/specs", help="Path to specs directory"),
) -> None:
    """Validate that all spec files are well-formed."""
    from toolforge.eval.specs import validate_all_specs

    validate_all_specs(specs_dir)


# --- Data commands ---
data_app = typer.Typer(help="Dataset management")
app.add_typer(data_app, name="data")


@data_app.command("download")
def data_download(
    output_dir: str = typer.Option("data/raw", help="Output directory for raw data"),
    max_examples: int = typer.Option(None, help="Limit examples per dataset (for dev speed)"),
) -> None:
    """Download raw datasets from HuggingFace Hub and convert to canonical format."""
    from toolforge.data.download import download_and_convert

    stats = download_and_convert(output_dir=output_dir, max_examples=max_examples)
    console.print(f"\n[bold green]Download complete:[/bold green] {stats}")


@data_app.command("prepare")
def data_prepare(
    max_examples: int = typer.Option(None, help="Limit total examples (for dev speed)"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Run the full data pipeline: download → validate → split → eval datasets."""
    from toolforge.data.prepare import run_full_pipeline

    run_full_pipeline(max_examples=max_examples, seed=seed)


@data_app.command("validate")
def data_validate(
    input_path: str = typer.Argument(..., help="Path to .jsonl file to validate"),
    keep_invalid: bool = typer.Option(False, help="Keep examples that fail checks"),
    keep_duplicates: bool = typer.Option(False, help="Keep duplicate examples"),
) -> None:
    """Validate a processed .jsonl dataset for quality and schema compliance."""
    from toolforge.data.validate import validate_dataset

    valid, report = validate_dataset(
        input_path=input_path,
        remove_duplicates=not keep_duplicates,
        remove_invalid=not keep_invalid,
    )
    console.print(f"\n[bold green]Validation complete:[/bold green] {len(valid)} valid examples")


@data_app.command("format")
def data_format(
    input_path: str = typer.Argument(
        "data/processed/train.jsonl",
        help="Path to .jsonl file with ToolCallingExample records",
    ),
    output_path: str = typer.Option(
        "data/processed/train_formatted.jsonl",
        help="Output path for chat-formatted .jsonl",
    ),
) -> None:
    """Format training data into Llama 3.2 chat template for SFT."""
    import json
    from pathlib import Path

    from toolforge.data.formatter import compute_token_stats, format_dataset_for_training
    from toolforge.data.schema import ToolCallingExample

    input_file = Path(input_path)
    if not input_file.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)

    # Load examples
    examples = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(ToolCallingExample(**json.loads(line)))

    console.print(f"\n[bold blue]Formatting {len(examples)} examples[/bold blue]")

    # Format
    formatted = format_dataset_for_training(examples)

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")

    # Stats
    stats = compute_token_stats(formatted)
    console.print(f"  Examples:           {stats['num_examples']}")
    console.print(f"  Est. total tokens:  {stats['estimated_total_tokens']:,}")
    console.print(f"  Est. avg tokens:    {stats['estimated_avg_tokens']:.0f}")
    console.print(f"  Over 2048 tokens:   {stats['over_2048_tokens']}")
    console.print(f"  Over 4096 tokens:   {stats['over_4096_tokens']}")
    console.print(f"\n[bold green]Saved to:[/bold green] {output_path}")


# --- Train commands ---
train_app = typer.Typer(help="Model training")
app.add_typer(train_app, name="train")


@train_app.command("sft")
def train_sft(
    config: str = typer.Option("configs/training/sft.yaml", help="SFT config path"),
    iters: int = typer.Option(None, help="Override training iterations"),
    batch_size: int = typer.Option(None, help="Override batch size"),
    learning_rate: float = typer.Option(None, "--lr", help="Override learning rate"),
    skip_data_prep: bool = typer.Option(False, help="Skip data conversion (if already done)"),
) -> None:
    """Run supervised fine-tuning with MLX LoRA on Apple Silicon."""
    from toolforge.training.sft import run_sft_pipeline

    run_sft_pipeline(
        config_path=config,
        iters=iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        skip_data_prep=skip_data_prep,
    )


@train_app.command("dpo")
def train_dpo(
    config: str = typer.Option("configs/training/dpo.yaml", help="DPO config path"),
) -> None:
    """Run Direct Preference Optimization on SFT checkpoint."""
    console.print("\n[yellow]⚠ DPO training will be implemented in Stage 5[/yellow]\n")


# --- Serve commands ---
serve_app = typer.Typer(help="Model serving")
app.add_typer(serve_app, name="serve")


@serve_app.command("start")
def serve_start(
    model_path: str = typer.Option(..., help="Path to merged model or adapter"),
    port: int = typer.Option(8000, help="Server port"),
) -> None:
    """Start the FastAPI inference server."""
    console.print("\n[yellow]⚠ Serving will be implemented in Stage 6[/yellow]\n")


if __name__ == "__main__":
    app()
