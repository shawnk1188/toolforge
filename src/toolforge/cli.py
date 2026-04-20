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
    output: str = typer.Option("artifacts/reports", help="Output directory for reports"),
) -> None:
    """Run behavioral specs against a model and generate a pass/fail report."""
    console.print(f"\n[bold blue]ToolForge Eval[/bold blue] — Stage: {stage}")
    console.print(f"  Specs dir:  {specs_dir}")
    console.print(f"  Model:      {model_path or 'not specified'}")
    console.print(f"  Output:     {output}")
    console.print("\n[yellow]⚠ Eval harness will be implemented in Stage 3[/yellow]\n")


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
def data_download() -> None:
    """Download raw datasets from HuggingFace Hub."""
    console.print("\n[yellow]⚠ Data pipeline will be implemented in Stage 2[/yellow]\n")


@data_app.command("prepare")
def data_prepare(
    config: str = typer.Option("configs/data/default.yaml", help="Data config path"),
) -> None:
    """Process raw data into training format."""
    console.print("\n[yellow]⚠ Data pipeline will be implemented in Stage 2[/yellow]\n")


@data_app.command("validate")
def data_validate(
    config: str = typer.Option("configs/data/default.yaml", help="Data config path"),
) -> None:
    """Validate processed dataset quality and schema compliance."""
    console.print("\n[yellow]⚠ Data pipeline will be implemented in Stage 2[/yellow]\n")


# --- Train commands ---
train_app = typer.Typer(help="Model training")
app.add_typer(train_app, name="train")


@train_app.command("sft")
def train_sft(
    config: str = typer.Option("configs/training/sft.yaml", help="SFT config path"),
) -> None:
    """Run supervised fine-tuning with QLoRA."""
    console.print("\n[yellow]⚠ SFT training will be implemented in Stage 4[/yellow]\n")


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
