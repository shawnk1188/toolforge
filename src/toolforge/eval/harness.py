"""
Evaluation Harness — runs behavioral specs against a model.

WHY THIS MODULE EXISTS:
  This is the "test runner" for our spec-driven approach. Just like
  pytest discovers test files and runs them, the harness discovers
  spec YAML files, loads the evaluation dataset for each spec,
  runs the model on each example, computes the metric, and produces
  a pass/fail report.

DESIGN:
  The harness is MODEL-AGNOSTIC. It doesn't know if it's evaluating
  a base Llama model, an SFT checkpoint, or a DPO-tuned adapter.
  It receives a "model_fn" callable that takes a prompt and returns
  a parsed tool-call dict. This decoupling means we can evaluate:
    - HuggingFace models (transformers)
    - MLX models (Apple Silicon)
    - Ollama models (local API)
    - Any future inference backend

  This is the Strategy Pattern — the harness defines the evaluation
  algorithm, and the model_fn is the pluggable strategy.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from toolforge.eval.metrics import get_metric
from toolforge.eval.specs import (
    BehavioralSpec,
    SpecReport,
    SpecResult,
    load_all_specs,
)

console = Console()

# Type alias: a function that takes a prompt string and returns a parsed dict
ModelFn = Callable[[str], dict[str, Any]]


def load_eval_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    """
    Load an evaluation dataset from a JSONL file.

    Each line is a JSON object with at minimum:
      - "prompt": the input to the model
      - "expected": the expected output (tool call or refusal)
      - "tool_schema": available tools for this example

    Returns:
        List of evaluation examples
    """
    filepath = Path(dataset_path)
    if not filepath.exists():
        console.print(f"[yellow]Dataset not found: {filepath} — skipping[/yellow]")
        return []

    examples = []
    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON at {filepath}:{line_num}: {e}[/red]")

    return examples


def run_spec(
    spec: BehavioralSpec,
    model_fn: ModelFn,
    max_samples: int | None = None,
    stage: str = "baseline",
) -> SpecResult:
    """
    Run a single behavioral spec against a model.

    Args:
        spec: The behavioral spec to evaluate
        model_fn: A callable that takes a prompt and returns a parsed tool-call dict
        max_samples: Override spec's num_samples (useful for quick checks)
        stage: Evaluation stage label (baseline, sft, dpo)

    Returns:
        SpecResult with score, pass/fail, and details
    """
    # Load the evaluation dataset
    examples = load_eval_dataset(spec.dataset)

    if not examples:
        return SpecResult(
            spec_name=spec.name,
            score=0.0,
            threshold=spec.threshold,
            passed=False,
            num_evaluated=0,
            num_passed=0,
            stage=stage,
            error=f"No evaluation data found at {spec.dataset}",
        )

    # Limit samples
    n_samples = min(max_samples or spec.num_samples, len(examples))
    examples = examples[:n_samples]

    # Get the metric function
    metric_fn = get_metric(spec.metric.value)

    # Evaluate each example
    num_passed = 0
    for example in examples:
        user_query = example.get("prompt", "")
        expected = example.get("expected", {})
        tool_schema = example.get("tool_schema", {})
        system_prompt = example.get("system_prompt", "")

        try:
            # Build a complete formatted prompt with tool context
            # WHY: model_fn receives a pre-formatted string (Llama 3.2 chat template)
            # that includes the system prompt with tool definitions. This way the
            # model adapter doesn't need to know about eval dataset structure.
            from toolforge.eval.models import build_eval_prompt

            formatted_prompt = build_eval_prompt(
                user_query=user_query,
                system_prompt=system_prompt,
                tool_schema=tool_schema,
            )

            # Run model inference
            predicted = model_fn(formatted_prompt)

            # Evaluate with the metric
            if metric_fn(predicted, expected, tool_schema):
                num_passed += 1
        except Exception as e:
            # Model errors count as failures
            console.print(f"[dim red]  Error on example: {e}[/dim red]")

    score = num_passed / n_samples if n_samples > 0 else 0.0
    passed = score >= spec.threshold

    return SpecResult(
        spec_name=spec.name,
        score=score,
        threshold=spec.threshold,
        passed=passed,
        num_evaluated=n_samples,
        num_passed=num_passed,
        stage=stage,
    )


def run_all_specs(
    specs_dir: str | Path,
    model_fn: ModelFn,
    stage: str = "baseline",
    model_id: str = "unknown",
    max_samples: int | None = None,
) -> SpecReport:
    """
    Run ALL behavioral specs and generate a report.

    This is the main entry point — equivalent to running `pytest` on
    all test files in a directory.
    """
    specs = load_all_specs(specs_dir)

    console.print(
        Panel(
            f"[bold]Running {len(specs)} specs[/bold]\n"
            f"Stage: {stage}  |  Model: {model_id}",
            title="ToolForge Eval",
            border_style="blue",
        )
    )

    results = []
    for spec in specs:
        console.print(f"\n  [cyan]Running:[/cyan] {spec.name} ({spec.metric.value})")
        result = run_spec(spec, model_fn, max_samples=max_samples, stage=stage)
        results.append(result)

        # Print inline result
        status = "[bold green]PASS[/bold green]" if result.passed else "[bold red]FAIL[/bold red]"
        console.print(
            f"  {status}  score={result.score:.3f}  "
            f"threshold={result.threshold:.2f}  "
            f"({result.num_passed}/{result.num_evaluated})"
        )

    # Build report
    passed_count = sum(1 for r in results if r.passed)
    report = SpecReport(
        stage=stage,
        results=results,
        total_specs=len(results),
        passed_specs=passed_count,
        failed_specs=len(results) - passed_count,
        model_id=model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Print summary
    _print_summary(report)

    return report


def _print_summary(report: SpecReport) -> None:
    """Print a beautiful summary table of spec results."""
    table = Table(title=f"\nSpec Results — Stage: {report.stage}", show_lines=True)
    table.add_column("Status", style="bold", width=8)
    table.add_column("Spec", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Samples", justify="right")

    for result in report.results:
        status = "[green]PASS" if result.passed else "[red]FAIL"
        score_style = "green" if result.passed else "red"
        table.add_row(
            status,
            result.spec_name,
            f"[{score_style}]{result.score:.3f}[/{score_style}]",
            f"{result.threshold:.2f}",
            f"{result.num_passed}/{result.num_evaluated}",
        )

    console.print(table)

    # Overall verdict
    if report.passed_specs == report.total_specs:
        console.print(
            f"\n[bold green]ALL SPECS PASSED ({report.passed_specs}/{report.total_specs})[/bold green]\n"
        )
    else:
        console.print(
            f"\n[bold red]{report.failed_specs} SPEC(S) FAILED "
            f"({report.passed_specs}/{report.total_specs} passed)[/bold red]\n"
        )


def save_report(report: SpecReport, output_dir: str | Path) -> Path:
    """Save the spec report as JSON for historical tracking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"spec_report_{report.stage}_{report.timestamp.replace(':', '-')}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        f.write(report.model_dump_json(indent=2))

    console.print(f"[dim]Report saved to {filepath}[/dim]")
    return filepath
