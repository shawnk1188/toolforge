"""
Behavioral Spec loader and validator.

WHY THIS MODULE EXISTS:
  Specs are YAML files, which means they can contain typos, missing fields,
  or invalid values. If a spec file has `threshold: 1.5` (impossible) or
  misspells the metric name, we want to catch that BEFORE running a 2-hour
  evaluation — not after.

  This module loads specs into Pydantic models that enforce the schema,
  so any invalid spec file is caught immediately with a clear error message.

DESIGN PATTERN:
  Parse → Validate → Use (same "validate at boundary" pattern from your
  Lumina project — validate YAML at load time, then trust the typed
  objects downstream).
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.table import Table

console = Console()


# ============================================================
# Schema: What a valid spec looks like
# ============================================================


class Priority(str, Enum):
    """Spec priority levels — determines training focus order."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class MetricName(str, Enum):
    """
    Allowed metric names — each maps to a function in toolforge.eval.metrics.

    WHY an enum instead of a free string?
      If someone types "excat_match" (typo), the Pydantic validator catches it
      immediately instead of silently using a nonexistent metric function.
    """

    EXACT_MATCH = "exact_match"
    JSON_SCHEMA_MATCH = "json_schema_match"
    TOOL_EXISTS_CHECK = "tool_exists_check"
    CORRECT_REFUSAL_RATE = "correct_refusal_rate"
    SEQUENCE_EXACT_MATCH = "sequence_exact_match"
    GRACEFUL_ERROR_RATE = "graceful_error_rate"


class BehavioralSpec(BaseModel):
    """
    A single behavioral specification for model evaluation.

    This is the contract: if the model scores >= threshold on the
    specified metric against the evaluation dataset, the spec PASSES.
    """

    name: str = Field(..., description="Unique spec identifier")
    description: str = Field(..., description="Human-readable description")
    metric: MetricName = Field(..., description="Metric function to compute")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Minimum score to pass (0-1)")
    dataset: str = Field(..., description="Path to evaluation dataset (.jsonl)")
    num_samples: int = Field(..., gt=0, description="Number of samples to evaluate")
    priority: Priority = Field(..., description="Spec priority level")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    baseline_expected: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Expected base model score (filled after baseline eval)"
    )

    @field_validator("name")
    @classmethod
    def name_must_be_snake_case(cls, v: str) -> str:
        """Enforce snake_case naming for consistency."""
        if not v.replace("_", "").isalnum():
            raise ValueError(f"Spec name must be snake_case alphanumeric, got: '{v}'")
        return v


# ============================================================
# Spec result: What evaluation produces
# ============================================================


class SpecResult(BaseModel):
    """Result of running a single spec against a model."""

    spec_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    passed: bool
    num_evaluated: int
    num_passed: int
    stage: str = Field(..., description="Evaluation stage: baseline, sft, dpo")
    error: Optional[str] = None

    @property
    def improvement_over_baseline(self) -> Optional[float]:
        """Calculate improvement if baseline score is known."""
        return None  # Will be computed when we have baseline data


class SpecReport(BaseModel):
    """Aggregated report across all specs for a given stage."""

    stage: str
    results: list[SpecResult]
    total_specs: int
    passed_specs: int
    failed_specs: int
    model_id: str
    timestamp: str

    @property
    def pass_rate(self) -> float:
        return self.passed_specs / self.total_specs if self.total_specs > 0 else 0.0


# ============================================================
# Loading and validation functions
# ============================================================


def load_spec(filepath: str | Path) -> BehavioralSpec:
    """
    Load a single spec from a YAML file.

    Raises:
        FileNotFoundError: if the file doesn't exist
        pydantic.ValidationError: if the YAML content is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Spec file not found: {filepath}")

    with open(filepath) as f:
        raw = yaml.safe_load(f)

    # Pydantic validates the schema — any invalid field raises immediately
    return BehavioralSpec(**raw)


def load_all_specs(specs_dir: str | Path) -> list[BehavioralSpec]:
    """
    Load all spec YAML files from a directory.

    Returns specs sorted by priority (critical first), then by name.
    """
    specs_dir = Path(specs_dir)
    if not specs_dir.is_dir():
        raise NotADirectoryError(f"Specs directory not found: {specs_dir}")

    specs = []
    for filepath in sorted(specs_dir.glob("*.yaml")):
        # Skip README or non-spec files
        if filepath.stem.startswith("_"):
            continue
        spec = load_spec(filepath)
        specs.append(spec)

    # Sort: critical first, then high, then medium
    priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2}
    specs.sort(key=lambda s: (priority_order[s.priority], s.name))

    return specs


def validate_all_specs(specs_dir: str | Path) -> bool:
    """
    Validate all spec files and print a summary table.

    This is called by `toolforge eval specs` and by the CI pipeline.
    Returns True if all specs are valid, False otherwise.
    """
    specs_dir = Path(specs_dir)
    console.print(f"\n[bold]Validating specs in:[/bold] {specs_dir}\n")

    # Build a rich table for pretty output
    table = Table(title="Behavioral Specs", show_lines=True)
    table.add_column("Status", style="bold", width=8)
    table.add_column("Name", style="cyan")
    table.add_column("Metric", style="green")
    table.add_column("Threshold", justify="right")
    table.add_column("Priority", style="yellow")
    table.add_column("Samples", justify="right")

    all_valid = True
    yaml_files = sorted(specs_dir.glob("*.yaml"))

    if not yaml_files:
        console.print("[red]No .yaml spec files found![/red]")
        return False

    for filepath in yaml_files:
        if filepath.stem.startswith("_"):
            continue
        try:
            spec = load_spec(filepath)
            table.add_row(
                "[green]VALID[/green]",
                spec.name,
                spec.metric.value,
                f"{spec.threshold:.2f}",
                spec.priority.value,
                str(spec.num_samples),
            )
        except Exception as e:
            all_valid = False
            table.add_row(
                "[red]ERROR[/red]",
                filepath.stem,
                str(e)[:40],
                "—",
                "—",
                "—",
            )

    console.print(table)

    if all_valid:
        console.print(f"\n[bold green]All {len(yaml_files)} specs are valid.[/bold green]\n")
    else:
        console.print("\n[bold red]Some specs have errors. Fix them before proceeding.[/bold red]\n")

    return all_valid
