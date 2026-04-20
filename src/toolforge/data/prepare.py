"""
Data preparation: splitting, eval dataset generation, and chat formatting.

WHY THIS MODULE EXISTS:
  After download (raw data) and validation (clean data), we need to:
  1. Split into train/val/test with stratification by example_type
  2. Generate the specific eval datasets that each behavioral spec points to
  3. Format training data into the Llama 3.2 chat template

  This is the final step before training — the output of this module
  is directly consumed by the SFT trainer (Stage 4).

DESIGN:
  Stratified splitting ensures each split has proportional representation
  of all example types. Without this, you might get a test set with zero
  NO_TOOL examples, making the relevance_detection spec impossible to evaluate.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from toolforge.data.schema import (
    DatasetManifest,
    DatasetSplit,
    ExampleType,
    ToolCallingExample,
)

console = Console()


# ============================================================
# Stratified Splitting
# ============================================================


def stratified_split(
    examples: list[ToolCallingExample],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list[ToolCallingExample]]:
    """
    Split examples into train/val/test with stratification by example_type.

    WHY STRATIFIED:
      If we have 5% NO_TOOL examples and split randomly, the test set
      might get 0 NO_TOOL examples. Stratification guarantees each split
      has proportional representation of every example type.

    Args:
        examples: All validated examples
        train_ratio: Fraction for training (default 0.85)
        val_ratio: Fraction for validation during training (default 0.10)
        test_ratio: Fraction for final evaluation — never seen during training (default 0.05)
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    rng = random.Random(seed)

    # Group by type
    by_type: dict[str, list[ToolCallingExample]] = defaultdict(list)
    for ex in examples:
        by_type[ex.example_type.value].append(ex)

    splits: dict[str, list[ToolCallingExample]] = {"train": [], "val": [], "test": []}

    for etype, type_examples in by_type.items():
        rng.shuffle(type_examples)
        n = len(type_examples)
        n_val = max(1, int(n * val_ratio))     # At least 1 per type
        n_test = max(1, int(n * test_ratio))    # At least 1 per type
        n_train = n - n_val - n_test

        if n_train < 1:
            # Very few examples of this type — put all in train, duplicate for val/test
            splits["train"].extend(type_examples)
            splits["val"].extend(type_examples[:1])
            splits["test"].extend(type_examples[:1])
        else:
            splits["train"].extend(type_examples[:n_train])
            splits["val"].extend(type_examples[n_train:n_train + n_val])
            splits["test"].extend(type_examples[n_train + n_val:])

    # Shuffle each split (so types are mixed, not grouped)
    for split in splits.values():
        rng.shuffle(split)

    return splits


# ============================================================
# Eval Dataset Generation
# ============================================================


def generate_eval_datasets(
    examples: list[ToolCallingExample],
    output_dir: str | Path = "data/eval",
    seed: int = 42,
) -> dict[str, int]:
    """
    Generate the evaluation .jsonl files that each behavioral spec points to.

    This is the critical link between our data pipeline and the spec system.
    Each spec's `dataset` field points to a file generated here.

    Mapping:
      - tool_selection_test.jsonl    ← SINGLE_TOOL examples
      - argument_accuracy_test.jsonl ← SINGLE_TOOL examples (same data, different metric)
      - hallucination_test.jsonl     ← mix of all types (tests if model invents tools)
      - no_tool_needed_test.jsonl    ← NO_TOOL examples
      - multi_tool_test.jsonl        ← MULTI_TOOL examples
      - error_recovery_test.jsonl    ← ERROR_HANDLING examples (synthesized)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Group by type
    by_type: dict[str, list[ToolCallingExample]] = defaultdict(list)
    for ex in examples:
        by_type[ex.example_type.value].append(ex)

    stats: dict[str, int] = {}

    # 1. Tool selection test — single tool examples
    single_tool = by_type.get("single_tool", [])
    rng.shuffle(single_tool)
    selection_examples = single_tool[:500]
    stats["tool_selection_test"] = _write_eval_file(
        selection_examples, output_dir / "tool_selection_test.jsonl"
    )

    # 2. Argument accuracy test — same pool, different evaluation
    arg_examples = single_tool[:500]
    stats["argument_accuracy_test"] = _write_eval_file(
        arg_examples, output_dir / "argument_accuracy_test.jsonl"
    )

    # 3. Hallucination test — mix all types
    all_examples = [ex for exs in by_type.values() for ex in exs]
    rng.shuffle(all_examples)
    halluc_examples = all_examples[:500]
    stats["hallucination_test"] = _write_eval_file(
        halluc_examples, output_dir / "hallucination_test.jsonl"
    )

    # 4. No tool needed test — NO_TOOL examples
    no_tool = by_type.get("no_tool", [])
    rng.shuffle(no_tool)
    no_tool_examples = no_tool[:300]
    stats["no_tool_needed_test"] = _write_eval_file(
        no_tool_examples, output_dir / "no_tool_needed_test.jsonl"
    )

    # 5. Multi-tool test — MULTI_TOOL examples
    multi_tool = by_type.get("multi_tool", [])
    rng.shuffle(multi_tool)
    multi_examples = multi_tool[:200]
    stats["multi_tool_test"] = _write_eval_file(
        multi_examples, output_dir / "multi_tool_test.jsonl"
    )

    # 6. Error recovery test — synthesize from single_tool examples
    error_examples = _synthesize_error_examples(single_tool[:200], rng)
    stats["error_recovery_test"] = _write_eval_file(
        error_examples, output_dir / "error_recovery_test.jsonl"
    )

    return stats


def _write_eval_file(examples: list[ToolCallingExample], filepath: Path) -> int:
    """Write examples in the eval harness format (prompt + expected + tool_schema)."""
    with open(filepath, "w") as f:
        for ex in examples:
            eval_record = ex.to_eval_format()
            f.write(json.dumps(eval_record) + "\n")
    console.print(f"  [dim]Wrote {len(examples)} examples to {filepath}[/dim]")
    return len(examples)


def _synthesize_error_examples(
    source_examples: list[ToolCallingExample],
    rng: random.Random,
) -> list[ToolCallingExample]:
    """
    Synthesize error-handling examples from normal tool-calling examples.

    WHY SYNTHESIZE:
      Very few open-source datasets include tool error scenarios. We create
      them by taking normal examples and adding an error context to the
      conversation, testing whether the model handles it gracefully.

    Approach:
      Take a normal tool call example, simulate that the tool returned an
      error, and set the expected response to acknowledge the error.
    """
    error_examples = []
    error_messages = [
        "Service temporarily unavailable (503)",
        "Rate limit exceeded. Please try again later.",
        "Invalid API key. Authentication failed.",
        "Resource not found (404)",
        "Request timeout after 30 seconds",
        "Internal server error (500)",
    ]

    for i, ex in enumerate(source_examples):
        if not ex.expected_tool_calls:
            continue

        tc = ex.expected_tool_calls[0]
        error_msg = rng.choice(error_messages)

        # Modify the query to include the error context
        error_query = (
            f"{ex.user_query}\n\n"
            f"[System: The tool '{tc.name}' was called but returned an error: "
            f'"{error_msg}". Please respond appropriately.]'
        )

        try:
            error_ex = ToolCallingExample(
                id=f"synthesized_error:{i}",
                system_prompt=ex.system_prompt,
                user_query=error_query,
                available_tools=ex.available_tools,
                expected_tool_calls=[],  # No tool call expected after error
                expected_response=(
                    f"I apologize, but I encountered an error when trying to use {tc.name}: "
                    f"{error_msg}. Would you like me to try again or help with something else?"
                ),
                example_type=ExampleType.ERROR_HANDLING,
                source_dataset="synthesized",
            )
            error_examples.append(error_ex)
        except Exception:
            continue

    return error_examples[:200]


# ============================================================
# Save Training Data
# ============================================================


def save_splits(
    splits: dict[str, list[ToolCallingExample]],
    output_dir: str | Path = "data/processed",
    source_datasets: list[str] | None = None,
) -> DatasetManifest:
    """
    Save train/val/test splits as .jsonl files and generate a manifest.

    The manifest is saved alongside the data so we always know:
    - How many examples per split
    - Distribution of example types
    - Which source datasets contributed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_meta = []

    for split_name, examples in splits.items():
        filepath = output_dir / f"{split_name}.jsonl"

        with open(filepath, "w") as f:
            for ex in examples:
                f.write(ex.model_dump_json() + "\n")

        type_dist = Counter(ex.example_type.value for ex in examples)
        source_dist = Counter(ex.source_dataset for ex in examples)

        split_meta.append(DatasetSplit(
            split_name=split_name,
            num_examples=len(examples),
            type_distribution=dict(type_dist),
            source_distribution=dict(source_dist),
            filepath=str(filepath),
        ))

    total = sum(s.num_examples for s in split_meta)

    manifest = DatasetManifest(
        total_examples=total,
        splits=split_meta,
        processing_timestamp=datetime.now(timezone.utc).isoformat(),
        source_datasets=source_datasets or [],
    )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    # Print summary
    _print_split_summary(splits, manifest)

    return manifest


def _print_split_summary(
    splits: dict[str, list[ToolCallingExample]],
    manifest: DatasetManifest,
) -> None:
    """Print a summary table of the processed splits."""
    table = Table(title="Dataset Splits", show_lines=True)
    table.add_column("Split", style="cyan")
    table.add_column("Examples", justify="right")
    table.add_column("single_tool", justify="right")
    table.add_column("multi_tool", justify="right")
    table.add_column("no_tool", justify="right")
    table.add_column("error_handling", justify="right")

    for split_name, examples in splits.items():
        type_dist = Counter(ex.example_type.value for ex in examples)
        table.add_row(
            split_name,
            f"[bold]{len(examples)}[/bold]",
            str(type_dist.get("single_tool", 0)),
            str(type_dist.get("multi_tool", 0)),
            str(type_dist.get("no_tool", 0)),
            str(type_dist.get("error_handling", 0)),
        )

    console.print(table)
    console.print(f"\n[bold green]Total: {manifest.total_examples} examples[/bold green]")


# ============================================================
# Full Pipeline (Download → Validate → Split → Eval Datasets)
# ============================================================


def run_full_pipeline(
    max_examples: int | None = None,
    seed: int = 42,
) -> None:
    """
    Run the complete data preparation pipeline.

    This is the main entry point that chains all steps:
    1. Download datasets from HuggingFace
    2. Validate and clean
    3. Split into train/val/test
    4. Generate eval datasets for specs
    """
    from toolforge.data.download import download_and_convert
    from toolforge.data.validate import validate_dataset

    console.print("\n[bold]=" * 60)
    console.print("[bold]  ToolForge Data Pipeline[/bold]")
    console.print("[bold]=" * 60)

    # Step 1: Download
    console.print("\n[bold blue]Step 1/4: Download & Convert[/bold blue]")
    download_stats = download_and_convert(
        output_dir="data/raw",
        max_examples=max_examples,
    )

    # Step 2: Validate
    console.print("\n[bold blue]Step 2/4: Validate & Clean[/bold blue]")
    all_examples: list[ToolCallingExample] = []
    for ds_key in download_stats:
        raw_path = Path("data/raw") / f"{ds_key}_converted.jsonl"
        if raw_path.exists():
            valid, report = validate_dataset(raw_path)
            all_examples.extend(valid)

    if not all_examples:
        console.print("[red]No valid examples after validation! Check your data.[/red]")
        return

    # Step 3: Split
    console.print("\n[bold blue]Step 3/4: Stratified Split[/bold blue]")
    splits = stratified_split(all_examples, seed=seed)
    manifest = save_splits(
        splits,
        source_datasets=list(download_stats.keys()),
    )

    # Step 4: Generate eval datasets
    console.print("\n[bold blue]Step 4/4: Generate Eval Datasets[/bold blue]")
    # Use test split for eval datasets (never seen during training)
    test_examples = splits["test"]
    # But we need enough examples, so supplement from val if test is small
    eval_pool = splits["test"] + splits["val"]
    eval_stats = generate_eval_datasets(eval_pool, seed=seed)

    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print(f"  Training data: data/processed/train.jsonl ({len(splits['train'])} examples)")
    console.print(f"  Eval datasets: data/eval/ ({sum(eval_stats.values())} total eval examples)")


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    run_full_pipeline(max_examples=5000)
