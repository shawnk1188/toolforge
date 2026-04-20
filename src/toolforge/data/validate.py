"""
Data validation pipeline.

WHY THIS MODULE EXISTS:
  Even after converting datasets to our canonical schema, we need a
  second pass that catches SEMANTIC issues the schema can't:
    - Duplicate examples (same user_query + same tool call)
    - Tool definitions with no parameters (likely malformed)
    - Unreasonably long/short queries
    - Imbalanced example types (90% single_tool, 2% no_tool)

  This module validates a .jsonl file and produces a quality report.
  Think of it as "data unit tests" — run before training to catch
  issues that would silently degrade model quality.

DESIGN:
  Each check is a standalone function that takes a ToolCallingExample
  and returns (pass: bool, reason: str). Checks are composable —
  you can add new checks without modifying existing ones.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from toolforge.data.schema import ExampleType, ToolCallingExample

console = Console()


# ============================================================
# Individual Validation Checks
# ============================================================


def check_query_length(example: ToolCallingExample, min_len: int = 10, max_len: int = 2000) -> tuple[bool, str]:
    """Query should be reasonable length — not 3 words, not a novel."""
    length = len(example.user_query)
    if length < min_len:
        return False, f"Query too short ({length} chars, min {min_len})"
    if length > max_len:
        return False, f"Query too long ({length} chars, max {max_len})"
    return True, ""


def check_tool_definitions(example: ToolCallingExample) -> tuple[bool, str]:
    """Each example should have at least one tool available (even no_tool examples need context)."""
    if not example.available_tools:
        return False, "No tool definitions provided"
    for tool in example.available_tools:
        if not tool.description or len(tool.description) < 5:
            return False, f"Tool '{tool.name}' has no meaningful description"
    return True, ""


def check_tool_count(example: ToolCallingExample, max_tools: int = 20) -> tuple[bool, str]:
    """
    Too many tools creates unrealistically long prompts.
    Production systems rarely have >20 tools in a single prompt.
    """
    if len(example.available_tools) > max_tools:
        return False, f"Too many tools ({len(example.available_tools)}, max {max_tools})"
    return True, ""


def check_argument_types(example: ToolCallingExample) -> tuple[bool, str]:
    """Tool call arguments should be basic JSON types, not nested garbage."""
    for tc in example.expected_tool_calls:
        for key, value in tc.arguments.items():
            if isinstance(value, (dict, list)):
                # Nested args are OK but flag deeply nested ones
                if isinstance(value, dict) and any(isinstance(v, dict) for v in value.values()):
                    return False, f"Deeply nested arguments in tool '{tc.name}'"
    return True, ""


def check_no_empty_tool_names(example: ToolCallingExample) -> tuple[bool, str]:
    """Tool names should not be empty strings."""
    for tool in example.available_tools:
        if not tool.name.strip():
            return False, "Empty tool name in available_tools"
    for tc in example.expected_tool_calls:
        if not tc.name.strip():
            return False, "Empty tool name in expected_tool_calls"
    return True, ""


# List of all checks to run
ALL_CHECKS = [
    check_query_length,
    check_tool_definitions,
    check_tool_count,
    check_argument_types,
    check_no_empty_tool_names,
]


# ============================================================
# Deduplication
# ============================================================


def compute_example_hash(example: ToolCallingExample) -> str:
    """
    Compute a content hash for deduplication.

    We hash the user_query + tool call names (not arguments, because
    slight argument variations of the same query are valuable training data).
    """
    key_parts = [
        example.user_query.strip().lower(),
        "|".join(sorted(tc.name for tc in example.expected_tool_calls)),
    ]
    key = "||".join(key_parts)
    return hashlib.md5(key.encode()).hexdigest()


# ============================================================
# Main Validation Pipeline
# ============================================================


def validate_dataset(
    input_path: str | Path,
    remove_duplicates: bool = True,
    remove_invalid: bool = True,
) -> tuple[list[ToolCallingExample], dict[str, Any]]:
    """
    Validate a .jsonl dataset and return cleaned examples + quality report.

    Args:
        input_path: Path to .jsonl file with ToolCallingExample records
        remove_duplicates: Whether to deduplicate
        remove_invalid: Whether to remove examples that fail checks

    Returns:
        (valid_examples, quality_report)
    """
    input_path = Path(input_path)
    console.print(f"\n[bold]Validating:[/bold] {input_path}")

    # Load and parse
    examples: list[ToolCallingExample] = []
    parse_errors = 0
    total_lines = 0

    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                example = ToolCallingExample(**raw)
                examples.append(example)
            except (json.JSONDecodeError, ValidationError) as e:
                parse_errors += 1

    console.print(f"  Parsed: {len(examples)}/{total_lines} lines ({parse_errors} errors)")

    # Run validation checks
    check_failures: Counter = Counter()
    valid_examples: list[ToolCallingExample] = []

    for example in examples:
        passed_all = True
        for check_fn in ALL_CHECKS:
            passed, reason = check_fn(example)
            if not passed:
                check_failures[check_fn.__name__] += 1
                passed_all = False
                break  # First failure is enough

        if passed_all or not remove_invalid:
            valid_examples.append(example)

    # Deduplication
    dedup_removed = 0
    if remove_duplicates:
        seen_hashes: set[str] = set()
        unique_examples: list[ToolCallingExample] = []
        for example in valid_examples:
            h = compute_example_hash(example)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_examples.append(example)
            else:
                dedup_removed += 1
        valid_examples = unique_examples

    # Compute type distribution
    type_dist = Counter(ex.example_type.value for ex in valid_examples)

    # Build quality report
    report = {
        "input_path": str(input_path),
        "total_lines": total_lines,
        "parse_errors": parse_errors,
        "check_failures": dict(check_failures),
        "duplicates_removed": dedup_removed,
        "valid_examples": len(valid_examples),
        "type_distribution": dict(type_dist),
    }

    # Print report
    _print_quality_report(report)

    return valid_examples, report


def _print_quality_report(report: dict[str, Any]) -> None:
    """Print a formatted quality report."""
    table = Table(title="Data Quality Report", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total lines", str(report["total_lines"]))
    table.add_row("Parse errors", f"[red]{report['parse_errors']}[/red]" if report["parse_errors"] else "0")
    table.add_row("Duplicates removed", str(report["duplicates_removed"]))
    table.add_row("[bold]Valid examples[/bold]", f"[bold green]{report['valid_examples']}[/bold green]")

    # Type distribution
    for etype, count in sorted(report["type_distribution"].items()):
        table.add_row(f"  {etype}", str(count))

    # Check failures
    if report["check_failures"]:
        table.add_row("", "")
        table.add_row("[bold yellow]Check Failures[/bold yellow]", "")
        for check_name, count in sorted(report["check_failures"].items()):
            table.add_row(f"  {check_name}", f"[yellow]{count}[/yellow]")

    console.print(table)
