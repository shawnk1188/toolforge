"""
Generate preference pairs for DPO training.

WHY THIS MODULE EXISTS:
  DPO (Direct Preference Optimization) needs paired data:
    - chosen:   the CORRECT response for a given prompt
    - rejected: the SFT model's INCORRECT response for that same prompt

  We generate these by running the SFT model on our eval datasets and
  comparing its output to the ground truth. Where the model gets it wrong,
  we have a natural (chosen, rejected) pair.

  This is exactly how production DPO datasets are built — the model learns
  from its OWN mistakes, not from hypothetical bad examples.

DATA FLOW:
  eval datasets (JSONL)
    → SFT model inference (get model's actual outputs)
    → compare to ground truth
    → keep only INCORRECT outputs (where model fails)
    → format as (prompt, chosen, rejected) triples
    → JSONL output for DPO training

WHY NOT JUST USE ALL EXAMPLES:
  DPO learns from CONTRASTS. If the model already gets an example right,
  there's no contrast to learn from. We only include examples where the
  model produces a different (worse) output than the ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def generate_preference_pairs(
    eval_dir: str | Path = "data/eval",
    adapter_path: str | Path = "artifacts/sft/adapters",
    output_path: str | Path = "data/dpo/train.jsonl",
    model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_per_spec: int | None = None,
) -> dict[str, Any]:
    """
    Generate DPO preference pairs from SFT model failures.

    For each eval dataset:
      1. Run the SFT model on each example
      2. Compare model output to expected output
      3. Where model fails → create (prompt, chosen, rejected) pair

    Args:
        eval_dir: Directory with eval JSONL files
        adapter_path: Path to SFT adapter weights
        output_path: Where to write the preference pairs
        model_id: Base model identifier
        max_per_spec: Max examples per eval dataset (None = all)

    Returns:
        Stats dict with counts per dataset
    """
    from toolforge.eval.models import MLXModelAdapter, build_eval_prompt

    eval_dir = Path(eval_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load SFT model
    console.print(f"\n[bold blue]Loading SFT model for preference pair generation[/bold blue]")
    adapter = MLXModelAdapter(
        model_id=model_id,
        adapter_path=str(adapter_path),
        max_tokens=512,
        temperature=0.0,
    )
    adapter.load()

    # Process each eval dataset
    stats = {"total_pairs": 0, "total_examples": 0, "per_dataset": {}}

    # Target the failing specs — these are where DPO will have the most impact
    datasets = [
        ("argument_accuracy_test.jsonl", "argument_accuracy"),
        ("error_recovery_test.jsonl", "error_recovery"),
        ("multi_tool_test.jsonl", "multi_tool_sequencing"),
        ("no_tool_needed_test.jsonl", "relevance_detection"),
    ]

    all_pairs = []

    for filename, spec_name in datasets:
        filepath = eval_dir / filename
        if not filepath.exists():
            console.print(f"  [yellow]{filepath} not found — skipping[/yellow]")
            continue

        console.print(f"\n  [cyan]Processing:[/cyan] {spec_name} ({filename})")

        examples = _load_eval_examples(filepath)
        if max_per_spec:
            examples = examples[:max_per_spec]

        pairs = []
        correct = 0

        for i, example in enumerate(examples):
            if (i + 1) % 10 == 0:
                console.print(f"    [{i+1}/{len(examples)}] {len(pairs)} pairs so far...")

            prompt = example.get("prompt", "")
            expected = example.get("expected", {})
            tool_schema = example.get("tool_schema", {})
            system_prompt = example.get("system_prompt", "")

            # Build the formatted prompt
            formatted_prompt = build_eval_prompt(
                user_query=prompt,
                system_prompt=system_prompt,
                tool_schema=tool_schema,
            )

            try:
                # Get SFT model's actual output
                raw_output = adapter.generate(formatted_prompt)
                model_parsed = adapter.parse_output(raw_output)

                # Format chosen and rejected as strings
                chosen_str = _format_expected_as_string(expected)
                rejected_str = raw_output.strip()

                # Check if model got it right
                if _is_correct(model_parsed, expected, spec_name):
                    correct += 1
                    continue  # Skip correct examples — no contrast to learn from

                # Only create pair if rejected is meaningfully different
                if rejected_str and rejected_str != chosen_str:
                    pair = {
                        "prompt": formatted_prompt,
                        "chosen": chosen_str,
                        "rejected": rejected_str,
                        "spec": spec_name,
                    }
                    pairs.append(pair)

            except Exception as e:
                # Model errors → the model's output is the error-causing garbage
                pair = {
                    "prompt": formatted_prompt,
                    "chosen": _format_expected_as_string(expected),
                    "rejected": f"[generation error: {str(e)[:100]}]",
                    "spec": spec_name,
                }
                pairs.append(pair)

        all_pairs.extend(pairs)
        stats["total_examples"] += len(examples)
        stats["per_dataset"][spec_name] = {
            "total": len(examples),
            "correct": correct,
            "pairs": len(pairs),
            "accuracy": correct / len(examples) if examples else 0,
        }

        console.print(
            f"    [green]{correct}/{len(examples)} correct[/green] → "
            f"[bold]{len(pairs)} preference pairs[/bold]"
        )

    # Write all pairs
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    stats["total_pairs"] = len(all_pairs)

    console.print(
        f"\n[bold green]Generated {len(all_pairs)} preference pairs[/bold green] "
        f"→ {output_path}"
    )

    return stats


def _load_eval_examples(filepath: Path) -> list[dict]:
    """Load evaluation examples from JSONL."""
    examples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def _format_expected_as_string(expected: dict) -> str:
    """
    Format the ground truth expected output as a string.

    WHY STRING NOT DICT:
      DPO trains on text sequences. Both chosen and rejected must be
      strings that the model could have generated. We format the
      expected dict as JSON (for tool calls) or plain text (for refusals).
    """
    if expected.get("tool"):
        # Tool call expected
        return json.dumps({
            "name": expected["tool"],
            "arguments": expected.get("arguments", {}),
        })
    elif expected.get("tools"):
        # Multi-tool call expected
        return json.dumps([
            {"name": t["tool"], "arguments": t.get("arguments", {})}
            for t in expected["tools"]
        ])
    elif expected.get("response"):
        # Text response expected (refusal / no-tool)
        return expected["response"]
    else:
        return json.dumps(expected)


def _is_correct(predicted: dict, expected: dict, spec_name: str) -> bool:
    """
    Check if the model's prediction matches the expected output.

    Uses simple heuristics per spec type — we don't need the full
    metric here, just a quick check to filter obvious failures.
    """
    if spec_name == "relevance_detection":
        # Should NOT call a tool → check if model also didn't call a tool
        return "response" in predicted and "tool" not in predicted

    elif spec_name == "error_recovery":
        # Should respond with text (acknowledge error) → check for text response
        return "response" in predicted and "tool" not in predicted

    elif spec_name == "multi_tool_sequencing":
        # Should produce multiple tool calls
        if "tools" not in predicted:
            return False
        expected_tools = expected.get("tools", [])
        predicted_tools = predicted.get("tools", [])
        if len(predicted_tools) != len(expected_tools):
            return False
        return all(
            p.get("tool") == e.get("tool")
            for p, e in zip(predicted_tools, expected_tools)
        )

    elif spec_name == "argument_accuracy":
        # Tool name + arguments must match
        if predicted.get("tool") != expected.get("tool"):
            return False
        return predicted.get("arguments") == expected.get("arguments")

    else:
        # Generic: tool name match
        return predicted.get("tool") == expected.get("tool")
