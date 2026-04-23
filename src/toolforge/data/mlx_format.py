"""
Convert ToolForge canonical format to MLX training format.

WHY THIS MODULE EXISTS:
  MLX's lora.train expects data in OpenAI chat format:
    {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]}

  The tokenizer's apply_chat_template handles Llama 3.2 special tokens
  automatically — we don't need to manually insert BOS/EOT/header tokens.
  This is actually MORE correct than our custom formatter because it uses
  the official tokenizer template.

  Additionally, MLX ChatDataset supports a "tools" field for tool definitions,
  which makes the system prompt cleaner.

DATA FLOW:
  ToolCallingExample (schema.py) → OpenAI chat messages → MLX ChatDataset
  Our canonical format         → This converter       → mlx-lm training
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from toolforge.data.schema import ExampleType, ToolCallingExample

console = Console()


def example_to_chat_messages(example: ToolCallingExample) -> dict[str, Any]:
    """
    Convert a ToolCallingExample to OpenAI chat format for MLX training.

    WHY OPENAI FORMAT:
      Llama 3.2's tokenizer was trained with this format. The chat template
      knows how to convert these messages into the correct special token
      sequence. Fighting the tokenizer by using a custom template is asking
      for trouble.

    Returns:
      {"messages": [...], "tools": [...]}
    """
    messages = []

    # System message — behavioral instructions (without tool definitions,
    # because MLX ChatDataset passes tools separately to apply_chat_template)
    system_content = example.system_prompt.strip() if example.system_prompt else ""
    if not system_content:
        system_content = (
            "You are a helpful assistant with access to tools. "
            "Use the provided tools when appropriate to answer the user's query. "
            "If no tool is needed, respond directly with text."
        )

    messages.append({"role": "system", "content": system_content})

    # User message
    messages.append({"role": "user", "content": example.user_query})

    # Assistant message — the target the model learns to generate
    assistant_content = _format_assistant_content(example)
    messages.append({"role": "assistant", "content": assistant_content})

    # Tool definitions in OpenAI format
    tools = _format_tools(example)

    result: dict[str, Any] = {"messages": messages}
    if tools:
        result["tools"] = tools

    return result


def _format_assistant_content(example: ToolCallingExample) -> str:
    """
    Format the assistant's expected response.

    Three cases:
    1. Single tool → JSON object with name + arguments
    2. Multi tool → JSON array of tool call objects
    3. No tool → plain text response

    WHY JSON IN CONTENT:
      For the base Llama 3.2 Instruct model, tool calls are expressed
      as JSON in the assistant message content. The model needs to learn
      to output valid JSON when calling tools. Some models use special
      <tool_call> tokens, but Llama 3.2 uses plain JSON.
    """
    if example.example_type == ExampleType.MULTI_TOOL and len(example.expected_tool_calls) > 1:
        calls = [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in example.expected_tool_calls
        ]
        return json.dumps(calls)

    elif example.expected_tool_calls:
        tc = example.expected_tool_calls[0]
        return json.dumps({"name": tc.name, "arguments": tc.arguments})

    else:
        return example.expected_response or "I can help with that."


def _format_tools(example: ToolCallingExample) -> list[dict[str, Any]]:
    """
    Format tool definitions in OpenAI function calling format.

    This matches the format that Llama 3.2's chat template expects
    when tools are passed to apply_chat_template().
    """
    tools = []
    for tool in example.available_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_dump(),
            },
        })
    return tools


def convert_dataset_to_mlx(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """
    Convert a ToolCallingExample JSONL file to MLX chat format JSONL.

    Args:
        input_path: Path to ToolCallingExample JSONL
        output_path: Path to write MLX-format JSONL

    Returns:
        Number of examples converted
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = ToolCallingExample(**json.loads(line))
                chat_data = example_to_chat_messages(example)
                fout.write(json.dumps(chat_data) + "\n")
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    console.print(f"[dim red]  Line {line_num}: {e}[/dim red]")

    if errors:
        console.print(f"[yellow]  {errors} conversion errors[/yellow]")

    return count


def prepare_mlx_training_data(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/mlx",
) -> dict[str, int]:
    """
    Convert all processed splits to MLX training format.

    MLX expects:
      data_dir/
        train.jsonl   — training data
        valid.jsonl   — validation data
        test.jsonl    — test data (optional)

    WHY SEPARATE OUTPUT DIR:
      Our processed/ dir has ToolCallingExample format. MLX needs
      OpenAI chat format. Keeping them separate avoids confusion
      and lets us regenerate either without affecting the other.
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    splits = [
        ("train.jsonl", "train.jsonl"),
        ("val.jsonl", "valid.jsonl"),  # MLX expects "valid", we have "val"
        ("test.jsonl", "test.jsonl"),
    ]

    for src_name, dst_name in splits:
        src_path = processed_dir / src_name
        dst_path = output_dir / dst_name

        if not src_path.exists():
            console.print(f"[yellow]  {src_path} not found — skipping[/yellow]")
            stats[dst_name] = 0
            continue

        console.print(f"  Converting {src_name} → {dst_name}")
        count = convert_dataset_to_mlx(src_path, dst_path)
        stats[dst_name] = count
        console.print(f"    [green]{count} examples[/green]")

    return stats
