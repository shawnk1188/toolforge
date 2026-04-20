"""
Dataset download and normalization pipeline.

WHY THIS MODULE EXISTS:
  Open-source tool-calling datasets come in different formats. This module
  downloads them from HuggingFace Hub and converts each into our canonical
  ToolCallingExample format. The conversion happens at download time so
  that all downstream code (validation, splitting, formatting) only works
  with one format.

SUPPORTED DATASETS:
  1. Glaive Function Calling v2 (glaive-ai/glaive-function-calling-v2)
     - 113K chat-style examples with system/user/assistant turns
     - Good variety of tool types and argument patterns

  2. NousResearch Hermes Function Calling v1
     - Curated tool-calling examples in structured format
     - Higher quality, smaller volume

DESIGN:
  Each dataset has a converter function that takes raw HF rows and yields
  ToolCallingExample objects. The converter is the ONLY place that knows
  the source format — downstream code is format-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Generator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from toolforge.data.schema import (
    ExampleType,
    ParameterProperty,
    ToolCall,
    ToolCallingExample,
    ToolDefinition,
    ToolParameters,
)

console = Console()


# ============================================================
# Glaive Function Calling v2 Converter
# ============================================================


def _parse_glaive_tools(system_text: str) -> list[ToolDefinition]:
    """
    Parse tool definitions from Glaive's system prompt format.

    Glaive embeds tool definitions in the system prompt as JSON-like
    blocks. We extract and parse them into ToolDefinition objects.
    """
    tools = []

    # Try to find JSON array of functions in system prompt
    # Glaive format: "You have access to the following functions: [...]"
    json_match = re.search(r'\[[\s\S]*\]', system_text)
    if json_match:
        try:
            raw_tools = json.loads(json_match.group())
            for raw in raw_tools:
                if not isinstance(raw, dict):
                    continue
                name = raw.get("name", "")
                desc = raw.get("description", "")
                params_raw = raw.get("parameters", {})

                properties = {}
                for pname, pdef in params_raw.get("properties", {}).items():
                    if isinstance(pdef, dict):
                        properties[pname] = ParameterProperty(
                            type=pdef.get("type", "string"),
                            description=pdef.get("description", ""),
                            enum=pdef.get("enum"),
                        )

                tools.append(ToolDefinition(
                    name=name,
                    description=desc,
                    parameters=ToolParameters(
                        type="object",
                        properties=properties,
                        required=params_raw.get("required", []),
                    ),
                ))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return tools


def _parse_glaive_tool_call(assistant_text: str) -> list[ToolCall]:
    """
    Parse tool calls from Glaive's assistant response format.

    Glaive format: <functioncall> {"name": "func", "arguments": {...}}

    WHY NOT SIMPLE REGEX:
      The naive regex `\{.*?\}` stops at the first closing brace,
      which breaks on nested JSON like {"arguments": {"city": "SF"}}.
      Instead, we find each <functioncall> tag and then use a brace-
      counting approach to extract the complete JSON object.
    """
    calls = []

    # Find all <functioncall> tag positions
    tag = "<functioncall>"
    start = 0
    while True:
        pos = assistant_text.find(tag, start)
        if pos == -1:
            break

        # Find the opening brace after the tag
        json_start = assistant_text.find("{", pos + len(tag))
        if json_start == -1:
            start = pos + len(tag)
            continue

        # Count braces to find the matching closing brace
        json_str = _extract_json_object(assistant_text, json_start)
        if json_str is None:
            start = pos + len(tag)
            continue

        try:
            raw = json.loads(json_str)
            name = raw.get("name", "")
            args = raw.get("arguments", {})

            # Sometimes arguments is a JSON string, not a dict
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if name:
                calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))
        except (json.JSONDecodeError, KeyError):
            pass

        start = json_start + len(json_str) if json_str else pos + len(tag)

    return calls


def _extract_json_object(text: str, start: int) -> str | None:
    """
    Extract a complete JSON object from text starting at position `start`.

    Uses brace counting to handle nested objects. Respects strings
    (doesn't count braces inside quoted strings).
    """
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None  # Unbalanced braces


def convert_glaive_example(
    row: dict[str, Any], index: int
) -> ToolCallingExample | None:
    """
    Convert a single Glaive dataset row to our canonical format.

    Returns None if the row can't be cleanly converted (malformed data).
    """
    try:
        system_text = row.get("system", "") or ""
        chat_text = row.get("chat", "") or ""

        # Parse available tools from system prompt
        tools = _parse_glaive_tools(system_text)
        if not tools:
            return None  # Skip examples without parseable tools

        # Extract user query and assistant response from chat
        # Glaive format: "USER: ... ASSISTANT: ..."
        user_match = re.search(r'USER:\s*(.*?)(?=ASSISTANT:|$)', chat_text, re.DOTALL)
        assistant_match = re.search(r'ASSISTANT:\s*(.*?)(?=USER:|FUNCTION RESPONSE:|$)', chat_text, re.DOTALL)

        if not user_match:
            return None

        user_query = user_match.group(1).strip()
        assistant_text = assistant_match.group(1).strip() if assistant_match else ""

        if not user_query or len(user_query) < 5:
            return None  # Skip very short/empty queries

        # Parse tool calls from assistant response
        tool_calls = _parse_glaive_tool_call(assistant_text)

        # Determine example type
        if tool_calls:
            if len(tool_calls) > 1:
                example_type = ExampleType.MULTI_TOOL
            else:
                example_type = ExampleType.SINGLE_TOOL
            expected_response = None
        else:
            example_type = ExampleType.NO_TOOL
            expected_response = assistant_text if assistant_text else "I can help with that."

        # Validate tool calls reference available tools
        available_names = {t.name for t in tools}
        valid_calls = [tc for tc in tool_calls if tc.name in available_names]

        if tool_calls and not valid_calls:
            return None  # All tool calls referenced non-existent tools

        return ToolCallingExample(
            id=f"glaive:{index}",
            system_prompt=system_text,
            user_query=user_query,
            available_tools=tools,
            expected_tool_calls=valid_calls,
            expected_response=expected_response,
            example_type=example_type,
            source_dataset="glaive_function_calling_v2",
        )

    except Exception:
        return None  # Skip any malformed examples


# ============================================================
# NousResearch Hermes Function Calling Converter
# ============================================================


def _parse_hermes_tool_call(gpt_text: str) -> list[ToolCall]:
    """
    Parse tool calls from Hermes <tool_call> format.

    Hermes format:
      <tool_call>
      {"name": "func", "arguments": {"key": "value"}}
      </tool_call>

    WHY SEPARATE PARSER:
      Hermes uses XML-like tags instead of Glaive's <functioncall> format.
      The JSON inside is already well-structured (not string-encoded),
      making parsing much more reliable.
    """
    calls = []
    # Find all <tool_call>...</tool_call> blocks
    for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', gpt_text, re.DOTALL):
        try:
            raw = json.loads(match.group(1))
            name = raw.get("name", "")
            args = raw.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


def _parse_hermes_tools(tools_str: str) -> list[ToolDefinition]:
    """
    Parse tool definitions from Hermes 'tools' column.

    Hermes stores tools as a JSON string: [{"type": "function", "function": {...}}]
    This is already OpenAI-compatible format — we just extract the function details.
    """
    tools = []
    try:
        raw_tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
        if not isinstance(raw_tools, list):
            return tools

        for raw in raw_tools:
            if not isinstance(raw, dict):
                continue
            # Handle {"type": "function", "function": {...}} wrapper
            func_def = raw.get("function", raw)
            name = func_def.get("name", "")
            desc = func_def.get("description", "")
            params_raw = func_def.get("parameters", {})

            properties = {}
            for pname, pdef in params_raw.get("properties", {}).items():
                if isinstance(pdef, dict):
                    properties[pname] = ParameterProperty(
                        type=pdef.get("type", "string"),
                        description=pdef.get("description", ""),
                        enum=pdef.get("enum"),
                    )

            tools.append(ToolDefinition(
                name=name,
                description=desc,
                parameters=ToolParameters(
                    type=params_raw.get("type", "object"),
                    properties=properties,
                    required=params_raw.get("required", []),
                ),
            ))
    except (json.JSONDecodeError, TypeError):
        pass

    return tools


def convert_hermes_example(
    row: dict[str, Any],
    index: int,
    config_name: str = "hermes",
) -> ToolCallingExample | None:
    """
    Convert a NousResearch/Hermes row to canonical ToolCallingExample.

    Hermes format:
      - conversations: list of {from, value} dicts (system/human/gpt/tool turns)
      - tools: JSON string of tool definitions

    WHY USE HERMES:
      When Glaive went behind auth, Hermes became the best open alternative.
      It actually INCLUDES Glaive data (glaive_func_calling config) plus
      higher-quality curated examples (func_calling_singleturn).
    """
    try:
        conversations = row.get("conversations", [])
        tools_str = row.get("tools", "")

        # Parse tools
        tools = _parse_hermes_tools(tools_str)
        if not tools:
            # Try parsing from system message
            for turn in conversations:
                if turn.get("from") == "system":
                    tools = _parse_glaive_tools(turn.get("value", ""))
                    break
        if not tools:
            return None

        # Extract first human query and first gpt response
        user_query = ""
        gpt_response = ""
        system_prompt = ""

        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "").strip()
            if role == "system" and not system_prompt:
                system_prompt = value
            elif role == "human" and not user_query:
                user_query = value
            elif role == "gpt" and not gpt_response and user_query:
                gpt_response = value
                break  # Take first human→gpt pair

        if not user_query or len(user_query) < 5:
            return None

        # Parse tool calls
        tool_calls = _parse_hermes_tool_call(gpt_response)

        # Determine example type
        if tool_calls:
            example_type = ExampleType.MULTI_TOOL if len(tool_calls) > 1 else ExampleType.SINGLE_TOOL
            expected_response = None
        else:
            example_type = ExampleType.NO_TOOL
            # Clean up response (remove any XML tags)
            clean_response = re.sub(r'</?tool_call>', '', gpt_response).strip()
            expected_response = clean_response if clean_response else "I can help with that."

        # Validate tool calls reference available tools
        available_names = {t.name for t in tools}
        valid_calls = [tc for tc in tool_calls if tc.name in available_names]

        if tool_calls and not valid_calls:
            return None

        return ToolCallingExample(
            id=f"{config_name}:{index}",
            system_prompt=system_prompt,
            user_query=user_query,
            available_tools=tools,
            expected_tool_calls=valid_calls,
            expected_response=expected_response,
            example_type=example_type,
            source_dataset=f"hermes_function_calling_{config_name}",
        )

    except Exception:
        return None


# ============================================================
# Download Orchestrator
# ============================================================


def download_and_convert(
    output_dir: str | Path = "data/raw",
    max_examples: int | None = None,
    datasets_to_use: list[str] | None = None,
) -> dict[str, int]:
    """
    Download datasets from HuggingFace and convert to canonical format.

    Args:
        output_dir: Where to save the converted .jsonl files
        max_examples: Limit per dataset (useful for development/testing)
        datasets_to_use: Which datasets to download (default: all)

    Returns:
        Dict of {dataset_name: num_examples_converted}
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_datasets = {
        # Hermes singleturn: highest quality, curated function-calling examples
        "hermes_singleturn": {
            "hf_name": "NousResearch/hermes-function-calling-v1",
            "hf_config": "func_calling_singleturn",
            "converter": lambda row, idx: convert_hermes_example(row, idx, "singleturn"),
            "split": "train",
        },
        # Hermes multiturn: function-calling with follow-up turns
        "hermes_multiturn": {
            "hf_name": "NousResearch/hermes-function-calling-v1",
            "hf_config": "func_calling",
            "converter": lambda row, idx: convert_hermes_example(row, idx, "multiturn"),
            "split": "train",
        },
        # Hermes Glaive: Glaive data reformatted in Hermes structure
        "hermes_glaive": {
            "hf_name": "NousResearch/hermes-function-calling-v1",
            "hf_config": "glaive_func_calling",
            "converter": lambda row, idx: convert_hermes_example(row, idx, "glaive"),
            "split": "train",
        },
    }

    if datasets_to_use is None:
        datasets_to_use = list(available_datasets.keys())

    stats: dict[str, int] = {}

    for ds_key in datasets_to_use:
        if ds_key not in available_datasets:
            console.print(f"[yellow]Unknown dataset: {ds_key} — skipping[/yellow]")
            continue

        ds_config = available_datasets[ds_key]
        console.print(f"\n[bold blue]Downloading:[/bold blue] {ds_config['hf_name']}")

        try:
            # Configure SSL for corporate proxy environments
            # WHY: Corporate proxies (e.g., Netskope, Zscaler) intercept HTTPS
            # with a self-signed root CA. The huggingface_hub library uses httpx
            # which doesn't trust the system keychain by default. We disable
            # SSL verification for the download only.
            import os

            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

            try:
                import httpx as _httpx

                # Monkey-patch httpx.Client to disable SSL verification
                _orig_client_init = _httpx.Client.__init__

                def _patched_client_init(self, *args, **kwargs):
                    kwargs.setdefault("verify", False)
                    _orig_client_init(self, *args, **kwargs)

                _httpx.Client.__init__ = _patched_client_init
            except ImportError:
                pass  # httpx not available — let it try with defaults

            # Download from HuggingFace
            hf_config = ds_config.get("hf_config")
            load_args = {"path": ds_config["hf_name"], "split": ds_config["split"]}
            if hf_config:
                load_args["name"] = hf_config
            hf_dataset = load_dataset(**load_args)

            # Limit examples if specified
            if max_examples:
                hf_dataset = hf_dataset.select(range(min(max_examples, len(hf_dataset))))

            # Convert each row
            output_path = output_dir / f"{ds_key}_converted.jsonl"
            converted = 0
            skipped = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Converting {ds_key}...", total=len(hf_dataset))

                with open(output_path, "w") as f:
                    for idx, row in enumerate(hf_dataset):
                        example = ds_config["converter"](row, idx)
                        if example is not None:
                            f.write(example.model_dump_json() + "\n")
                            converted += 1
                        else:
                            skipped += 1
                        progress.update(task, advance=1)

            stats[ds_key] = converted
            console.print(
                f"  [green]Converted:[/green] {converted} examples "
                f"([dim]skipped {skipped} malformed[/dim])"
            )
            console.print(f"  [dim]Saved to: {output_path}[/dim]")

        except Exception as e:
            console.print(f"  [red]Error downloading {ds_key}: {e}[/red]")
            stats[ds_key] = 0

    return stats


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    """Run with: python -m toolforge.data.download"""
    console.print("\n[bold]ToolForge Data Download Pipeline[/bold]\n")
    stats = download_and_convert(max_examples=5000)  # Start with 5K for dev speed
    console.print(f"\n[bold green]Download complete:[/bold green] {stats}")
