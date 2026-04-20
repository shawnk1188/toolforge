"""
Chat template formatter for Llama 3.2 tool-calling fine-tuning.

WHY THIS MODULE EXISTS:
  The SFT trainer needs data in the EXACT chat template format that
  Llama 3.2 expects. If we train with the wrong template, the model
  learns the wrong stop tokens and its generations become garbled.

  Llama 3.2 uses a specific chat template with special tokens:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    ...
    <|start_header_id|>user<|end_header_id|>
    ...
    <|start_header_id|>assistant<|end_header_id|>
    ...
    <|eot_id|>

  For tool-calling, the model's output is a JSON tool call wrapped
  in a specific format that the inference engine can parse.

DESIGN:
  format_example() takes a ToolCallingExample and returns the full
  chat-formatted string. This is what gets tokenized and fed to SFT.

  We also support formatting for the EVAL harness (prompt-only, no
  assistant response) so the model can generate its own completion.
"""

from __future__ import annotations

import json
from typing import Any

from toolforge.data.schema import ExampleType, ToolCallingExample


# ============================================================
# Llama 3.2 Special Tokens
# ============================================================

BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"
EOT = "<|eot_id|>"
HEADER_START = "<|start_header_id|>"
HEADER_END = "<|end_header_id|>"

# Llama 3.2 tool-calling uses this format for tool invocations
# Reference: meta-llama/Llama-3.2-3B-Instruct chat template
TOOL_CALL_START = "<|python_tag|>"


# ============================================================
# Tool Schema Formatting
# ============================================================


def _format_tool_schema(example: ToolCallingExample) -> str:
    """
    Format available tools as a JSON block for the system prompt.

    WHY JSON IN SYSTEM PROMPT:
      Llama 3.2's tool-calling template expects tool definitions
      in the system message as a structured block. The model was
      pre-trained to parse this format during its instruction tuning.
    """
    tools_json = []
    for tool in example.available_tools:
        tool_dict = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_dump(),
            },
        }
        tools_json.append(tool_dict)

    return json.dumps(tools_json, indent=2)


# ============================================================
# System Prompt Construction
# ============================================================


def _build_system_content(example: ToolCallingExample) -> str:
    """
    Build the system message content with tool definitions.

    The system prompt has two parts:
    1. Environment/behavioral instructions (from the example)
    2. Available tool definitions (formatted as JSON)

    WHY COMBINE:
      Llama 3.2 expects a single system message. Splitting tools
      into a separate message would deviate from the training format.
    """
    parts = []

    # Base system prompt (behavioral instructions)
    base_prompt = example.system_prompt.strip()
    if base_prompt:
        parts.append(base_prompt)
    else:
        parts.append(
            "You are a helpful assistant with access to tools. "
            "Use the provided tools when appropriate to answer the user's query. "
            "If no tool is needed, respond directly."
        )

    # Tool definitions
    tools_json = _format_tool_schema(example)
    parts.append(
        "\n\nYou have access to the following tools:\n"
        f"{tools_json}\n\n"
        "When you need to call a tool, respond with a JSON object in this format:\n"
        '{"name": "<tool_name>", "arguments": {<arg_name>: <arg_value>, ...}}\n\n'
        "If no tool is needed, respond normally with text."
    )

    return "\n".join(parts)


# ============================================================
# Assistant Response Formatting
# ============================================================


def _format_assistant_response(example: ToolCallingExample) -> str:
    """
    Format the expected assistant response for training.

    Three cases:
    1. Single tool call → JSON object
    2. Multi tool call → JSON array of objects
    3. No tool needed → plain text response
    """
    if example.example_type == ExampleType.MULTI_TOOL and len(example.expected_tool_calls) > 1:
        # Multi-tool: array of tool calls
        calls = [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in example.expected_tool_calls
        ]
        return json.dumps(calls)

    elif example.expected_tool_calls:
        # Single tool call
        tc = example.expected_tool_calls[0]
        return json.dumps({"name": tc.name, "arguments": tc.arguments})

    else:
        # No tool needed — plain text
        return example.expected_response or "I can help with that."


# ============================================================
# Full Chat Template Formatting
# ============================================================


def format_for_training(example: ToolCallingExample) -> str:
    """
    Format a ToolCallingExample into the Llama 3.2 chat template for SFT.

    This produces the COMPLETE sequence including the assistant response,
    which is what the model learns to generate during fine-tuning.

    Output format:
      <|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>

      {system_content}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>

      {user_query}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>

      {assistant_response}<|eot_id|>
      <|end_of_text|>

    WHY THIS EXACT FORMAT:
      The special tokens (eot_id, start_header_id, etc.) are part of
      Llama 3.2's vocabulary and training. Using different delimiters
      would mean the model's attention patterns don't transfer from
      pre-training, defeating the purpose of fine-tuning.
    """
    system_content = _build_system_content(example)
    assistant_response = _format_assistant_response(example)

    return (
        f"{BOS}"
        f"{HEADER_START}system{HEADER_END}\n\n"
        f"{system_content}{EOT}"
        f"{HEADER_START}user{HEADER_END}\n\n"
        f"{example.user_query}{EOT}"
        f"{HEADER_START}assistant{HEADER_END}\n\n"
        f"{assistant_response}{EOT}"
        f"{EOS}"
    )


def format_for_inference(example: ToolCallingExample) -> str:
    """
    Format a ToolCallingExample as a prompt for INFERENCE (no assistant response).

    Used by the eval harness to create prompts that the model completes.
    The assistant header is included but empty — the model generates from there.

    WHY SEPARATE FROM TRAINING FORMAT:
      During eval, we want the model to GENERATE the response, not
      memorize it. So we provide everything up to the assistant turn
      and let the model complete it.
    """
    system_content = _build_system_content(example)

    return (
        f"{BOS}"
        f"{HEADER_START}system{HEADER_END}\n\n"
        f"{system_content}{EOT}"
        f"{HEADER_START}user{HEADER_END}\n\n"
        f"{example.user_query}{EOT}"
        f"{HEADER_START}assistant{HEADER_END}\n\n"
    )


# ============================================================
# Batch Formatting
# ============================================================


def format_dataset_for_training(
    examples: list[ToolCallingExample],
) -> list[dict[str, str]]:
    """
    Format a list of examples into the training format.

    Returns a list of dicts with:
      - "text": the full chat-formatted string (for SFT)
      - "id": the example ID (for tracking)

    WHY RETURN DICTS:
      The SFT trainer (HuggingFace TRL) expects a dataset with a "text"
      column. We also keep the ID for traceability — if a training run
      shows anomalous loss on certain examples, we can trace back.
    """
    formatted = []
    for example in examples:
        formatted.append({
            "text": format_for_training(example),
            "id": example.id,
        })
    return formatted


def compute_token_stats(
    formatted_texts: list[dict[str, str]],
    chars_per_token: float = 3.5,
) -> dict[str, Any]:
    """
    Estimate token statistics for the formatted dataset.

    WHY ESTIMATE:
      Actual tokenization requires loading the model's tokenizer.
      For planning purposes (will it fit in context? how long will
      training take?), a character-based estimate is good enough.
      Llama tokenizers average ~3.5 chars per token for English text.
    """
    lengths = [len(item["text"]) for item in formatted_texts]
    token_lengths = [int(l / chars_per_token) for l in lengths]

    return {
        "num_examples": len(formatted_texts),
        "total_chars": sum(lengths),
        "avg_chars": sum(lengths) / len(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "estimated_total_tokens": sum(token_lengths),
        "estimated_avg_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
        "estimated_max_tokens": max(token_lengths) if token_lengths else 0,
        "over_2048_tokens": sum(1 for t in token_lengths if t > 2048),
        "over_4096_tokens": sum(1 for t in token_lengths if t > 4096),
    }
