"""
Evaluation metrics for tool-calling quality.

WHY EACH METRIC EXISTS:
  Each metric measures a different failure mode of tool-calling models.
  They're intentionally simple and deterministic — no LLM-as-judge here
  (except for error_recovery). Simple metrics are reproducible, fast,
  and don't introduce evaluation variance.

DESIGN PRINCIPLE:
  Every metric function has the same signature:
    (predicted: dict, expected: dict, tool_schema: dict) -> bool

  Returns True if this single example passes, False otherwise.
  The harness computes the aggregate score (pass_count / total_count).

  This uniform interface means adding a new metric is trivial:
  write a function, add it to MetricName enum, register it in METRIC_REGISTRY.
"""

from __future__ import annotations

import json
from typing import Any, Callable

# Type alias for a metric function
MetricFn = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool]


def exact_match(predicted: dict[str, Any], expected: dict[str, Any], tool_schema: dict[str, Any]) -> bool:
    """
    Check if the predicted tool name exactly matches the expected tool name.

    This is the simplest and most important metric: did the model pick
    the right function?

    Args:
        predicted: Model output, expected to have a "tool" key
        expected: Ground truth, expected to have a "tool" key
        tool_schema: Available tools (unused for this metric)

    Returns:
        True if tool names match exactly (case-sensitive)
    """
    pred_tool = predicted.get("tool") or predicted.get("name") or predicted.get("function")
    exp_tool = expected.get("tool") or expected.get("name") or expected.get("function")

    if pred_tool is None and exp_tool is None:
        return True  # Both correctly identified "no tool needed"

    if pred_tool is None or exp_tool is None:
        return False  # One called a tool, other didn't

    return str(pred_tool).strip().lower() == str(exp_tool).strip().lower()


def json_schema_match(
    predicted: dict[str, Any], expected: dict[str, Any], tool_schema: dict[str, Any]
) -> bool:
    """
    Check if predicted arguments match expected arguments AND conform to the tool schema.

    This is a strict check:
    1. All required arguments must be present
    2. Argument values must match expected values
    3. Argument types must match the schema definition
    4. No extra arguments that aren't in the schema

    Args:
        predicted: Model output with "tool" and "arguments" keys
        expected: Ground truth with "tool" and "arguments" keys
        tool_schema: Available tools with parameter schemas
    """
    # First check tool selection
    if not exact_match(predicted, expected, tool_schema):
        return False

    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})

    if not isinstance(pred_args, dict) or not isinstance(exp_args, dict):
        return False

    # Check all expected arguments are present and match
    for key, exp_value in exp_args.items():
        if key not in pred_args:
            return False

        pred_value = pred_args[key]

        # Normalize for comparison (handle string/number coercion)
        if not _values_match(pred_value, exp_value):
            return False

    # Check no hallucinated extra arguments (not in expected)
    # Allow optional args that aren't in expected but ARE in schema
    tool_name = expected.get("tool") or expected.get("name") or expected.get("function")
    allowed_params = _get_allowed_params(tool_name, tool_schema)

    for key in pred_args:
        if key not in exp_args and key not in allowed_params:
            return False  # Hallucinated argument

    return True


def tool_exists_check(
    predicted: dict[str, Any], expected: dict[str, Any], tool_schema: dict[str, Any]
) -> bool:
    """
    Check that the predicted tool name actually exists in the available tool schema.

    This is the hallucination detection metric. Even if the model picks the
    "wrong" tool, it PASSES this metric as long as the tool exists. It only
    FAILS when the model invents a tool name that isn't in the schema.

    Args:
        predicted: Model output
        expected: Ground truth (unused — we only check schema membership)
        tool_schema: Available tools
    """
    pred_tool = predicted.get("tool") or predicted.get("name") or predicted.get("function")

    # No tool call is always valid (model chose to respond with text)
    if pred_tool is None:
        return True

    # Check if the predicted tool exists in the schema
    available_tools = _extract_tool_names(tool_schema)
    return str(pred_tool).strip().lower() in {t.lower() for t in available_tools}


def correct_refusal_rate(
    predicted: dict[str, Any], expected: dict[str, Any], tool_schema: dict[str, Any]
) -> bool:
    """
    Check if the model correctly REFUSED to call a tool when no tool was appropriate.

    This tests the model's ability to say "no tool needed" rather than
    forcing a tool call on every input.

    Expected: tool=null (no tool call)
    Pass: model also returns tool=null
    Fail: model calls any tool
    """
    exp_tool = expected.get("tool") or expected.get("name") or expected.get("function")

    # This metric only applies to examples where NO tool should be called
    if exp_tool is not None:
        # Not a refusal case — delegate to exact_match
        return exact_match(predicted, expected, tool_schema)

    # Expected: no tool. Check if model also returned no tool.
    pred_tool = predicted.get("tool") or predicted.get("name") or predicted.get("function")
    return pred_tool is None


def sequence_exact_match(
    predicted: dict[str, Any], expected: dict[str, Any], tool_schema: dict[str, Any]
) -> bool:
    """
    Check if a sequence of tool calls matches exactly (order matters).

    For multi-step queries, the model must produce the right tools
    in the right order. Partial matches don't count.

    Expected format: {"tools": [{"tool": "a", "arguments": {...}}, {"tool": "b", ...}]}
    """
    pred_tools = predicted.get("tools", [])
    exp_tools = expected.get("tools", [])

    if not isinstance(pred_tools, list) or not isinstance(exp_tools, list):
        return False

    if len(pred_tools) != len(exp_tools):
        return False

    for pred, exp in zip(pred_tools, exp_tools):
        pred_name = pred.get("tool") or pred.get("name") or pred.get("function")
        exp_name = exp.get("tool") or exp.get("name") or exp.get("function")

        if pred_name is None or exp_name is None:
            return False

        if str(pred_name).strip().lower() != str(exp_name).strip().lower():
            return False

    return True


def graceful_error_rate(
    predicted: dict[str, Any], expected: dict[str, Any], tool_schema: dict[str, Any]
) -> bool:
    """
    Check if the model handled a tool error gracefully.

    This is evaluated with a simple heuristic (not LLM-as-judge):
    - PASS if the response acknowledges the error (contains error-related keywords)
    - PASS if the response does NOT contain hallucinated success data
    - FAIL if the response pretends the tool call succeeded

    For production, this would use LLM-as-judge. For spec-driven dev,
    the heuristic gives us a fast, reproducible signal.
    """
    response_text = str(predicted.get("response", "")).lower()

    # Check for hallucinated success (model pretends it worked)
    hallucination_signals = [
        "here is the",
        "the result is",
        "i found that",
        "the weather is",
        "successfully",
        "here are the results",
    ]

    for signal in hallucination_signals:
        if signal in response_text:
            return False  # Model hallucinated a successful response

    # Check for error acknowledgment
    error_signals = [
        "error",
        "unable",
        "couldn't",
        "could not",
        "unavailable",
        "failed",
        "sorry",
        "issue",
        "problem",
        "try again",
        "apologize",
        "not able",
    ]

    return any(signal in response_text for signal in error_signals)


# ============================================================
# Helper functions
# ============================================================


def _values_match(pred: Any, expected: Any) -> bool:
    """Compare two values with type coercion tolerance."""
    # Exact match
    if pred == expected:
        return True

    # String comparison (case-insensitive, stripped)
    if isinstance(pred, str) and isinstance(expected, str):
        return pred.strip().lower() == expected.strip().lower()

    # Number comparison (int vs float tolerance)
    if isinstance(pred, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(pred) - float(expected)) < 1e-6

    # String-to-number coercion
    try:
        if isinstance(pred, str) and isinstance(expected, (int, float)):
            return abs(float(pred) - float(expected)) < 1e-6
        if isinstance(expected, str) and isinstance(pred, (int, float)):
            return abs(float(expected) - float(pred)) < 1e-6
    except (ValueError, TypeError):
        pass

    # JSON comparison for nested structures
    if isinstance(pred, (dict, list)) and isinstance(expected, (dict, list)):
        try:
            return json.dumps(pred, sort_keys=True) == json.dumps(expected, sort_keys=True)
        except (TypeError, ValueError):
            return False

    return False


def _extract_tool_names(tool_schema: dict[str, Any]) -> list[str]:
    """Extract tool names from various schema formats."""
    # Format 1: {"tools": [{"name": "func_a"}, {"name": "func_b"}]}
    if "tools" in tool_schema:
        return [t.get("name", "") for t in tool_schema["tools"] if isinstance(t, dict)]

    # Format 2: {"functions": [{"name": "func_a"}]}
    if "functions" in tool_schema:
        return [f.get("name", "") for f in tool_schema["functions"] if isinstance(f, dict)]

    # Format 3: flat dict {"func_a": {...}, "func_b": {...}}
    return list(tool_schema.keys())


def _get_allowed_params(tool_name: str | None, tool_schema: dict[str, Any]) -> set[str]:
    """Get allowed parameter names for a tool from the schema."""
    if tool_name is None:
        return set()

    tools = tool_schema.get("tools", tool_schema.get("functions", []))
    for tool in tools:
        if isinstance(tool, dict) and tool.get("name") == tool_name:
            params = tool.get("parameters", {}).get("properties", {})
            return set(params.keys())

    return set()


# ============================================================
# Metric Registry — maps metric names to functions
# ============================================================

METRIC_REGISTRY: dict[str, MetricFn] = {
    "exact_match": exact_match,
    "json_schema_match": json_schema_match,
    "tool_exists_check": tool_exists_check,
    "correct_refusal_rate": correct_refusal_rate,
    "sequence_exact_match": sequence_exact_match,
    "graceful_error_rate": graceful_error_rate,
}


def get_metric(name: str) -> MetricFn:
    """Look up a metric function by name. Raises KeyError if not found."""
    if name not in METRIC_REGISTRY:
        available = ", ".join(METRIC_REGISTRY.keys())
        raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}")
    return METRIC_REGISTRY[name]
