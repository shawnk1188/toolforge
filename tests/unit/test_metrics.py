"""
Tests for evaluation metrics.

WHY THESE TESTS MATTER:
  Metrics are the foundation of spec-driven development. If exact_match
  has a bug (e.g., it's case-sensitive when it shouldn't be), every
  spec that uses it will report incorrect scores. We test each metric
  exhaustively, including edge cases.

DESIGN:
  Each test class covers one metric function. Tests follow the pattern:
    1. Clear pass case
    2. Clear fail case
    3. Edge cases (null values, type mismatches, empty inputs)
"""

import pytest

from toolforge.eval.metrics import (
    correct_refusal_rate,
    exact_match,
    graceful_error_rate,
    json_schema_match,
    sequence_exact_match,
    tool_exists_check,
    get_metric,
)


SAMPLE_SCHEMA = {
    "tools": [
        {
            "name": "get_weather",
            "parameters": {
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                }
            },
        },
        {
            "name": "send_email",
            "parameters": {
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                }
            },
        },
    ]
}


# ============================================================
# Test: exact_match metric
# ============================================================


class TestExactMatch:
    def test_matching_tool_names(self):
        predicted = {"tool": "get_weather"}
        expected = {"tool": "get_weather"}
        assert exact_match(predicted, expected, SAMPLE_SCHEMA) is True

    def test_different_tool_names(self):
        predicted = {"tool": "send_email"}
        expected = {"tool": "get_weather"}
        assert exact_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_case_insensitive(self):
        """Tool names should match case-insensitively."""
        predicted = {"tool": "Get_Weather"}
        expected = {"tool": "get_weather"}
        assert exact_match(predicted, expected, SAMPLE_SCHEMA) is True

    def test_both_null_is_match(self):
        """If both predict no tool, that's a correct match."""
        predicted = {"tool": None}
        expected = {"tool": None}
        assert exact_match(predicted, expected, SAMPLE_SCHEMA) is True

    def test_predicted_null_expected_not(self):
        """Model didn't call a tool but should have."""
        predicted = {"tool": None}
        expected = {"tool": "get_weather"}
        assert exact_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_predicted_not_null_expected_null(self):
        """Model called a tool but shouldn't have."""
        predicted = {"tool": "get_weather"}
        expected = {"tool": None}
        assert exact_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_whitespace_tolerance(self):
        """Leading/trailing whitespace should be stripped."""
        predicted = {"tool": "  get_weather  "}
        expected = {"tool": "get_weather"}
        assert exact_match(predicted, expected, SAMPLE_SCHEMA) is True

    def test_alternative_key_names(self):
        """Support 'function' and 'name' as alternative keys."""
        predicted = {"function": "get_weather"}
        expected = {"name": "get_weather"}
        assert exact_match(predicted, expected, SAMPLE_SCHEMA) is True


# ============================================================
# Test: json_schema_match metric
# ============================================================


class TestJsonSchemaMatch:
    def test_correct_tool_and_args(self):
        predicted = {"tool": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}
        expected = {"tool": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}
        assert json_schema_match(predicted, expected, SAMPLE_SCHEMA) is True

    def test_wrong_tool_fails(self):
        """Even if args are right, wrong tool name fails."""
        predicted = {"tool": "send_email", "arguments": {"city": "Tokyo"}}
        expected = {"tool": "get_weather", "arguments": {"city": "Tokyo"}}
        assert json_schema_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_missing_required_arg_fails(self):
        predicted = {"tool": "get_weather", "arguments": {}}
        expected = {"tool": "get_weather", "arguments": {"city": "Tokyo"}}
        assert json_schema_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_wrong_arg_value_fails(self):
        predicted = {"tool": "get_weather", "arguments": {"city": "London"}}
        expected = {"tool": "get_weather", "arguments": {"city": "Tokyo"}}
        assert json_schema_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_extra_hallucinated_arg_fails(self):
        """Arguments not in the tool schema should fail."""
        predicted = {
            "tool": "get_weather",
            "arguments": {"city": "Tokyo", "fake_param": "hello"},
        }
        expected = {"tool": "get_weather", "arguments": {"city": "Tokyo"}}
        assert json_schema_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_extra_schema_arg_allowed(self):
        """Extra args that ARE in the schema (optional params) are allowed."""
        predicted = {
            "tool": "get_weather",
            "arguments": {"city": "Tokyo", "unit": "fahrenheit"},
        }
        expected = {"tool": "get_weather", "arguments": {"city": "Tokyo"}}
        # "unit" is in the schema — allowed even if not in expected
        assert json_schema_match(predicted, expected, SAMPLE_SCHEMA) is True


# ============================================================
# Test: tool_exists_check metric
# ============================================================


class TestToolExistsCheck:
    def test_existing_tool_passes(self):
        predicted = {"tool": "get_weather"}
        expected = {"tool": "get_weather"}
        assert tool_exists_check(predicted, expected, SAMPLE_SCHEMA) is True

    def test_hallucinated_tool_fails(self):
        """Tool name that doesn't exist in schema."""
        predicted = {"tool": "translate_text"}
        expected = {"tool": None}
        assert tool_exists_check(predicted, expected, SAMPLE_SCHEMA) is False

    def test_no_tool_call_passes(self):
        """Choosing not to call a tool is always valid."""
        predicted = {"tool": None}
        expected = {"tool": None}
        assert tool_exists_check(predicted, expected, SAMPLE_SCHEMA) is True

    def test_wrong_tool_but_exists_passes(self):
        """Wrong tool, but it EXISTS in schema — passes this metric."""
        predicted = {"tool": "send_email"}  # Wrong tool, but exists
        expected = {"tool": "get_weather"}
        # This metric only checks existence, not correctness
        assert tool_exists_check(predicted, expected, SAMPLE_SCHEMA) is True


# ============================================================
# Test: correct_refusal_rate metric
# ============================================================


class TestCorrectRefusalRate:
    def test_correct_refusal(self):
        """Model correctly refuses to call a tool."""
        predicted = {"tool": None, "response": "I can't help with that."}
        expected = {"tool": None}
        assert correct_refusal_rate(predicted, expected, SAMPLE_SCHEMA) is True

    def test_incorrect_tool_call_when_should_refuse(self):
        """Model calls a tool when it shouldn't have."""
        predicted = {"tool": "get_weather", "arguments": {"city": "Tokyo"}}
        expected = {"tool": None}
        assert correct_refusal_rate(predicted, expected, SAMPLE_SCHEMA) is False

    def test_delegates_to_exact_match_for_tool_cases(self):
        """When a tool IS expected, this metric delegates to exact_match."""
        predicted = {"tool": "get_weather"}
        expected = {"tool": "get_weather"}
        assert correct_refusal_rate(predicted, expected, SAMPLE_SCHEMA) is True


# ============================================================
# Test: sequence_exact_match metric
# ============================================================


class TestSequenceExactMatch:
    def test_correct_sequence(self):
        predicted = {"tools": [{"tool": "get_weather"}, {"tool": "send_email"}]}
        expected = {"tools": [{"tool": "get_weather"}, {"tool": "send_email"}]}
        assert sequence_exact_match(predicted, expected, SAMPLE_SCHEMA) is True

    def test_wrong_order_fails(self):
        """Same tools but in wrong order."""
        predicted = {"tools": [{"tool": "send_email"}, {"tool": "get_weather"}]}
        expected = {"tools": [{"tool": "get_weather"}, {"tool": "send_email"}]}
        assert sequence_exact_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_different_length_fails(self):
        predicted = {"tools": [{"tool": "get_weather"}]}
        expected = {"tools": [{"tool": "get_weather"}, {"tool": "send_email"}]}
        assert sequence_exact_match(predicted, expected, SAMPLE_SCHEMA) is False

    def test_empty_sequences_match(self):
        predicted = {"tools": []}
        expected = {"tools": []}
        assert sequence_exact_match(predicted, expected, SAMPLE_SCHEMA) is True


# ============================================================
# Test: graceful_error_rate metric
# ============================================================


class TestGracefulErrorRate:
    def test_acknowledges_error(self):
        predicted = {"response": "I'm sorry, I wasn't able to complete that request."}
        expected = {}
        assert graceful_error_rate(predicted, expected, SAMPLE_SCHEMA) is True

    def test_hallucinated_success_fails(self):
        """Model pretends the tool call succeeded."""
        predicted = {"response": "The weather is 72°F and sunny in Tokyo."}
        expected = {}
        assert graceful_error_rate(predicted, expected, SAMPLE_SCHEMA) is False

    def test_offers_retry(self):
        predicted = {"response": "There was an issue. Would you like me to try again?"}
        expected = {}
        assert graceful_error_rate(predicted, expected, SAMPLE_SCHEMA) is True


# ============================================================
# Test: Metric registry
# ============================================================


class TestMetricRegistry:
    def test_all_metrics_registered(self):
        """Every metric name in MetricName enum should have a registered function."""
        from toolforge.eval.specs import MetricName

        for metric in MetricName:
            fn = get_metric(metric.value)
            assert callable(fn), f"Metric {metric.value} is not callable"

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError, match="Unknown metric"):
            get_metric("nonexistent_metric")
