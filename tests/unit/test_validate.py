"""
Tests for the data validation pipeline.

WHY THESE TESTS MATTER:
  Validation is the last line of defense before bad data enters the
  training pipeline. If check_query_length silently passes a 3-char
  query, the model trains on garbage. If deduplication breaks, the
  model memorizes instead of generalizing. We test each check
  function individually and the full pipeline end-to-end.

DESIGN:
  Each test class covers one validation function. Tests follow the pattern:
    1. Clear pass case
    2. Clear fail case
    3. Edge cases
  The validate_dataset tests use temporary .jsonl files to exercise
  the full pipeline including parsing, validation, and deduplication.
"""

import json
import tempfile
from pathlib import Path

import pytest

from toolforge.data.schema import (
    ExampleType,
    ToolCall,
    ToolCallingExample,
    ToolDefinition,
    ToolParameters,
)
from toolforge.data.validate import (
    check_argument_types,
    check_no_empty_tool_names,
    check_query_length,
    check_tool_count,
    check_tool_definitions,
    compute_example_hash,
    validate_dataset,
)


# ============================================================
# Helper
# ============================================================


def _make_example(
    id="test:1",
    query="What is the weather?",
    tool_name="get_weather",
    example_type=ExampleType.SINGLE_TOOL,
):
    return ToolCallingExample(
        id=id,
        user_query=query,
        available_tools=[
            ToolDefinition(
                name=tool_name, description="Get weather info", parameters=ToolParameters()
            )
        ],
        expected_tool_calls=[ToolCall(name=tool_name, arguments={"city": "SF"})],
        example_type=example_type,
        source_dataset="test",
    )


# ============================================================
# Test: check_query_length
# ============================================================


class TestCheckQueryLength:
    def test_valid_length(self):
        example = _make_example(query="What is the weather in San Francisco?")
        passed, reason = check_query_length(example)
        assert passed is True
        assert reason == ""

    def test_too_short(self):
        example = _make_example(query="Hi there?")
        passed, reason = check_query_length(example)
        assert passed is False
        assert "too short" in reason.lower()

    def test_too_long(self):
        example = _make_example(query="x" * 2001)
        passed, reason = check_query_length(example)
        assert passed is False
        assert "too long" in reason.lower()

    def test_exact_min_boundary(self):
        example = _make_example(query="a" * 10)
        passed, _ = check_query_length(example)
        assert passed is True

    def test_exact_max_boundary(self):
        example = _make_example(query="a" * 2000)
        passed, _ = check_query_length(example)
        assert passed is True

    def test_one_below_min(self):
        example = _make_example(query="a" * 9)
        passed, _ = check_query_length(example)
        assert passed is False


# ============================================================
# Test: check_tool_definitions
# ============================================================


class TestCheckToolDefinitions:
    def test_valid_tools(self):
        example = _make_example()
        passed, reason = check_tool_definitions(example)
        assert passed is True
        assert reason == ""

    def test_no_tools(self):
        example = _make_example()
        example = ToolCallingExample(
            id="test:2",
            user_query="What is the weather?",
            available_tools=[],
            expected_tool_calls=[],
            expected_response="I cannot help with that.",
            example_type=ExampleType.NO_TOOL,
            source_dataset="test",
        )
        passed, reason = check_tool_definitions(example)
        assert passed is False
        assert "no tool definitions" in reason.lower()

    def test_tool_with_empty_description(self):
        example = ToolCallingExample(
            id="test:3",
            user_query="What is the weather?",
            available_tools=[
                ToolDefinition(name="get_weather", description="", parameters=ToolParameters())
            ],
            expected_tool_calls=[ToolCall(name="get_weather", arguments={"city": "SF"})],
            example_type=ExampleType.SINGLE_TOOL,
            source_dataset="test",
        )
        passed, reason = check_tool_definitions(example)
        assert passed is False
        assert "no meaningful description" in reason.lower()

    def test_tool_with_short_description(self):
        example = ToolCallingExample(
            id="test:4",
            user_query="What is the weather?",
            available_tools=[
                ToolDefinition(name="get_weather", description="abc", parameters=ToolParameters())
            ],
            expected_tool_calls=[ToolCall(name="get_weather", arguments={"city": "SF"})],
            example_type=ExampleType.SINGLE_TOOL,
            source_dataset="test",
        )
        passed, reason = check_tool_definitions(example)
        assert passed is False
        assert "no meaningful description" in reason.lower()


# ============================================================
# Test: check_tool_count
# ============================================================


class TestCheckToolCount:
    def test_valid_count(self):
        example = _make_example()
        passed, reason = check_tool_count(example)
        assert passed is True
        assert reason == ""

    def test_over_max_tools(self):
        tools = [
            ToolDefinition(
                name=f"tool_{i}", description="Does something useful", parameters=ToolParameters()
            )
            for i in range(21)
        ]
        example = ToolCallingExample(
            id="test:5",
            user_query="What is the weather?",
            available_tools=tools,
            expected_tool_calls=[ToolCall(name="tool_0", arguments={})],
            example_type=ExampleType.SINGLE_TOOL,
            source_dataset="test",
        )
        passed, reason = check_tool_count(example)
        assert passed is False
        assert "too many tools" in reason.lower()

    def test_exactly_max_tools(self):
        tools = [
            ToolDefinition(
                name=f"tool_{i}", description="Does something useful", parameters=ToolParameters()
            )
            for i in range(20)
        ]
        example = ToolCallingExample(
            id="test:6",
            user_query="What is the weather?",
            available_tools=tools,
            expected_tool_calls=[ToolCall(name="tool_0", arguments={})],
            example_type=ExampleType.SINGLE_TOOL,
            source_dataset="test",
        )
        passed, reason = check_tool_count(example)
        assert passed is True


# ============================================================
# Test: check_argument_types
# ============================================================


class TestCheckArgumentTypes:
    def test_normal_args(self):
        example = _make_example()
        passed, reason = check_argument_types(example)
        assert passed is True
        assert reason == ""

    def test_deeply_nested_dict_args_fail(self):
        example = ToolCallingExample(
            id="test:7",
            user_query="What is the weather?",
            available_tools=[
                ToolDefinition(
                    name="get_weather", description="Get weather info", parameters=ToolParameters()
                )
            ],
            expected_tool_calls=[
                ToolCall(
                    name="get_weather",
                    arguments={"location": {"city": {"name": "San Francisco"}}},
                )
            ],
            example_type=ExampleType.SINGLE_TOOL,
            source_dataset="test",
        )
        passed, reason = check_argument_types(example)
        assert passed is False
        assert "deeply nested" in reason.lower()

    def test_shallow_dict_args_pass(self):
        example = ToolCallingExample(
            id="test:8",
            user_query="What is the weather?",
            available_tools=[
                ToolDefinition(
                    name="get_weather", description="Get weather info", parameters=ToolParameters()
                )
            ],
            expected_tool_calls=[
                ToolCall(
                    name="get_weather",
                    arguments={"location": {"city": "SF", "state": "CA"}},
                )
            ],
            example_type=ExampleType.SINGLE_TOOL,
            source_dataset="test",
        )
        passed, reason = check_argument_types(example)
        assert passed is True

    def test_list_args_pass(self):
        example = ToolCallingExample(
            id="test:9",
            user_query="What is the weather?",
            available_tools=[
                ToolDefinition(
                    name="get_weather", description="Get weather info", parameters=ToolParameters()
                )
            ],
            expected_tool_calls=[
                ToolCall(
                    name="get_weather",
                    arguments={"cities": ["SF", "LA", "NYC"]},
                )
            ],
            example_type=ExampleType.SINGLE_TOOL,
            source_dataset="test",
        )
        passed, reason = check_argument_types(example)
        assert passed is True


# ============================================================
# Test: check_no_empty_tool_names
# ============================================================


class TestCheckNoEmptyToolNames:
    def test_valid_names(self):
        example = _make_example()
        passed, reason = check_no_empty_tool_names(example)
        assert passed is True
        assert reason == ""

    def test_empty_name_in_available_tools(self):
        """Build with a valid name, then mutate to empty to bypass pydantic validator."""
        example = _make_example()
        example.available_tools[0].name = ""
        passed, reason = check_no_empty_tool_names(example)
        assert passed is False
        assert "empty tool name in available_tools" in reason.lower()

    def test_whitespace_name_in_available_tools(self):
        example = _make_example()
        example.available_tools[0].name = "   "
        passed, reason = check_no_empty_tool_names(example)
        assert passed is False
        assert "empty tool name in available_tools" in reason.lower()

    def test_empty_name_in_expected_tool_calls(self):
        example = _make_example()
        example.expected_tool_calls[0].name = ""
        passed, reason = check_no_empty_tool_names(example)
        assert passed is False
        assert "empty tool name in expected_tool_calls" in reason.lower()


# ============================================================
# Test: compute_example_hash
# ============================================================


class TestComputeExampleHash:
    def test_same_query_and_tools_same_hash(self):
        ex1 = _make_example(id="test:1")
        ex2 = _make_example(id="test:2")
        assert compute_example_hash(ex1) == compute_example_hash(ex2)

    def test_different_query_different_hash(self):
        ex1 = _make_example(query="What is the weather in SF?")
        ex2 = _make_example(query="What is the weather in NYC?")
        assert compute_example_hash(ex1) != compute_example_hash(ex2)

    def test_case_insensitive(self):
        ex1 = _make_example(query="What is the WEATHER?")
        ex2 = _make_example(query="what is the weather?")
        assert compute_example_hash(ex1) == compute_example_hash(ex2)

    def test_different_tool_names_different_hash(self):
        ex1 = _make_example(tool_name="get_weather")
        ex2 = _make_example(tool_name="get_forecast")
        assert compute_example_hash(ex1) != compute_example_hash(ex2)

    def test_same_arguments_same_hash(self):
        """Arguments are not included in the hash, so different args -> same hash."""
        ex1 = _make_example()
        ex2 = _make_example()
        ex2.expected_tool_calls[0].arguments = {"city": "NYC"}
        assert compute_example_hash(ex1) == compute_example_hash(ex2)


# ============================================================
# Test: validate_dataset (end-to-end pipeline)
# ============================================================


def _write_jsonl(path: Path, examples: list[ToolCallingExample | str]):
    """Write examples to a .jsonl file. Strings are written as-is (for parse error tests)."""
    with open(path, "w") as f:
        for ex in examples:
            if isinstance(ex, str):
                f.write(ex + "\n")
            else:
                f.write(ex.model_dump_json() + "\n")


class TestValidateDataset:
    def test_valid_file_returns_correct_count(self, tmp_path):
        ex1 = _make_example(id="test:1", query="What is the weather in SF?")
        ex2 = _make_example(id="test:2", query="What is the weather in NYC?")
        ex3 = _make_example(id="test:3", query="What is the weather in LA?")
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [ex1, ex2, ex3])

        examples, report = validate_dataset(path)
        assert len(examples) == 3
        assert report["total_lines"] == 3
        assert report["parse_errors"] == 0
        assert report["valid_examples"] == 3

    def test_parse_errors_reported_but_continues(self, tmp_path):
        ex1 = _make_example(id="test:1", query="What is the weather in SF?")
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [ex1, "this is not valid json {{{", ex1])

        examples, report = validate_dataset(path, remove_duplicates=False)
        assert report["parse_errors"] == 1
        assert report["total_lines"] == 3
        # Should still have the 2 valid examples
        assert len(examples) == 2

    def test_deduplication_removes_identical_examples(self, tmp_path):
        ex1 = _make_example(id="test:1", query="What is the weather in SF?")
        ex2 = _make_example(id="test:2", query="What is the weather in SF?")
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [ex1, ex2])

        examples, report = validate_dataset(path, remove_duplicates=True)
        assert len(examples) == 1
        assert report["duplicates_removed"] == 1

    def test_invalid_examples_removed_when_remove_invalid(self, tmp_path):
        valid = _make_example(id="test:1", query="What is the weather in SF?")
        # Short query will fail check_query_length
        short = _make_example(id="test:2", query="Hi there?")
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [valid, short])

        examples, report = validate_dataset(path, remove_invalid=True)
        assert len(examples) == 1
        assert report["check_failures"]["check_query_length"] == 1

    def test_invalid_examples_kept_when_remove_invalid_false(self, tmp_path):
        valid = _make_example(id="test:1", query="What is the weather in SF?")
        short = _make_example(id="test:2", query="Hi there?")
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [valid, short])

        examples, report = validate_dataset(path, remove_invalid=False)
        assert len(examples) == 2
        assert report["check_failures"]["check_query_length"] == 1

    def test_empty_lines_skipped(self, tmp_path):
        ex1 = _make_example(id="test:1", query="What is the weather in SF?")
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            f.write(ex1.model_dump_json() + "\n")
            f.write("\n")
            f.write("\n")
            f.write(ex1.model_dump_json() + "\n")

        examples, report = validate_dataset(path, remove_duplicates=False)
        assert report["total_lines"] == 4
        assert report["parse_errors"] == 0
        assert len(examples) == 2

    def test_type_distribution_in_report(self, tmp_path):
        ex1 = _make_example(id="test:1", query="What is the weather in SF?")
        ex2 = _make_example(
            id="test:2", query="What is the weather in NYC?",
        )
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [ex1, ex2])

        _, report = validate_dataset(path)
        assert "single_tool" in report["type_distribution"]
        assert report["type_distribution"]["single_tool"] == 2
