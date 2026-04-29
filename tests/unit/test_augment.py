"""
Tests for data augmentation module.

WHY THESE TESTS MATTER:
  The augmentation module generates synthetic training examples for
  underrepresented example types (error_handling, no_tool, multi_tool).
  Bad augmentation data will teach the model wrong behaviors — these
  tests ensure the generated examples are valid and correctly formatted.
"""

import json
import tempfile
from pathlib import Path

import pytest

from toolforge.data.augment import (
    generate_error_handling_examples,
    generate_multi_tool_reinforcement,
    generate_no_tool_examples,
    run_augmentation,
)
from toolforge.data.schema import (
    ExampleType,
    ParameterProperty,
    ToolCall,
    ToolCallingExample,
    ToolDefinition,
    ToolParameters,
)


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def sample_single_tool_examples():
    """Create sample single_tool examples for augmentation."""
    tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city",
        parameters=ToolParameters(
            type="object",
            properties={
                "city": ParameterProperty(type="string", description="City name"),
                "units": ParameterProperty(type="string", description="Temperature units"),
            },
            required=["city"],
        ),
    )

    examples = []
    for i in range(20):
        ex = ToolCallingExample(
            id=f"test_single:{i}",
            system_prompt="You are a function calling AI model.",
            user_query=f"What's the weather in city_{i}?",
            available_tools=[tool],
            expected_tool_calls=[
                ToolCall(name="get_weather", arguments={"city": f"city_{i}"}),
            ],
            example_type=ExampleType.SINGLE_TOOL,
            source_dataset="test",
        )
        examples.append(ex)
    return examples


@pytest.fixture
def sample_multi_tool_examples():
    """Create sample multi_tool examples for augmentation."""
    tool_a = ToolDefinition(
        name="search_flights",
        description="Search for available flights",
        parameters=ToolParameters(
            type="object",
            properties={
                "from": ParameterProperty(type="string", description="Departure city"),
                "to": ParameterProperty(type="string", description="Arrival city"),
            },
            required=["from", "to"],
        ),
    )
    tool_b = ToolDefinition(
        name="book_hotel",
        description="Book a hotel room",
        parameters=ToolParameters(
            type="object",
            properties={
                "city": ParameterProperty(type="string", description="City"),
                "checkin": ParameterProperty(type="string", description="Check-in date"),
            },
            required=["city"],
        ),
    )

    examples = []
    for i in range(10):
        ex = ToolCallingExample(
            id=f"test_multi:{i}",
            system_prompt="You are a function calling AI model.",
            user_query=f"Book a flight from NYC to LAX and a hotel in LAX for trip {i}",
            available_tools=[tool_a, tool_b],
            expected_tool_calls=[
                ToolCall(name="search_flights", arguments={"from": "NYC", "to": "LAX"}),
                ToolCall(name="book_hotel", arguments={"city": "LAX"}),
            ],
            example_type=ExampleType.MULTI_TOOL,
            source_dataset="test",
        )
        examples.append(ex)
    return examples


# ============================================================
# Tests: Error Handling Generation
# ============================================================


class TestGenerateErrorHandlingExamples:
    """Tests for error_handling example generation."""

    def test_generates_requested_count(self, sample_single_tool_examples):
        result = generate_error_handling_examples(
            sample_single_tool_examples, count=50, seed=42,
        )
        assert len(result) == 50

    def test_all_examples_are_error_handling_type(self, sample_single_tool_examples):
        result = generate_error_handling_examples(
            sample_single_tool_examples, count=10, seed=42,
        )
        for ex in result:
            assert ex.example_type == ExampleType.ERROR_HANDLING

    def test_error_examples_have_no_tool_calls(self, sample_single_tool_examples):
        """Error recovery means responding with text, not calling more tools."""
        result = generate_error_handling_examples(
            sample_single_tool_examples, count=10, seed=42,
        )
        for ex in result:
            assert len(ex.expected_tool_calls) == 0
            assert ex.expected_response is not None

    def test_error_examples_include_error_context(self, sample_single_tool_examples):
        """The user query should include the [System: error] context."""
        result = generate_error_handling_examples(
            sample_single_tool_examples, count=10, seed=42,
        )
        for ex in result:
            assert "[System:" in ex.user_query
            assert "error" in ex.user_query.lower()

    def test_error_responses_acknowledge_the_error(self, sample_single_tool_examples):
        """Expected responses should contain error-acknowledging language."""
        result = generate_error_handling_examples(
            sample_single_tool_examples, count=10, seed=42,
        )
        error_signals = ["apologize", "sorry", "error", "unable", "problem", "issue", "wasn't able"]
        for ex in result:
            resp_lower = ex.expected_response.lower()
            assert any(signal in resp_lower for signal in error_signals), \
                f"Response doesn't acknowledge error: {ex.expected_response[:100]}"

    def test_error_examples_preserve_tool_definitions(self, sample_single_tool_examples):
        """Augmented examples should keep the original tools available."""
        result = generate_error_handling_examples(
            sample_single_tool_examples, count=5, seed=42,
        )
        for ex in result:
            assert len(ex.available_tools) > 0
            assert ex.available_tools[0].name == "get_weather"

    def test_all_examples_pass_pydantic_validation(self, sample_single_tool_examples):
        """Every generated example must pass Pydantic schema validation."""
        result = generate_error_handling_examples(
            sample_single_tool_examples, count=20, seed=42,
        )
        assert len(result) == 20
        for ex in result:
            # Verify round-trip serialization works
            data = json.loads(ex.model_dump_json())
            restored = ToolCallingExample(**data)
            assert restored.example_type == ExampleType.ERROR_HANDLING

    def test_source_dataset_is_augmented(self, sample_single_tool_examples):
        result = generate_error_handling_examples(
            sample_single_tool_examples, count=5, seed=42,
        )
        for ex in result:
            assert ex.source_dataset == "augmented"

    def test_returns_empty_for_no_candidates(self):
        """If no single_tool examples exist, return empty list."""
        result = generate_error_handling_examples([], count=10, seed=42)
        assert result == []


# ============================================================
# Tests: No-Tool Example Generation
# ============================================================


class TestGenerateNoToolExamples:
    """Tests for no_tool example generation."""

    def test_generates_requested_count(self, sample_single_tool_examples):
        result = generate_no_tool_examples(
            sample_single_tool_examples, count=30, seed=42,
        )
        assert len(result) == 30

    def test_all_examples_are_no_tool_type(self, sample_single_tool_examples):
        result = generate_no_tool_examples(
            sample_single_tool_examples, count=10, seed=42,
        )
        for ex in result:
            assert ex.example_type == ExampleType.NO_TOOL

    def test_no_tool_examples_have_no_tool_calls(self, sample_single_tool_examples):
        result = generate_no_tool_examples(
            sample_single_tool_examples, count=10, seed=42,
        )
        for ex in result:
            assert len(ex.expected_tool_calls) == 0
            assert ex.expected_response is not None

    def test_no_tool_examples_have_tools_available(self, sample_single_tool_examples):
        """Even though no tool should be called, tools should be AVAILABLE."""
        result = generate_no_tool_examples(
            sample_single_tool_examples, count=10, seed=42,
        )
        for ex in result:
            assert len(ex.available_tools) > 0

    def test_queries_are_diverse(self, sample_single_tool_examples):
        """Generated queries should be varied, not all identical."""
        result = generate_no_tool_examples(
            sample_single_tool_examples, count=50, seed=42,
        )
        unique_queries = set(ex.user_query for ex in result)
        # At least 60% should be unique
        assert len(unique_queries) >= len(result) * 0.6

    def test_different_seeds_produce_different_data(self, sample_single_tool_examples):
        result_a = generate_no_tool_examples(
            sample_single_tool_examples, count=10, seed=42,
        )
        result_b = generate_no_tool_examples(
            sample_single_tool_examples, count=10, seed=123,
        )
        queries_a = [ex.user_query for ex in result_a]
        queries_b = [ex.user_query for ex in result_b]
        assert queries_a != queries_b


# ============================================================
# Tests: Multi-Tool Reinforcement
# ============================================================


class TestGenerateMultiToolReinforcement:
    """Tests for multi_tool reinforcement generation."""

    def test_generates_requested_count(self, sample_multi_tool_examples):
        result = generate_multi_tool_reinforcement(
            sample_multi_tool_examples, count=20, seed=42,
        )
        assert len(result) == 20

    def test_all_examples_are_multi_tool_type(self, sample_multi_tool_examples):
        result = generate_multi_tool_reinforcement(
            sample_multi_tool_examples, count=10, seed=42,
        )
        for ex in result:
            assert ex.example_type == ExampleType.MULTI_TOOL

    def test_multi_tool_examples_have_multiple_calls(self, sample_multi_tool_examples):
        result = generate_multi_tool_reinforcement(
            sample_multi_tool_examples, count=10, seed=42,
        )
        for ex in result:
            assert len(ex.expected_tool_calls) >= 2

    def test_returns_empty_for_no_multi_examples(self, sample_single_tool_examples):
        """No multi_tool examples → empty result."""
        result = generate_multi_tool_reinforcement(
            sample_single_tool_examples, count=10, seed=42,
        )
        assert result == []


# ============================================================
# Tests: Full Augmentation Pipeline
# ============================================================


class TestRunAugmentation:
    """Tests for the end-to-end augmentation pipeline."""

    def test_augmentation_requires_training_data(self, tmp_path):
        """Should raise if processed training data doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Training data not found"):
            run_augmentation(
                processed_dir=str(tmp_path / "nonexistent"),
                output_dir=str(tmp_path / "output"),
            )

    def test_augmentation_creates_output_files(self, tmp_path, sample_single_tool_examples):
        """Should create augmented training file and metadata file."""
        # Write sample data
        processed = tmp_path / "processed"
        processed.mkdir()
        train_path = processed / "train.jsonl"
        with open(train_path, "w") as f:
            for ex in sample_single_tool_examples:
                f.write(ex.model_dump_json() + "\n")

        output = tmp_path / "output"
        stats = run_augmentation(
            processed_dir=str(processed),
            output_dir=str(output),
            error_count=10,
            no_tool_count=10,
            multi_tool_count=0,  # No multi_tool sources
            seed=42,
        )

        assert (output / "train_augmented.jsonl").exists()
        assert (output / "augmented_only.jsonl").exists()
        assert stats["combined_count"] > len(sample_single_tool_examples)

    def test_augmented_data_is_valid_jsonl(self, tmp_path, sample_single_tool_examples):
        """Every line in augmented output should be valid JSON."""
        processed = tmp_path / "processed"
        processed.mkdir()
        with open(processed / "train.jsonl", "w") as f:
            for ex in sample_single_tool_examples:
                f.write(ex.model_dump_json() + "\n")

        output = tmp_path / "output"
        run_augmentation(
            processed_dir=str(processed),
            output_dir=str(output),
            error_count=5,
            no_tool_count=5,
            multi_tool_count=0,
            seed=42,
        )

        with open(output / "train_augmented.jsonl") as f:
            for line in f:
                data = json.loads(line.strip())
                # Should be valid ToolCallingExample
                ToolCallingExample(**data)
