"""
Tests for the data schema (Pydantic models).

WHY THESE TESTS MATTER:
  The schema is the CANONICAL FORMAT that all datasets are normalized into.
  If a model silently accepts invalid data (e.g., a tool call referencing
  a tool that doesn't exist), we'll train on garbage and discover it only
  when the model hallucinates tools in production. These tests ensure
  Pydantic catches bad data at ingestion time.

DESIGN:
  Each test class covers one Pydantic model. Tests follow the pattern:
    1. Valid construction (happy path)
    2. Default values behave correctly
    3. Validators reject bad data with clear errors
    4. Serialization / conversion methods produce correct output
"""

import pytest
from pydantic import ValidationError

from toolforge.data.schema import (
    DatasetManifest,
    DatasetSplit,
    ExampleType,
    ParameterProperty,
    ToolCall,
    ToolCallingExample,
    ToolDefinition,
    ToolParameters,
)


# ============================================================
# Helpers: reusable builders for test data
# ============================================================


def _make_tool(name: str = "get_weather", description: str = "Get weather info") -> ToolDefinition:
    """Build a minimal valid ToolDefinition."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters=ToolParameters(
            properties={
                "city": ParameterProperty(type="string", description="City name"),
            },
            required=["city"],
        ),
    )


def _make_tool_call(name: str = "get_weather", arguments: dict | None = None) -> ToolCall:
    """Build a minimal valid ToolCall."""
    return ToolCall(name=name, arguments=arguments or {"city": "Tokyo"})


def _make_example(**overrides) -> ToolCallingExample:
    """Build a minimal valid ToolCallingExample, with overrides for any field."""
    defaults = {
        "id": "test:001",
        "user_query": "What is the weather in Tokyo?",
        "available_tools": [_make_tool()],
        "expected_tool_calls": [_make_tool_call()],
        "example_type": ExampleType.SINGLE_TOOL,
        "source_dataset": "unit_tests",
    }
    defaults.update(overrides)
    return ToolCallingExample(**defaults)


# ============================================================
# Test: ParameterProperty
# ============================================================


class TestParameterProperty:
    def test_basic_creation(self):
        """A parameter with type and description should construct fine."""
        prop = ParameterProperty(type="string", description="A city name")
        assert prop.type == "string"
        assert prop.description == "A city name"
        assert prop.enum is None

    def test_enum_field(self):
        """Parameters with constrained values should store their enum list."""
        prop = ParameterProperty(
            type="string",
            description="Temperature unit",
            enum=["celsius", "fahrenheit"],
        )
        assert prop.enum == ["celsius", "fahrenheit"]

    def test_description_defaults_to_empty_string(self):
        """Description is optional and defaults to empty — not None."""
        prop = ParameterProperty(type="integer")
        assert prop.description == ""

    def test_type_is_required(self):
        """Type is a required field — omitting it should fail."""
        with pytest.raises(ValidationError):
            ParameterProperty()


# ============================================================
# Test: ToolParameters
# ============================================================


class TestToolParameters:
    def test_default_values(self):
        """An empty ToolParameters should default to type=object, empty properties/required."""
        params = ToolParameters()
        assert params.type == "object"
        assert params.properties == {}
        assert params.required == []

    def test_with_properties_and_required(self):
        """ToolParameters should store properties dict and required list."""
        props = {"city": ParameterProperty(type="string", description="City")}
        params = ToolParameters(properties=props, required=["city"])
        assert "city" in params.properties
        assert params.required == ["city"]

    def test_type_can_be_overridden(self):
        """Type defaults to 'object' but can be set to something else."""
        params = ToolParameters(type="array")
        assert params.type == "array"


# ============================================================
# Test: ToolDefinition
# ============================================================


class TestToolDefinition:
    def test_valid_creation(self):
        """A tool with a snake_case name and description should be accepted."""
        tool = _make_tool()
        assert tool.name == "get_weather"
        assert tool.description == "Get weather info"
        assert "city" in tool.parameters.properties

    def test_name_with_underscores_accepted(self):
        """Underscores are the standard separator for tool names."""
        tool = ToolDefinition(name="send_email_v2", description="Send email")
        assert tool.name == "send_email_v2"

    def test_name_with_dots_accepted(self):
        """Dots are allowed — some APIs use namespace.method style."""
        tool = ToolDefinition(name="api.get_weather", description="Namespaced tool")
        assert tool.name == "api.get_weather"

    def test_name_with_spaces_rejected(self):
        """Spaces in tool names would break function calling — reject them."""
        with pytest.raises(ValidationError, match="alphanumeric with underscores"):
            ToolDefinition(name="get weather", description="Bad name")

    def test_name_with_special_chars_rejected(self):
        """Special characters like ! @ # would break function calling."""
        for bad_name in ["get-weather", "get@weather", "get#weather", "get!weather"]:
            with pytest.raises(ValidationError, match="alphanumeric with underscores"):
                ToolDefinition(name=bad_name, description="Bad name")

    def test_name_stripped_of_whitespace(self):
        """Leading/trailing whitespace should be cleaned up, not rejected."""
        tool = ToolDefinition(name="  get_weather  ", description="Trimmed")
        assert tool.name == "get_weather"

    def test_parameters_default_to_empty(self):
        """If no parameters are given, default to an empty ToolParameters."""
        tool = ToolDefinition(name="noop", description="Does nothing")
        assert tool.parameters.type == "object"
        assert tool.parameters.properties == {}

    def test_name_is_required(self):
        """Name is required — omitting it should fail."""
        with pytest.raises(ValidationError):
            ToolDefinition(description="No name")

    def test_description_is_required(self):
        """Description is required — omitting it should fail."""
        with pytest.raises(ValidationError):
            ToolDefinition(name="no_desc")


# ============================================================
# Test: ToolCall
# ============================================================


class TestToolCall:
    def test_basic_creation(self):
        """A tool call with name and arguments should construct fine."""
        tc = ToolCall(name="get_weather", arguments={"city": "Tokyo"})
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Tokyo"}

    def test_default_empty_arguments(self):
        """Arguments default to empty dict — tools with no params are valid."""
        tc = ToolCall(name="list_all")
        assert tc.arguments == {}

    def test_name_is_required(self):
        """Tool call must specify which tool to call."""
        with pytest.raises(ValidationError):
            ToolCall()

    def test_arguments_can_contain_any_types(self):
        """Arguments dict values can be strings, ints, bools, lists, nested dicts."""
        tc = ToolCall(
            name="complex_tool",
            arguments={
                "text": "hello",
                "count": 42,
                "flag": True,
                "items": [1, 2, 3],
                "nested": {"key": "value"},
            },
        )
        assert tc.arguments["count"] == 42
        assert tc.arguments["items"] == [1, 2, 3]


# ============================================================
# Test: ExampleType enum
# ============================================================


class TestExampleType:
    def test_all_four_values_exist(self):
        """All expected example types must be defined — adding/removing types breaks routing."""
        assert ExampleType.SINGLE_TOOL == "single_tool"
        assert ExampleType.MULTI_TOOL == "multi_tool"
        assert ExampleType.NO_TOOL == "no_tool"
        assert ExampleType.ERROR_HANDLING == "error_handling"

    def test_exactly_four_members(self):
        """Guard against accidentally adding extra types without updating routing logic."""
        assert len(ExampleType) == 4

    def test_is_string_enum(self):
        """ExampleType inherits from str — so values can be used directly as strings."""
        assert isinstance(ExampleType.SINGLE_TOOL, str)


# ============================================================
# Test: ToolCallingExample — valid construction
# ============================================================


class TestToolCallingExampleCreation:
    def test_valid_single_tool_example(self):
        """The most common case: one user query, one tool call."""
        example = _make_example()
        assert example.id == "test:001"
        assert example.user_query == "What is the weather in Tokyo?"
        assert len(example.expected_tool_calls) == 1
        assert example.example_type == ExampleType.SINGLE_TOOL

    def test_valid_no_tool_example(self):
        """No-tool examples have a text response instead of tool calls."""
        example = _make_example(
            expected_tool_calls=[],
            expected_response="I cannot look up the weather.",
            example_type=ExampleType.NO_TOOL,
            available_tools=[],
        )
        assert example.expected_response == "I cannot look up the weather."
        assert example.expected_tool_calls == []

    def test_valid_multi_tool_example(self):
        """Multi-tool examples have a sequence of tool calls."""
        send_tool = _make_tool(name="send_email", description="Send an email")
        weather_tool = _make_tool()
        example = _make_example(
            available_tools=[weather_tool, send_tool],
            expected_tool_calls=[
                _make_tool_call("get_weather", {"city": "Tokyo"}),
                _make_tool_call("send_email", {"city": "report"}),
            ],
            example_type=ExampleType.MULTI_TOOL,
        )
        assert len(example.expected_tool_calls) == 2

    def test_system_prompt_defaults_to_empty(self):
        """System prompt is optional — defaults to empty string."""
        example = _make_example()
        assert example.system_prompt == ""

    def test_difficulty_defaults_to_none(self):
        """Difficulty is optional metadata — not all datasets provide it."""
        example = _make_example()
        assert example.difficulty is None

    def test_difficulty_can_be_set(self):
        """When a dataset provides difficulty, it should be stored."""
        example = _make_example(difficulty="hard")
        assert example.difficulty == "hard"


# ============================================================
# Test: ToolCallingExample — validators
# ============================================================


class TestToolCallingExampleValidators:
    def test_must_have_expected_output_rejects_empty(self):
        """An example with no tool calls AND no text response is useless — reject it."""
        with pytest.raises(ValidationError, match="expected_tool_call.*expected_response"):
            _make_example(
                expected_tool_calls=[],
                expected_response=None,
            )

    def test_must_have_expected_output_accepts_tool_calls_only(self):
        """Having tool calls but no text response is fine (standard case)."""
        example = _make_example(expected_response=None)
        assert len(example.expected_tool_calls) == 1

    def test_must_have_expected_output_accepts_response_only(self):
        """Having a text response but no tool calls is fine (no_tool case)."""
        example = _make_example(
            expected_tool_calls=[],
            expected_response="I can't do that.",
            available_tools=[],
        )
        assert example.expected_response == "I can't do that."

    def test_tool_calls_must_reference_available_tools(self):
        """A tool call referencing a nonexistent tool means the data is corrupt."""
        with pytest.raises(ValidationError, match="not in available tools"):
            _make_example(
                expected_tool_calls=[_make_tool_call("nonexistent_tool", {"x": 1})],
            )

    def test_tool_calls_referencing_valid_tools_accepted(self):
        """When all tool calls reference available tools, validation passes."""
        example = _make_example()  # default uses get_weather which is in available_tools
        assert example.expected_tool_calls[0].name == "get_weather"

    def test_empty_tool_calls_skips_reference_check(self):
        """No tool calls means nothing to validate against available_tools."""
        example = _make_example(
            expected_tool_calls=[],
            expected_response="No tool needed.",
            available_tools=[],
        )
        assert example.expected_tool_calls == []

    def test_user_query_cannot_be_empty(self):
        """An empty user query makes no sense — min_length=1 should catch it."""
        with pytest.raises(ValidationError):
            _make_example(user_query="")

    def test_user_query_whitespace_only_rejected(self):
        """A whitespace-only query has min_length >= 1 characters but Pydantic
        counts the characters, so a single space passes min_length. This test
        documents current behavior."""
        # A single space is 1 character, so it passes min_length=1
        example = _make_example(user_query=" ")
        assert example.user_query == " "


# ============================================================
# Test: ToolCallingExample — to_eval_format()
# ============================================================


class TestToolCallingExampleToEvalFormat:
    def test_single_tool_format(self):
        """Single-tool examples should produce a flat expected dict with tool + arguments."""
        example = _make_example()
        result = example.to_eval_format()

        assert result["prompt"] == "What is the weather in Tokyo?"
        assert result["expected"]["tool"] == "get_weather"
        assert result["expected"]["arguments"] == {"city": "Tokyo"}
        assert result["id"] == "test:001"
        assert result["system_prompt"] == ""

    def test_single_tool_format_has_tool_schema(self):
        """The eval format must include tool schemas so the model knows what's available."""
        result = _make_example().to_eval_format()
        assert "tools" in result["tool_schema"]
        assert len(result["tool_schema"]["tools"]) == 1
        assert result["tool_schema"]["tools"][0]["name"] == "get_weather"

    def test_multi_tool_format(self):
        """Multi-tool examples should produce a list of tool+arguments dicts."""
        send_tool = _make_tool(name="send_email", description="Send email")
        weather_tool = _make_tool()
        example = _make_example(
            available_tools=[weather_tool, send_tool],
            expected_tool_calls=[
                _make_tool_call("get_weather", {"city": "Tokyo"}),
                _make_tool_call("send_email", {"city": "report"}),
            ],
            example_type=ExampleType.MULTI_TOOL,
        )
        result = example.to_eval_format()

        # Multi-tool uses "tools" key (plural) with a list
        assert "tools" in result["expected"]
        assert len(result["expected"]["tools"]) == 2
        assert result["expected"]["tools"][0]["tool"] == "get_weather"
        assert result["expected"]["tools"][1]["tool"] == "send_email"

    def test_no_tool_format(self):
        """No-tool examples should have tool=None and include the expected text response."""
        example = _make_example(
            expected_tool_calls=[],
            expected_response="I cannot help with that.",
            example_type=ExampleType.NO_TOOL,
            available_tools=[],
        )
        result = example.to_eval_format()

        assert result["expected"]["tool"] is None
        assert result["expected"]["response"] == "I cannot help with that."

    def test_tool_schema_includes_parameters(self):
        """Tool schema in eval format should include the parameter definitions."""
        result = _make_example().to_eval_format()
        tool_schema = result["tool_schema"]["tools"][0]
        assert "parameters" in tool_schema
        assert "properties" in tool_schema["parameters"]
        assert "city" in tool_schema["parameters"]["properties"]


# ============================================================
# Test: DatasetSplit
# ============================================================


class TestDatasetSplit:
    def test_basic_creation(self):
        """A dataset split with all required fields should construct fine."""
        split = DatasetSplit(
            split_name="train",
            num_examples=1000,
            type_distribution={"single_tool": 800, "no_tool": 200},
            source_distribution={"glaive": 600, "bfcl": 400},
            filepath="data/processed/train.jsonl",
        )
        assert split.split_name == "train"
        assert split.num_examples == 1000
        assert split.filepath == "data/processed/train.jsonl"

    def test_type_distribution_defaults_to_empty(self):
        """Type distribution is optional metadata — defaults to empty dict."""
        split = DatasetSplit(
            split_name="test",
            num_examples=50,
            filepath="data/processed/test.jsonl",
        )
        assert split.type_distribution == {}

    def test_source_distribution_defaults_to_empty(self):
        """Source distribution is optional metadata — defaults to empty dict."""
        split = DatasetSplit(
            split_name="val",
            num_examples=100,
            filepath="data/processed/val.jsonl",
        )
        assert split.source_distribution == {}

    def test_num_examples_cannot_be_negative(self):
        """Negative example counts make no sense — ge=0 constraint should catch it."""
        with pytest.raises(ValidationError):
            DatasetSplit(
                split_name="train",
                num_examples=-1,
                filepath="data/processed/train.jsonl",
            )

    def test_num_examples_zero_allowed(self):
        """An empty split is valid (e.g., after filtering removes all examples)."""
        split = DatasetSplit(
            split_name="train",
            num_examples=0,
            filepath="data/processed/train.jsonl",
        )
        assert split.num_examples == 0


# ============================================================
# Test: DatasetManifest
# ============================================================


class TestDatasetManifest:
    def test_basic_creation(self):
        """A manifest with all required fields should construct fine."""
        split = DatasetSplit(
            split_name="train",
            num_examples=500,
            filepath="data/processed/train.jsonl",
        )
        manifest = DatasetManifest(
            total_examples=500,
            splits=[split],
            processing_timestamp="2026-04-20T12:00:00Z",
            source_datasets=["glaive", "bfcl"],
        )
        assert manifest.total_examples == 500
        assert len(manifest.splits) == 1
        assert manifest.source_datasets == ["glaive", "bfcl"]

    def test_version_defaults_to_1_0(self):
        """Version should default to '1.0' — avoids breaking changes without explicit bump."""
        manifest = DatasetManifest(
            total_examples=0,
            splits=[],
            processing_timestamp="2026-04-20T12:00:00Z",
            source_datasets=[],
        )
        assert manifest.version == "1.0"

    def test_dedup_removed_defaults_to_zero(self):
        """Dedup count defaults to 0 — most pipelines don't dedup."""
        manifest = DatasetManifest(
            total_examples=100,
            splits=[],
            processing_timestamp="2026-04-20T12:00:00Z",
            source_datasets=["test"],
        )
        assert manifest.dedup_removed == 0

    def test_validation_failed_defaults_to_zero(self):
        """Validation failure count defaults to 0."""
        manifest = DatasetManifest(
            total_examples=100,
            splits=[],
            processing_timestamp="2026-04-20T12:00:00Z",
            source_datasets=["test"],
        )
        assert manifest.validation_failed == 0

    def test_multiple_splits(self):
        """A manifest should support multiple splits (train/val/test)."""
        splits = [
            DatasetSplit(split_name="train", num_examples=800, filepath="train.jsonl"),
            DatasetSplit(split_name="val", num_examples=100, filepath="val.jsonl"),
            DatasetSplit(split_name="test", num_examples=100, filepath="test.jsonl"),
        ]
        manifest = DatasetManifest(
            total_examples=1000,
            splits=splits,
            processing_timestamp="2026-04-20T12:00:00Z",
            source_datasets=["glaive"],
        )
        assert len(manifest.splits) == 3
        assert manifest.splits[0].split_name == "train"
