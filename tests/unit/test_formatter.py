"""
Tests for the Llama 3.2 chat template formatter.

WHY THESE TESTS MATTER:
  If the chat template is wrong, the model learns the wrong stop tokens
  and its generations become garbled. These tests verify the exact format
  that Llama 3.2 expects, including special token placement.
"""

import json

import pytest

from toolforge.data.formatter import (
    BOS,
    EOS,
    EOT,
    HEADER_END,
    HEADER_START,
    _build_system_content,
    _format_assistant_response,
    _format_tool_schema,
    compute_token_stats,
    format_dataset_for_training,
    format_for_inference,
    format_for_training,
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
# Fixtures
# ============================================================


def _make_tool(name: str = "get_weather", desc: str = "Get weather info") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=desc,
        parameters=ToolParameters(
            type="object",
            properties={
                "city": ParameterProperty(type="string", description="The city name"),
            },
            required=["city"],
        ),
    )


def _make_single_tool_example() -> ToolCallingExample:
    """Standard single tool call example."""
    return ToolCallingExample(
        id="test:1",
        system_prompt="You are a weather assistant.",
        user_query="What is the weather in Tokyo?",
        available_tools=[_make_tool()],
        expected_tool_calls=[ToolCall(name="get_weather", arguments={"city": "Tokyo"})],
        example_type=ExampleType.SINGLE_TOOL,
        source_dataset="test",
    )


def _make_no_tool_example() -> ToolCallingExample:
    """Example where no tool should be called."""
    return ToolCallingExample(
        id="test:2",
        system_prompt="You are a weather assistant.",
        user_query="Tell me a joke about weather.",
        available_tools=[_make_tool()],
        expected_tool_calls=[],
        expected_response="Why did the sun go to school? To get brighter!",
        example_type=ExampleType.NO_TOOL,
        source_dataset="test",
    )


def _make_multi_tool_example() -> ToolCallingExample:
    """Multi-tool call example."""
    return ToolCallingExample(
        id="test:3",
        user_query="Get the weather in Tokyo and London.",
        available_tools=[_make_tool()],
        expected_tool_calls=[
            ToolCall(name="get_weather", arguments={"city": "Tokyo"}),
            ToolCall(name="get_weather", arguments={"city": "London"}),
        ],
        example_type=ExampleType.MULTI_TOOL,
        source_dataset="test",
    )


# ============================================================
# Test: Special Token Constants
# ============================================================


class TestSpecialTokens:
    def test_bos_token(self):
        assert BOS == "<|begin_of_text|>"

    def test_eos_token(self):
        assert EOS == "<|end_of_text|>"

    def test_eot_token(self):
        assert EOT == "<|eot_id|>"

    def test_header_tokens(self):
        assert HEADER_START == "<|start_header_id|>"
        assert HEADER_END == "<|end_header_id|>"


# ============================================================
# Test: Tool Schema Formatting
# ============================================================


class TestFormatToolSchema:
    def test_produces_valid_json(self):
        """Tool schema should be valid JSON with type=function wrapper."""
        example = _make_single_tool_example()
        result = _format_tool_schema(example)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["type"] == "function"
        assert parsed[0]["function"]["name"] == "get_weather"

    def test_includes_parameters(self):
        """Tool parameters should be included in the schema."""
        example = _make_single_tool_example()
        result = _format_tool_schema(example)
        parsed = json.loads(result)
        params = parsed[0]["function"]["parameters"]
        assert "city" in params["properties"]
        assert params["required"] == ["city"]


# ============================================================
# Test: System Content Building
# ============================================================


class TestBuildSystemContent:
    def test_includes_custom_system_prompt(self):
        """Custom system prompt from the example should be included."""
        example = _make_single_tool_example()
        content = _build_system_content(example)
        assert "You are a weather assistant." in content

    def test_includes_tool_definitions(self):
        """Tool definitions should appear in system content."""
        example = _make_single_tool_example()
        content = _build_system_content(example)
        assert "get_weather" in content
        assert "Get weather info" in content

    def test_includes_format_instructions(self):
        """Format instructions for JSON tool calls should be present."""
        example = _make_single_tool_example()
        content = _build_system_content(example)
        assert '{"name":' in content or '"name"' in content

    def test_default_system_prompt_when_empty(self):
        """When example has no system prompt, a default is used."""
        example = _make_multi_tool_example()  # Has empty system_prompt
        content = _build_system_content(example)
        assert "helpful assistant" in content


# ============================================================
# Test: Assistant Response Formatting
# ============================================================


class TestFormatAssistantResponse:
    def test_single_tool_produces_json(self):
        """Single tool call should produce a JSON object."""
        example = _make_single_tool_example()
        response = _format_assistant_response(example)
        parsed = json.loads(response)
        assert parsed["name"] == "get_weather"
        assert parsed["arguments"] == {"city": "Tokyo"}

    def test_multi_tool_produces_json_array(self):
        """Multi tool calls should produce a JSON array."""
        example = _make_multi_tool_example()
        response = _format_assistant_response(example)
        parsed = json.loads(response)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "get_weather"
        assert parsed[1]["arguments"] == {"city": "London"}

    def test_no_tool_produces_text(self):
        """No-tool example should produce plain text response."""
        example = _make_no_tool_example()
        response = _format_assistant_response(example)
        assert "brighter" in response
        # Should NOT be JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(response)

    def test_no_tool_example_uses_expected_response(self):
        """No-tool example uses the expected_response field directly."""
        example = _make_no_tool_example()
        response = _format_assistant_response(example)
        assert response == "Why did the sun go to school? To get brighter!"


# ============================================================
# Test: Full Training Format
# ============================================================


class TestFormatForTraining:
    def test_starts_with_bos(self):
        """Training format must start with BOS token."""
        example = _make_single_tool_example()
        result = format_for_training(example)
        assert result.startswith(BOS)

    def test_ends_with_eos(self):
        """Training format must end with EOS token."""
        example = _make_single_tool_example()
        result = format_for_training(example)
        assert result.endswith(EOS)

    def test_contains_system_header(self):
        """Must contain system header tokens."""
        example = _make_single_tool_example()
        result = format_for_training(example)
        assert f"{HEADER_START}system{HEADER_END}" in result

    def test_contains_user_header(self):
        """Must contain user header tokens."""
        example = _make_single_tool_example()
        result = format_for_training(example)
        assert f"{HEADER_START}user{HEADER_END}" in result

    def test_contains_assistant_header(self):
        """Must contain assistant header tokens."""
        example = _make_single_tool_example()
        result = format_for_training(example)
        assert f"{HEADER_START}assistant{HEADER_END}" in result

    def test_contains_user_query(self):
        """User's actual query must appear in the formatted output."""
        example = _make_single_tool_example()
        result = format_for_training(example)
        assert "What is the weather in Tokyo?" in result

    def test_contains_tool_call_json(self):
        """For tool-calling examples, the assistant section must contain tool call JSON."""
        example = _make_single_tool_example()
        result = format_for_training(example)
        assert '"get_weather"' in result
        assert '"Tokyo"' in result

    def test_eot_after_each_turn(self):
        """EOT token should appear after each turn (system, user, assistant)."""
        example = _make_single_tool_example()
        result = format_for_training(example)
        assert result.count(EOT) == 3  # system + user + assistant

    def test_no_tool_example_has_text_response(self):
        """No-tool example should have text (not JSON) in assistant section."""
        example = _make_no_tool_example()
        result = format_for_training(example)
        assert "brighter" in result


# ============================================================
# Test: Inference Format
# ============================================================


class TestFormatForInference:
    def test_starts_with_bos(self):
        """Inference format must start with BOS token."""
        example = _make_single_tool_example()
        result = format_for_inference(example)
        assert result.startswith(BOS)

    def test_does_not_end_with_eos(self):
        """Inference format should NOT end with EOS — model generates from here."""
        example = _make_single_tool_example()
        result = format_for_inference(example)
        assert not result.endswith(EOS)

    def test_ends_with_assistant_header(self):
        """Should end after the assistant header (model generates the response)."""
        example = _make_single_tool_example()
        result = format_for_inference(example)
        assert result.endswith(f"{HEADER_START}assistant{HEADER_END}\n\n")

    def test_does_not_contain_tool_call(self):
        """Inference prompt should NOT contain the expected tool call."""
        example = _make_single_tool_example()
        result = format_for_inference(example)
        # The tool name might appear in tool definitions, but not as a JSON call
        # Check that the formatted assistant response is not there
        assert '"arguments"' not in result.split(f"{HEADER_START}assistant{HEADER_END}")[-1]

    def test_has_only_two_eot_tokens(self):
        """Only system and user turns have EOT — assistant turn is incomplete."""
        example = _make_single_tool_example()
        result = format_for_inference(example)
        assert result.count(EOT) == 2  # system + user only


# ============================================================
# Test: Batch Formatting
# ============================================================


class TestFormatDatasetForTraining:
    def test_returns_list_of_dicts(self):
        """Should return list of {text, id} dicts."""
        examples = [_make_single_tool_example(), _make_no_tool_example()]
        result = format_dataset_for_training(examples)
        assert len(result) == 2
        assert all("text" in item for item in result)
        assert all("id" in item for item in result)

    def test_ids_are_preserved(self):
        """Example IDs should be carried through."""
        examples = [_make_single_tool_example()]
        result = format_dataset_for_training(examples)
        assert result[0]["id"] == "test:1"

    def test_text_is_formatted(self):
        """Text field should contain the full chat template."""
        examples = [_make_single_tool_example()]
        result = format_dataset_for_training(examples)
        assert BOS in result[0]["text"]
        assert EOS in result[0]["text"]


# ============================================================
# Test: Token Statistics
# ============================================================


class TestComputeTokenStats:
    def test_basic_stats(self):
        """Should compute correct basic statistics."""
        formatted = [
            {"text": "a" * 350, "id": "1"},   # ~100 tokens
            {"text": "b" * 700, "id": "2"},   # ~200 tokens
        ]
        stats = compute_token_stats(formatted)
        assert stats["num_examples"] == 2
        assert stats["total_chars"] == 1050
        assert stats["avg_chars"] == 525
        assert stats["max_chars"] == 700
        assert stats["min_chars"] == 350

    def test_empty_list(self):
        """Empty input should not crash."""
        stats = compute_token_stats([])
        assert stats["num_examples"] == 0

    def test_over_threshold_counting(self):
        """Should count examples exceeding token thresholds."""
        # 2048 tokens * 3.5 chars/token = 7168 chars
        formatted = [
            {"text": "a" * 8000, "id": "1"},   # Over 2048 tokens
            {"text": "b" * 100, "id": "2"},     # Under
            {"text": "c" * 15000, "id": "3"},   # Over 4096 tokens
        ]
        stats = compute_token_stats(formatted)
        assert stats["over_2048_tokens"] == 2  # 8000 and 15000
        assert stats["over_4096_tokens"] == 1  # only 15000
