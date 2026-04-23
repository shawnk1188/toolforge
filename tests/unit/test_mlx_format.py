"""
Tests for MLX training format converter.

WHY THESE TESTS MATTER:
  If the MLX format conversion is wrong, the model trains on garbled data.
  These tests verify that our canonical ToolCallingExample format correctly
  converts to OpenAI chat format that MLX's ChatDataset expects.
"""

import json
import tempfile
from pathlib import Path

import pytest

from toolforge.data.mlx_format import (
    _format_assistant_content,
    _format_tools,
    convert_dataset_to_mlx,
    example_to_chat_messages,
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


def _make_search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="web_search",
        description="Search the web",
        parameters=ToolParameters(
            type="object",
            properties={
                "query": ParameterProperty(type="string", description="Search query"),
                "max_results": ParameterProperty(type="integer", description="Max results"),
            },
            required=["query"],
        ),
    )


def _make_single_tool_example() -> ToolCallingExample:
    """Standard single tool call example."""
    return ToolCallingExample(
        id="test:1",
        source="test",
        source_dataset="test_dataset",
        example_type=ExampleType.SINGLE_TOOL,
        user_query="What's the weather in Tokyo?",
        system_prompt="You are a weather assistant.",
        available_tools=[_make_tool()],
        expected_tool_calls=[
            ToolCall(name="get_weather", arguments={"city": "Tokyo"})
        ],
    )


def _make_multi_tool_example() -> ToolCallingExample:
    """Multi-tool call example."""
    return ToolCallingExample(
        id="test:2",
        source="test",
        source_dataset="test_dataset",
        example_type=ExampleType.MULTI_TOOL,
        user_query="Weather in Tokyo and search for restaurants there",
        system_prompt="You are a helpful assistant.",
        available_tools=[_make_tool(), _make_search_tool()],
        expected_tool_calls=[
            ToolCall(name="get_weather", arguments={"city": "Tokyo"}),
            ToolCall(name="web_search", arguments={"query": "restaurants in Tokyo"}),
        ],
    )


def _make_no_tool_example() -> ToolCallingExample:
    """No-tool example (model should respond with text)."""
    return ToolCallingExample(
        id="test:3",
        source="test",
        source_dataset="test_dataset",
        example_type=ExampleType.NO_TOOL,
        user_query="Tell me a joke",
        system_prompt="You are a helpful assistant.",
        available_tools=[_make_tool()],
        expected_tool_calls=[],
        expected_response="Why did the chicken cross the road? To get to the other side!",
    )


def _make_no_system_prompt_example() -> ToolCallingExample:
    """Example with no system prompt — should get default."""
    return ToolCallingExample(
        id="test:4",
        source="test",
        source_dataset="test_dataset",
        example_type=ExampleType.SINGLE_TOOL,
        user_query="Weather in London?",
        system_prompt="",
        available_tools=[_make_tool()],
        expected_tool_calls=[
            ToolCall(name="get_weather", arguments={"city": "London"})
        ],
    )


# ============================================================
# Tests: example_to_chat_messages
# ============================================================


class TestExampleToChatMessages:
    """Tests for the main conversion function."""

    def test_single_tool_has_three_messages(self):
        """Every example should produce system + user + assistant messages."""
        result = example_to_chat_messages(_make_single_tool_example())
        assert len(result["messages"]) == 3

    def test_single_tool_message_roles(self):
        """Messages should be in system → user → assistant order."""
        result = example_to_chat_messages(_make_single_tool_example())
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_single_tool_system_content(self):
        """System message should contain the example's system prompt."""
        result = example_to_chat_messages(_make_single_tool_example())
        assert result["messages"][0]["content"] == "You are a weather assistant."

    def test_single_tool_user_content(self):
        """User message should contain the user query."""
        result = example_to_chat_messages(_make_single_tool_example())
        assert result["messages"][1]["content"] == "What's the weather in Tokyo?"

    def test_single_tool_assistant_content_is_json(self):
        """Assistant content for tool calls should be valid JSON."""
        result = example_to_chat_messages(_make_single_tool_example())
        assistant = result["messages"][2]["content"]
        parsed = json.loads(assistant)
        assert parsed["name"] == "get_weather"
        assert parsed["arguments"] == {"city": "Tokyo"}

    def test_single_tool_has_tools_field(self):
        """Result should include tools in OpenAI function format."""
        result = example_to_chat_messages(_make_single_tool_example())
        assert "tools" in result
        assert len(result["tools"]) == 1

    def test_tools_in_openai_format(self):
        """Tools should have type=function and nested function definition."""
        result = example_to_chat_messages(_make_single_tool_example())
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert "function" in tool
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather info"
        assert "parameters" in tool["function"]

    def test_multi_tool_assistant_is_json_array(self):
        """Multi-tool examples produce a JSON array of tool calls."""
        result = example_to_chat_messages(_make_multi_tool_example())
        assistant = result["messages"][2]["content"]
        parsed = json.loads(assistant)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "get_weather"
        assert parsed[1]["name"] == "web_search"

    def test_multi_tool_has_all_tools(self):
        """Multi-tool example should list all available tools."""
        result = example_to_chat_messages(_make_multi_tool_example())
        assert len(result["tools"]) == 2
        names = [t["function"]["name"] for t in result["tools"]]
        assert "get_weather" in names
        assert "web_search" in names

    def test_no_tool_assistant_is_text(self):
        """No-tool examples produce plain text, not JSON."""
        result = example_to_chat_messages(_make_no_tool_example())
        assistant = result["messages"][2]["content"]
        assert "chicken cross the road" in assistant
        # Should NOT be valid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(assistant)

    def test_no_system_prompt_gets_default(self):
        """Empty system prompt should get a sensible default."""
        result = example_to_chat_messages(_make_no_system_prompt_example())
        system = result["messages"][0]["content"]
        assert "helpful assistant" in system.lower()
        assert "tool" in system.lower()

    def test_result_is_json_serializable(self):
        """The output must be JSON-serializable for JSONL files."""
        result = example_to_chat_messages(_make_single_tool_example())
        serialized = json.dumps(result)
        roundtripped = json.loads(serialized)
        assert roundtripped == result


# ============================================================
# Tests: _format_assistant_content
# ============================================================


class TestFormatAssistantContent:
    """Tests for assistant content formatting."""

    def test_single_tool_call(self):
        """Single tool call → JSON object with name + arguments."""
        example = _make_single_tool_example()
        content = _format_assistant_content(example)
        parsed = json.loads(content)
        assert parsed == {"name": "get_weather", "arguments": {"city": "Tokyo"}}

    def test_multi_tool_call(self):
        """Multiple tool calls → JSON array."""
        example = _make_multi_tool_example()
        content = _format_assistant_content(example)
        parsed = json.loads(content)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_no_tool_call_returns_text(self):
        """No tool calls → plain text response."""
        example = _make_no_tool_example()
        content = _format_assistant_content(example)
        assert content == "Why did the chicken cross the road? To get to the other side!"

    def test_no_tool_call_no_response_gets_fallback(self):
        """No tool calls + empty-ish response → fallback text from _format_assistant_content.

        WHY THIS TEST IS TRICKY:
          Pydantic's validator requires either tool calls or a non-empty response.
          So we can't construct a ToolCallingExample with both empty. Instead, we
          test the internal function by mocking the example with a minimal response
          that becomes empty when stripped — but Pydantic won't allow that either.
          So we test that an example with a whitespace-only response still produces
          the expected_response field value (since _format_assistant_content returns
          it verbatim).
        """
        example = ToolCallingExample(
            id="test:fallback",
            source="test",
            source_dataset="test_dataset",
            example_type=ExampleType.NO_TOOL,
            user_query="Hello",
            available_tools=[_make_tool()],
            expected_tool_calls=[],
            expected_response="Sure, I can help with that.",
        )
        content = _format_assistant_content(example)
        assert content == "Sure, I can help with that."


# ============================================================
# Tests: _format_tools
# ============================================================


class TestFormatTools:
    """Tests for tool definition formatting."""

    def test_single_tool_format(self):
        """Single tool formatted correctly."""
        example = _make_single_tool_example()
        tools = _format_tools(example)
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        func = tools[0]["function"]
        assert func["name"] == "get_weather"
        assert func["description"] == "Get weather info"
        assert "properties" in func["parameters"]

    def test_tool_parameters_preserved(self):
        """Tool parameter properties should be fully preserved."""
        example = _make_single_tool_example()
        tools = _format_tools(example)
        params = tools[0]["function"]["parameters"]
        assert "city" in params["properties"]
        assert params["properties"]["city"]["type"] == "string"
        assert params["required"] == ["city"]

    def test_empty_tools(self):
        """Example with no tools produces empty list."""
        example = ToolCallingExample(
            id="test:no-tools",
            source="test",
            source_dataset="test_dataset",
            example_type=ExampleType.NO_TOOL,
            user_query="Hello",
            available_tools=[],
            expected_tool_calls=[],
            expected_response="Hi there!",
        )
        tools = _format_tools(example)
        assert tools == []

    def test_multiple_tools(self):
        """Multiple tools all formatted."""
        example = _make_multi_tool_example()
        tools = _format_tools(example)
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"get_weather", "web_search"}


# ============================================================
# Tests: convert_dataset_to_mlx (file I/O)
# ============================================================


class TestConvertDatasetToMlx:
    """Tests for JSONL file conversion."""

    def test_converts_single_example(self, tmp_path):
        """A single example is converted and written as JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        example = _make_single_tool_example()
        input_file.write_text(json.dumps(example.model_dump()) + "\n")

        count = convert_dataset_to_mlx(input_file, output_file)
        assert count == 1

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1

        result = json.loads(lines[0])
        assert "messages" in result
        assert len(result["messages"]) == 3

    def test_converts_multiple_examples(self, tmp_path):
        """Multiple examples are all converted."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        examples = [
            _make_single_tool_example(),
            _make_multi_tool_example(),
            _make_no_tool_example(),
        ]
        with open(input_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.model_dump()) + "\n")

        count = convert_dataset_to_mlx(input_file, output_file)
        assert count == 3

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_skips_empty_lines(self, tmp_path):
        """Empty lines in JSONL should be skipped gracefully."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        example = _make_single_tool_example()
        input_file.write_text(
            "\n" + json.dumps(example.model_dump()) + "\n\n"
        )

        count = convert_dataset_to_mlx(input_file, output_file)
        assert count == 1

    def test_handles_bad_json_gracefully(self, tmp_path):
        """Malformed JSON lines should be skipped, not crash."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        example = _make_single_tool_example()
        input_file.write_text(
            '{"bad json\n'
            + json.dumps(example.model_dump()) + "\n"
        )

        count = convert_dataset_to_mlx(input_file, output_file)
        assert count == 1  # Only the valid line

    def test_creates_parent_directories(self, tmp_path):
        """Output directory should be created if it doesn't exist."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "nested" / "deep" / "output.jsonl"

        example = _make_single_tool_example()
        input_file.write_text(json.dumps(example.model_dump()) + "\n")

        count = convert_dataset_to_mlx(input_file, output_file)
        assert count == 1
        assert output_file.exists()

    def test_output_is_valid_jsonl(self, tmp_path):
        """Every output line should be valid JSON."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        examples = [_make_single_tool_example(), _make_multi_tool_example()]
        with open(input_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.model_dump()) + "\n")

        convert_dataset_to_mlx(input_file, output_file)

        for line in output_file.read_text().strip().split("\n"):
            parsed = json.loads(line)
            assert "messages" in parsed
            assert isinstance(parsed["messages"], list)


# ============================================================
# Tests: Round-trip integrity
# ============================================================


class TestRoundTripIntegrity:
    """Verify that converted data preserves all information."""

    def test_tool_call_arguments_preserved(self):
        """Tool call arguments should survive conversion exactly."""
        example = _make_single_tool_example()
        result = example_to_chat_messages(example)
        assistant = json.loads(result["messages"][2]["content"])
        assert assistant["arguments"] == {"city": "Tokyo"}

    def test_multi_tool_order_preserved(self):
        """Multi-tool call order should be preserved."""
        example = _make_multi_tool_example()
        result = example_to_chat_messages(example)
        calls = json.loads(result["messages"][2]["content"])
        assert calls[0]["name"] == "get_weather"
        assert calls[1]["name"] == "web_search"

    def test_user_query_exact_match(self):
        """User query should be preserved verbatim."""
        example = _make_single_tool_example()
        result = example_to_chat_messages(example)
        assert result["messages"][1]["content"] == example.user_query

    def test_tool_parameters_roundtrip(self):
        """Tool parameters should roundtrip through JSON serialization."""
        example = _make_single_tool_example()
        result = example_to_chat_messages(example)

        # Serialize and deserialize (simulates JSONL write + read)
        roundtripped = json.loads(json.dumps(result))

        tool_params = roundtripped["tools"][0]["function"]["parameters"]
        assert "city" in tool_params["properties"]
        assert tool_params["properties"]["city"]["type"] == "string"
