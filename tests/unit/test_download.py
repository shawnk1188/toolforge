"""
Tests for Glaive dataset converter functions.

WHY THESE TESTS MATTER:
  The converter functions are the bridge between raw HuggingFace data and
  our canonical ToolCallingExample format. If parsing is wrong, every
  downstream pipeline (validation, splitting, formatting) produces garbage.
  We test each pure function exhaustively, including malformed inputs.

DESIGN:
  Each test class covers one converter function. Tests follow the pattern:
    1. Valid / happy-path input
    2. Edge cases (empty, missing fields)
    3. Malformed input (bad JSON, unexpected types)
"""

import pytest

from toolforge.data.download import (
    _parse_glaive_tools,
    _parse_glaive_tool_call,
    convert_glaive_example,
)
from toolforge.data.schema import ExampleType, ToolCallingExample


# ============================================================
# Shared fixtures
# ============================================================

VALID_TOOL_JSON = """
[
  {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "The city name"},
        "unit": {"type": "string", "description": "Temperature unit", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["city"]
    }
  }
]
""".strip()

MULTI_TOOL_JSON = """
[
  {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name"}
      },
      "required": ["city"]
    }
  },
  {
    "name": "send_email",
    "description": "Send an email",
    "parameters": {
      "type": "object",
      "properties": {
        "to": {"type": "string", "description": "Recipient"},
        "body": {"type": "string", "description": "Email body"}
      },
      "required": ["to", "body"]
    }
  }
]
""".strip()

SYSTEM_WITH_TOOLS = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required - \n" + VALID_TOOL_JSON
)

SYSTEM_WITH_MULTI_TOOLS = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required - \n" + MULTI_TOOL_JSON
)


# ============================================================
# Test: _parse_glaive_tools
# ============================================================


class TestParseGlaiveTools:
    def test_valid_system_text_with_tools(self):
        """Parse a system prompt containing a JSON array of tool definitions."""
        tools = _parse_glaive_tools(SYSTEM_WITH_TOOLS)
        assert len(tools) == 1
        assert tools[0].name == "get_weather"
        assert tools[0].description == "Get current weather for a city"

    def test_properties_are_parsed(self):
        """Tool parameters and their properties should be parsed correctly."""
        tools = _parse_glaive_tools(SYSTEM_WITH_TOOLS)
        params = tools[0].parameters
        assert "city" in params.properties
        assert params.properties["city"].type == "string"
        assert params.properties["city"].description == "The city name"

    def test_enum_is_parsed(self):
        """Enum values on a parameter should be captured."""
        tools = _parse_glaive_tools(SYSTEM_WITH_TOOLS)
        unit_prop = tools[0].parameters.properties["unit"]
        assert unit_prop.enum == ["celsius", "fahrenheit"]

    def test_required_fields_are_parsed(self):
        """Required parameter list should be captured."""
        tools = _parse_glaive_tools(SYSTEM_WITH_TOOLS)
        assert tools[0].parameters.required == ["city"]

    def test_multiple_tools(self):
        """System prompt with multiple tool definitions."""
        tools = _parse_glaive_tools(SYSTEM_WITH_MULTI_TOOLS)
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"get_weather", "send_email"}

    def test_no_tools_in_system_text(self):
        """System text without any JSON array returns empty list."""
        tools = _parse_glaive_tools("You are a helpful assistant.")
        assert tools == []

    def test_empty_string(self):
        """Empty system text returns empty list."""
        tools = _parse_glaive_tools("")
        assert tools == []

    def test_malformed_json_returns_empty(self):
        """Malformed JSON should not crash; returns empty list."""
        system_text = "You have tools: [{bad json here}]"
        tools = _parse_glaive_tools(system_text)
        assert tools == []

    def test_json_array_with_non_dict_elements(self):
        """Non-dict elements in the JSON array are skipped."""
        system_text = 'Tools: ["not_a_dict", 42, null]'
        tools = _parse_glaive_tools(system_text)
        assert tools == []

    def test_tool_with_no_parameters(self):
        """Tool definition missing the parameters key gets defaults."""
        system_text = '[{"name": "simple_tool", "description": "A simple tool"}]'
        tools = _parse_glaive_tools(system_text)
        assert len(tools) == 1
        assert tools[0].name == "simple_tool"
        assert tools[0].parameters.properties == {}
        assert tools[0].parameters.required == []


# ============================================================
# Test: _parse_glaive_tool_call
# ============================================================


class TestParseGlaiveToolCall:
    def test_valid_functioncall_with_string_arguments(self):
        """Standard functioncall tag with arguments as a JSON-encoded string (Glaive format)."""
        text = '<functioncall> {"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"}'
        calls = _parse_glaive_tool_call(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Tokyo"}

    def test_valid_functioncall_with_flat_arguments(self):
        """Functioncall with flat dict arguments (no nested braces) parses correctly."""
        text = '<functioncall> {"name": "get_weather"}'
        calls = _parse_glaive_tool_call(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {}

    def test_nested_brace_arguments_parse_correctly(self):
        """Nested dict arguments parse correctly with brace-counting parser."""
        text = '<functioncall> {"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        calls = _parse_glaive_tool_call(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Tokyo"}

    def test_multiple_functioncalls(self):
        """Multiple functioncall tags in one response (using string-format arguments)."""
        text = (
            '<functioncall> {"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"} '
            'Some text in between. '
            '<functioncall> {"name": "send_email", "arguments": "{\\"to\\": \\"bob@example.com\\"}"}'
        )
        calls = _parse_glaive_tool_call(text)
        assert len(calls) == 2
        assert calls[0].name == "get_weather"
        assert calls[1].name == "send_email"

    def test_arguments_as_unparseable_string(self):
        """Arguments as an invalid JSON string should fall back to empty dict."""
        text = '<functioncall> {"name": "get_weather", "arguments": "not valid json"}'
        calls = _parse_glaive_tool_call(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {}

    def test_no_functioncall_in_text(self):
        """Text without functioncall tags returns empty list."""
        text = "Sure, I can help you with that. The weather in Tokyo is sunny."
        calls = _parse_glaive_tool_call(text)
        assert calls == []

    def test_empty_string(self):
        """Empty string returns empty list."""
        calls = _parse_glaive_tool_call("")
        assert calls == []

    def test_malformed_json_in_functioncall(self):
        """Malformed JSON after functioncall tag is skipped gracefully."""
        text = '<functioncall> {not valid json}'
        calls = _parse_glaive_tool_call(text)
        assert calls == []

    def test_functioncall_with_no_name(self):
        """A functioncall with an empty name is skipped."""
        text = '<functioncall> {"name": "", "arguments": "{\\"city\\": \\"Tokyo\\"}"}'
        calls = _parse_glaive_tool_call(text)
        assert calls == []

    def test_functioncall_with_no_arguments_key(self):
        """A functioncall missing the arguments key defaults to empty dict."""
        text = '<functioncall> {"name": "get_weather"}'
        calls = _parse_glaive_tool_call(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {}


# ============================================================
# Test: convert_glaive_example
# ============================================================


class TestConvertGlaiveExample:
    def test_valid_single_tool_example(self):
        """Valid row with system tools + USER/ASSISTANT chat returns ToolCallingExample."""
        row = {
            "system": SYSTEM_WITH_TOOLS,
            "chat": (
                "USER: What is the weather in Tokyo? "
                'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"}'
            ),
        }
        result = convert_glaive_example(row, 0)
        assert result is not None
        assert isinstance(result, ToolCallingExample)
        assert result.id == "glaive:0"
        assert result.example_type == ExampleType.SINGLE_TOOL
        assert len(result.expected_tool_calls) == 1
        assert result.expected_tool_calls[0].name == "get_weather"
        assert result.expected_tool_calls[0].arguments == {"city": "Tokyo"}
        assert result.expected_response is None
        assert result.source_dataset == "glaive_function_calling_v2"

    def test_no_tool_example(self):
        """Assistant responds without functioncall -- NO_TOOL example."""
        row = {
            "system": SYSTEM_WITH_TOOLS,
            "chat": (
                "USER: Tell me a joke about the weather. "
                "ASSISTANT: Why did the weather forecaster bring a bar of soap? Because they predicted showers!"
            ),
        }
        result = convert_glaive_example(row, 5)
        assert result is not None
        assert result.example_type == ExampleType.NO_TOOL
        assert result.expected_tool_calls == []
        assert result.expected_response is not None
        assert "showers" in result.expected_response

    def test_multi_tool_example(self):
        """Multiple functioncalls produce a MULTI_TOOL example."""
        row = {
            "system": SYSTEM_WITH_MULTI_TOOLS,
            "chat": (
                "USER: Get the weather in Tokyo and email it to bob@example.com "
                "ASSISTANT: "
                '<functioncall> {"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"} '
                '<functioncall> {"name": "send_email", "arguments": "{\\"to\\": \\"bob@example.com\\", \\"body\\": \\"weather report\\"}"}'
            ),
        }
        result = convert_glaive_example(row, 10)
        assert result is not None
        assert result.example_type == ExampleType.MULTI_TOOL
        assert len(result.expected_tool_calls) == 2

    def test_row_without_parseable_tools_returns_none(self):
        """Row with system text that has no tool definitions returns None."""
        row = {
            "system": "You are a helpful assistant.",
            "chat": "USER: Hello! ASSISTANT: Hi there!",
        }
        result = convert_glaive_example(row, 1)
        assert result is None

    def test_row_without_user_query_returns_none(self):
        """Row with no USER turn in the chat returns None."""
        row = {
            "system": SYSTEM_WITH_TOOLS,
            "chat": "ASSISTANT: Hello, how can I help?",
        }
        result = convert_glaive_example(row, 2)
        assert result is None

    def test_very_short_user_query_returns_none(self):
        """User query shorter than 5 characters is rejected."""
        row = {
            "system": SYSTEM_WITH_TOOLS,
            "chat": "USER: Hi ASSISTANT: Hello!",
        }
        result = convert_glaive_example(row, 3)
        assert result is None

    def test_tool_call_referencing_nonexistent_tool_returns_none(self):
        """Tool call referencing a tool not in the available tools returns None."""
        row = {
            "system": SYSTEM_WITH_TOOLS,  # Only has get_weather
            "chat": (
                "USER: Send an email to bob@example.com about the weather "
                'ASSISTANT: <functioncall> {"name": "send_email", "arguments": "{\\"to\\": \\"bob@example.com\\"}"}'
            ),
        }
        result = convert_glaive_example(row, 4)
        assert result is None

    def test_missing_system_key_returns_none(self):
        """Row with no system key returns None (no parseable tools)."""
        row = {
            "chat": "USER: What is the weather? ASSISTANT: It is sunny.",
        }
        result = convert_glaive_example(row, 6)
        assert result is None

    def test_missing_chat_key_returns_none(self):
        """Row with no chat key returns None (no user query)."""
        row = {
            "system": SYSTEM_WITH_TOOLS,
        }
        result = convert_glaive_example(row, 7)
        assert result is None

    def test_index_is_used_in_id(self):
        """The index parameter appears in the example ID."""
        row = {
            "system": SYSTEM_WITH_TOOLS,
            "chat": (
                "USER: What is the weather in London? "
                'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": "{\\"city\\": \\"London\\"}"}'
            ),
        }
        result = convert_glaive_example(row, 42)
        assert result is not None
        assert result.id == "glaive:42"

    def test_available_tools_are_populated(self):
        """The available_tools field should contain parsed tool definitions."""
        row = {
            "system": SYSTEM_WITH_TOOLS,
            "chat": (
                "USER: What is the weather in Paris? "
                'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}'
            ),
        }
        result = convert_glaive_example(row, 0)
        assert result is not None
        assert len(result.available_tools) == 1
        assert result.available_tools[0].name == "get_weather"

    def test_no_tool_with_empty_assistant_gets_default_response(self):
        """When assistant text is empty and no tool call, a default response is used."""
        row = {
            "system": SYSTEM_WITH_TOOLS,
            "chat": "USER: What is the weather in Tokyo? ASSISTANT: ",
        }
        result = convert_glaive_example(row, 0)
        assert result is not None
        assert result.example_type == ExampleType.NO_TOOL
        assert result.expected_response == "I can help with that."

    def test_partial_valid_tool_calls_kept(self):
        """When some tool calls are valid and some reference non-existent tools, only valid ones are kept."""
        row = {
            "system": SYSTEM_WITH_TOOLS,  # Only has get_weather
            "chat": (
                "USER: Get the weather and send an email about it "
                "ASSISTANT: "
                '<functioncall> {"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"} '
                '<functioncall> {"name": "send_email", "arguments": "{\\"to\\": \\"a@b.com\\"}"}'
            ),
        }
        result = convert_glaive_example(row, 0)
        assert result is not None
        # send_email is filtered out, only get_weather remains
        assert len(result.expected_tool_calls) == 1
        assert result.expected_tool_calls[0].name == "get_weather"
