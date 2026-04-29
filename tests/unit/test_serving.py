"""
Tests for ToolForge serving module — FastAPI endpoint and Gradio demo.

Tests the API request/response models, endpoint logic, and demo construction
WITHOUT loading a real model (uses mocks for the model adapter).

NOTE: FastAPI and Gradio are optional runtime deps. Tests that require them
are marked with pytest.importorskip and will be skipped in environments
where they aren't installed (e.g., behind Websense firewall).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from toolforge.serving.demo import DEMO_TOOLS


# ============================================================
# Demo Module Tests (no external deps needed)
# ============================================================


class TestDemoTools:
    """Tests for the demo tool definitions."""

    def test_demo_tools_count(self):
        assert len(DEMO_TOOLS) == 5

    def test_demo_tools_names(self):
        names = {t["name"] for t in DEMO_TOOLS}
        assert names == {"get_weather", "search_web", "calculate", "send_email", "get_stock_price"}

    def test_demo_tools_have_required_fields(self):
        for tool in DEMO_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "properties" in tool["parameters"]
            assert "required" in tool["parameters"]

    def test_demo_tools_are_valid_json(self):
        """Tools can be serialized/deserialized without loss."""
        roundtrip = json.loads(json.dumps(DEMO_TOOLS))
        assert roundtrip == DEMO_TOOLS

    def test_weather_tool_schema(self):
        weather = next(t for t in DEMO_TOOLS if t["name"] == "get_weather")
        assert "city" in weather["parameters"]["properties"]
        assert "unit" in weather["parameters"]["properties"]
        assert weather["parameters"]["required"] == ["city"]

    def test_calculate_tool_schema(self):
        calc = next(t for t in DEMO_TOOLS if t["name"] == "calculate")
        assert "expression" in calc["parameters"]["properties"]
        assert calc["parameters"]["required"] == ["expression"]

    def test_send_email_requires_all_fields(self):
        email = next(t for t in DEMO_TOOLS if t["name"] == "send_email")
        assert set(email["parameters"]["required"]) == {"to", "subject", "body"}

    def test_stock_tool_schema(self):
        stock = next(t for t in DEMO_TOOLS if t["name"] == "get_stock_price")
        assert stock["parameters"]["required"] == ["ticker"]

    def test_search_tool_schema(self):
        search = next(t for t in DEMO_TOOLS if t["name"] == "search_web")
        assert "query" in search["parameters"]["properties"]
        assert search["parameters"]["required"] == ["query"]


# ============================================================
# CLI Integration Tests (only needs typer)
# ============================================================


class TestCLIServeCommands:
    """Tests for the serve CLI subcommands."""

    def test_serve_start_exists(self):
        """serve start command is registered."""
        from toolforge.cli import serve_app
        command_names = [cmd.name for cmd in serve_app.registered_commands]
        assert "start" in command_names

    def test_serve_demo_exists(self):
        """serve demo command is registered."""
        from toolforge.cli import serve_app
        command_names = [cmd.name for cmd in serve_app.registered_commands]
        assert "demo" in command_names

    def test_serve_start_help(self):
        """serve start --help runs without error."""
        from typer.testing import CliRunner
        from toolforge.cli import app as cli_app

        runner = CliRunner()
        result = runner.invoke(cli_app, ["serve", "start", "--help"])
        assert result.exit_code == 0
        assert "inference server" in result.output.lower()

    def test_serve_demo_help(self):
        """serve demo --help runs without error."""
        from typer.testing import CliRunner
        from toolforge.cli import app as cli_app

        runner = CliRunner()
        result = runner.invoke(cli_app, ["serve", "demo", "--help"])
        assert result.exit_code == 0
        assert "demo" in result.output.lower()

    def test_all_serve_commands(self):
        """Both serve subcommands are registered."""
        from toolforge.cli import serve_app
        names = {cmd.name for cmd in serve_app.registered_commands}
        assert names == {"start", "demo"}


# ============================================================
# FastAPI Tests (require fastapi — skipped if not installed)
# ============================================================


@pytest.fixture()
def _require_fastapi():
    """Skip tests if fastapi is not installed."""
    pytest.importorskip("fastapi", reason="fastapi not installed")


@pytest.mark.usefixtures("_require_fastapi")
class TestPydanticModels:
    """Tests for Pydantic request/response models."""

    def test_tool_definition_basic(self):
        from toolforge.serving.api import ToolDefinition
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert tool.name == "get_weather"

    def test_tool_definition_empty_params(self):
        from toolforge.serving.api import ToolDefinition
        tool = ToolDefinition(name="noop", description="Does nothing")
        assert tool.parameters == {}

    def test_tool_definition_requires_name(self):
        from toolforge.serving.api import ToolDefinition
        with pytest.raises(Exception):
            ToolDefinition(description="Missing name")

    def test_tool_definition_requires_description(self):
        from toolforge.serving.api import ToolDefinition
        with pytest.raises(Exception):
            ToolDefinition(name="no_desc")

    def test_request_defaults(self):
        from toolforge.serving.api import ToolCallRequest, ToolDefinition
        req = ToolCallRequest(
            query="What's the weather?",
            tools=[ToolDefinition(name="weather", description="Get weather")],
        )
        assert req.system_prompt is None
        assert req.max_tokens == 512

    def test_request_custom_options(self):
        from toolforge.serving.api import ToolCallRequest, ToolDefinition
        req = ToolCallRequest(
            query="Search for AI",
            tools=[ToolDefinition(name="search", description="Web search")],
            system_prompt="You are helpful",
            max_tokens=256,
        )
        assert req.system_prompt == "You are helpful"
        assert req.max_tokens == 256

    def test_response_single_tool(self):
        from toolforge.serving.api import ToolCallResponse, ToolCallResult
        resp = ToolCallResponse(
            tool_call=ToolCallResult(name="weather", arguments={"city": "Tokyo"}),
            model="test-model",
            latency_ms=42.5,
        )
        assert resp.tool_call.name == "weather"
        assert resp.response is None

    def test_response_text_only(self):
        from toolforge.serving.api import ToolCallResponse
        resp = ToolCallResponse(
            response="Here's a joke...",
            model="test-model",
            latency_ms=100.0,
        )
        assert resp.response == "Here's a joke..."
        assert resp.tool_call is None

    def test_response_multi_tool(self):
        from toolforge.serving.api import ToolCallResponse, ToolCallResult
        resp = ToolCallResponse(
            tool_calls=[
                ToolCallResult(name="search", arguments={"query": "AI"}),
                ToolCallResult(name="weather", arguments={"city": "NYC"}),
            ],
            model="test-model",
            latency_ms=200.0,
        )
        assert len(resp.tool_calls) == 2

    def test_health_response(self):
        from toolforge.serving.api import HealthResponse
        resp = HealthResponse(status="healthy", model="llama-3.2", model_loaded=True)
        assert resp.status == "healthy"
        assert resp.model_loaded is True


@pytest.mark.usefixtures("_require_fastapi")
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_no_model(self):
        from fastapi.testclient import TestClient
        import toolforge.serving.api as api_mod
        original = api_mod._model_adapter
        api_mod._model_adapter = None
        try:
            client = TestClient(api_mod.app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "not_ready"
            assert resp.json()["model_loaded"] is False
        finally:
            api_mod._model_adapter = original

    def test_health_with_model(self):
        from fastapi.testclient import TestClient
        import toolforge.serving.api as api_mod
        mock_adapter = MagicMock()
        mock_adapter.model_id = "test-model-3b"
        mock_adapter._loaded = True
        original = api_mod._model_adapter
        api_mod._model_adapter = mock_adapter
        try:
            client = TestClient(api_mod.app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["model"] == "test-model-3b"
            assert resp.json()["model_loaded"] is True
        finally:
            api_mod._model_adapter = original


@pytest.mark.usefixtures("_require_fastapi")
class TestToolCallEndpoint:
    """Tests for POST /v1/tool-call."""

    def _setup_mock_adapter(self):
        mock = MagicMock()
        mock.model_id = "test-model"
        mock._loaded = True
        return mock

    def test_no_model_returns_503(self):
        from fastapi.testclient import TestClient
        import toolforge.serving.api as api_mod
        original = api_mod._model_adapter
        api_mod._model_adapter = None
        try:
            client = TestClient(api_mod.app)
            resp = client.post("/v1/tool-call", json={
                "query": "test",
                "tools": [{"name": "t", "description": "d"}],
            })
            assert resp.status_code == 503
        finally:
            api_mod._model_adapter = original

    @patch("toolforge.serving.api.build_eval_prompt", return_value="<prompt>")
    def test_single_tool_call(self, mock_prompt):
        from fastapi.testclient import TestClient
        import toolforge.serving.api as api_mod
        mock_adapter = self._setup_mock_adapter()
        mock_adapter.generate.return_value = '{"tool": "weather"}'
        mock_adapter.parse_output.return_value = {
            "tool": "weather",
            "arguments": {"city": "Tokyo"},
        }
        original = api_mod._model_adapter
        api_mod._model_adapter = mock_adapter
        try:
            client = TestClient(api_mod.app)
            resp = client.post("/v1/tool-call", json={
                "query": "Weather in Tokyo?",
                "tools": [{"name": "weather", "description": "Get weather"}],
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["tool_call"]["name"] == "weather"
            assert data["tool_call"]["arguments"]["city"] == "Tokyo"
            assert data["latency_ms"] > 0
        finally:
            api_mod._model_adapter = original

    @patch("toolforge.serving.api.build_eval_prompt", return_value="<prompt>")
    def test_text_response(self, mock_prompt):
        from fastapi.testclient import TestClient
        import toolforge.serving.api as api_mod
        mock_adapter = self._setup_mock_adapter()
        mock_adapter.generate.return_value = "Here is a joke..."
        mock_adapter.parse_output.return_value = {"response": "Here is a joke..."}
        original = api_mod._model_adapter
        api_mod._model_adapter = mock_adapter
        try:
            client = TestClient(api_mod.app)
            resp = client.post("/v1/tool-call", json={
                "query": "Tell me a joke",
                "tools": [{"name": "weather", "description": "Get weather"}],
            })
            data = resp.json()
            assert data["tool_call"] is None
            assert data["response"] == "Here is a joke..."
        finally:
            api_mod._model_adapter = original

    @patch("toolforge.serving.api.build_eval_prompt", return_value="<prompt>")
    def test_multi_tool_response(self, mock_prompt):
        from fastapi.testclient import TestClient
        import toolforge.serving.api as api_mod
        mock_adapter = self._setup_mock_adapter()
        mock_adapter.generate.return_value = "multi"
        mock_adapter.parse_output.return_value = {
            "tools": [
                {"tool": "search", "arguments": {"query": "AI news"}},
                {"tool": "weather", "arguments": {"city": "SF"}},
            ]
        }
        original = api_mod._model_adapter
        api_mod._model_adapter = mock_adapter
        try:
            client = TestClient(api_mod.app)
            resp = client.post("/v1/tool-call", json={
                "query": "Search AI news and weather in SF",
                "tools": [
                    {"name": "search", "description": "Web search"},
                    {"name": "weather", "description": "Get weather"},
                ],
            })
            data = resp.json()
            assert len(data["tool_calls"]) == 2
            assert data["tool_calls"][0]["name"] == "search"
        finally:
            api_mod._model_adapter = original

    @patch("toolforge.serving.api.build_eval_prompt", return_value="<prompt>")
    def test_raw_output_fallback(self, mock_prompt):
        from fastapi.testclient import TestClient
        import toolforge.serving.api as api_mod
        mock_adapter = self._setup_mock_adapter()
        mock_adapter.generate.return_value = "gibberish output"
        mock_adapter.parse_output.return_value = {}
        original = api_mod._model_adapter
        api_mod._model_adapter = mock_adapter
        try:
            client = TestClient(api_mod.app)
            resp = client.post("/v1/tool-call", json={
                "query": "???",
                "tools": [{"name": "t", "description": "d"}],
            })
            assert resp.json()["response"] == "gibberish output"
        finally:
            api_mod._model_adapter = original

    @patch("toolforge.serving.api.build_eval_prompt", return_value="<prompt>")
    def test_lazy_model_loading(self, mock_prompt):
        from fastapi.testclient import TestClient
        import toolforge.serving.api as api_mod
        mock_adapter = self._setup_mock_adapter()
        mock_adapter._loaded = False
        mock_adapter.generate.return_value = "ok"
        mock_adapter.parse_output.return_value = {"response": "ok"}
        original = api_mod._model_adapter
        api_mod._model_adapter = mock_adapter
        try:
            client = TestClient(api_mod.app)
            client.post("/v1/tool-call", json={
                "query": "test",
                "tools": [{"name": "t", "description": "d"}],
            })
            mock_adapter.load.assert_called_once()
        finally:
            api_mod._model_adapter = original


@pytest.mark.usefixtures("_require_fastapi")
class TestOpenAPISchema:
    """Tests that the API generates a valid OpenAPI schema."""

    def test_openapi_schema_generates(self):
        from toolforge.serving.api import app
        schema = app.openapi()
        assert schema["info"]["title"] == "ToolForge"
        assert "/health" in schema["paths"]
        assert "/v1/tool-call" in schema["paths"]

    def test_tool_call_endpoint_methods(self):
        from toolforge.serving.api import app
        schema = app.openapi()
        assert "post" in schema["paths"]["/v1/tool-call"]
        assert "get" in schema["paths"]["/health"]


# ============================================================
# Gradio Tests (require gradio — skipped if not installed)
# ============================================================


@pytest.fixture()
def _require_gradio():
    pytest.importorskip("gradio", reason="gradio not installed")


@pytest.mark.usefixtures("_require_gradio")
class TestCreateDemo:
    """Tests for the Gradio demo creation (without launching)."""

    @patch("toolforge.serving.demo.create_model_adapter")
    def test_create_demo_returns_blocks(self, mock_create):
        import gradio as gr
        mock_adapter = MagicMock()
        mock_adapter.model_id = "test-model"
        mock_create.return_value = mock_adapter

        from toolforge.serving.demo import create_demo
        demo = create_demo(model_id="test-model", backend="dummy")
        assert isinstance(demo, gr.Blocks)
        mock_adapter.load.assert_called_once()

    @patch("toolforge.serving.demo.create_model_adapter")
    def test_create_demo_with_adapter(self, mock_create):
        import gradio as gr
        mock_adapter = MagicMock()
        mock_adapter.model_id = "test-model"
        mock_create.return_value = mock_adapter

        from toolforge.serving.demo import create_demo
        demo = create_demo(
            model_id="test-model",
            adapter_path="artifacts/sft/adapters",
            backend="dummy",
        )
        assert isinstance(demo, gr.Blocks)
        mock_create.assert_called_once_with(
            backend="dummy",
            model_id="test-model",
            adapter_path="artifacts/sft/adapters",
        )
