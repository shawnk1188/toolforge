"""
FastAPI inference endpoint for ToolForge.

WHY FASTAPI:
  - Auto-generates OpenAPI docs at /docs
  - Async support for concurrent requests
  - Pydantic integration for request/response validation
  - Industry standard for ML model serving

ENDPOINT DESIGN:
  POST /v1/tool-call
    Input:  { query, tools, system_prompt? }
    Output: { tool_call | response, model, latency_ms }

  This mirrors OpenAI's function calling API format so clients
  can switch between GPT-4o and ToolForge with minimal changes.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="ToolForge",
    description="Spec-driven fine-tuned tool-calling API",
    version="0.1.0",
)

# Global model adapter (loaded on startup)
_model_adapter = None


# ============================================================
# Request / Response Models
# ============================================================


class ToolDefinition(BaseModel):
    """A tool the model can call."""
    name: str = Field(..., description="Function name")
    description: str = Field(..., description="What this tool does")
    parameters: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for parameters")


class ToolCallRequest(BaseModel):
    """Request to invoke tool-calling inference."""
    query: str = Field(..., description="User's natural language query")
    tools: list[ToolDefinition] = Field(..., description="Available tools")
    system_prompt: str | None = Field(None, description="Optional system instructions")
    max_tokens: int = Field(512, description="Maximum tokens to generate")


class ToolCallResult(BaseModel):
    """A single tool call result."""
    name: str = Field(..., description="Tool name to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


class ToolCallResponse(BaseModel):
    """Response from tool-calling inference."""
    tool_call: ToolCallResult | None = Field(None, description="Tool call (if model decided to call a tool)")
    tool_calls: list[ToolCallResult] | None = Field(None, description="Multiple tool calls (if multi-tool)")
    response: str | None = Field(None, description="Text response (if no tool needed)")
    model: str = Field(..., description="Model identifier")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model: str
    model_loaded: bool


# ============================================================
# Endpoints
# ============================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verify model is loaded and ready."""
    if _model_adapter is None:
        return HealthResponse(
            status="not_ready",
            model="none",
            model_loaded=False,
        )
    return HealthResponse(
        status="healthy",
        model=_model_adapter.model_id,
        model_loaded=_model_adapter._loaded,
    )


@app.post("/v1/tool-call", response_model=ToolCallResponse)
async def tool_call(request: ToolCallRequest):
    """
    Run tool-calling inference.

    Send a user query + available tools → get back either a tool call
    (with name + arguments) or a text response (if no tool is needed).
    """
    if _model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Start server with --model-path.")

    # Build prompt
    from toolforge.eval.models import build_eval_prompt

    tool_schema = {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in request.tools
        ]
    }

    prompt = build_eval_prompt(
        user_query=request.query,
        system_prompt=request.system_prompt or "",
        tool_schema=tool_schema,
    )

    # Run inference
    start = time.time()
    if not _model_adapter._loaded:
        _model_adapter.load()

    raw_output = _model_adapter.generate(prompt, max_tokens=request.max_tokens)
    parsed = _model_adapter.parse_output(raw_output)
    latency_ms = (time.time() - start) * 1000

    # Build response
    response = ToolCallResponse(
        model=_model_adapter.model_id,
        latency_ms=round(latency_ms, 1),
    )

    if "tools" in parsed:
        # Multi-tool response
        response.tool_calls = [
            ToolCallResult(
                name=t.get("tool", t.get("name", "")),
                arguments=t.get("arguments", {}),
            )
            for t in parsed["tools"]
        ]
    elif "tool" in parsed:
        # Single tool call
        response.tool_call = ToolCallResult(
            name=parsed["tool"],
            arguments=parsed.get("arguments", {}),
        )
    elif "response" in parsed:
        # Text response (no tool needed)
        response.response = parsed["response"]
    else:
        response.response = raw_output

    return response


# ============================================================
# Server Startup
# ============================================================


def create_app(
    model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    adapter_path: str | None = None,
    backend: str = "mlx",
) -> FastAPI:
    """
    Create the FastAPI app with a loaded model.

    WHY FACTORY:
      The model needs to be loaded before the first request.
      This factory creates the app and loads the model, so the
      global `_model_adapter` is ready when requests arrive.
    """
    global _model_adapter

    from toolforge.eval.models import create_model_adapter

    _model_adapter = create_model_adapter(
        backend=backend,
        model_id=model_id,
        adapter_path=adapter_path,
    )

    # Eagerly load the model
    _model_adapter.load()

    return app


def run_server(
    model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    adapter_path: str | None = None,
    backend: str = "mlx",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Start the inference server."""
    import uvicorn

    create_app(model_id=model_id, adapter_path=adapter_path, backend=backend)

    print(f"\n🔧 ToolForge serving on http://{host}:{port}")
    print(f"   Model: {model_id}")
    if adapter_path:
        print(f"   Adapters: {adapter_path}")
    print(f"   Docs: http://{host}:{port}/docs\n")

    uvicorn.run(app, host=host, port=port)
