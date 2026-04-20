"""
Model adapters for the evaluation harness.

WHY THIS MODULE EXISTS:
  The eval harness is model-agnostic — it defines the evaluation algorithm
  but not HOW to run inference. This module provides concrete ModelFn
  implementations for different inference backends.

  Each adapter converts:
    (user_query, system_prompt, tool_schema) → formatted prompt → model → parsed dict

ADAPTERS:
  1. MLXModelAdapter    — Apple Silicon native inference via mlx-lm (recommended for M1/M2/M3)
  2. DummyModelAdapter  — Returns random/empty predictions (for testing the harness)
  3. OllamaModelAdapter — Ollama API (for quick experiments without GPU memory management)

DESIGN:
  Each adapter is a class with a `__call__` method matching the ModelFn signature.
  The adapter owns:
    - Model loading (lazy, on first call)
    - Prompt formatting (Llama 3.2 chat template)
    - Output parsing (JSON extraction from generated text)
    - Resource cleanup

  This is the Adapter Pattern — each class adapts a different inference
  engine to the harness's uniform interface.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


# ============================================================
# Base Adapter Interface
# ============================================================


class BaseModelAdapter(ABC):
    """
    Abstract base for model adapters.

    WHY AN ABC:
      Forces every adapter to implement the same interface.
      The harness can accept any BaseModelAdapter subclass.
    """

    def __init__(self, model_id: str, **kwargs: Any):
        self.model_id = model_id
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called lazily on first inference."""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate raw text from a formatted prompt string."""
        ...

    def __call__(self, prompt: str) -> dict[str, Any]:
        """
        ModelFn interface: take a pre-formatted prompt, generate, and parse.

        WHY PARSE HERE:
          The harness expects a dict with tool call info. The model produces
          raw text. Parsing belongs in the adapter because different models
          produce different output formats (JSON, XML, function call syntax).
        """
        if not self._loaded:
            self.load()

        raw_output = self.generate(prompt)
        return self.parse_output(raw_output)

    def parse_output(self, raw_text: str) -> dict[str, Any]:
        """
        Parse model output into a structured dict.

        Expected output format (what the metrics expect):
          For tool calls:    {"tool": "func_name", "arguments": {...}}
          For multi-tool:    {"tools": [{"tool": "f1", "arguments": {...}}, ...]}
          For no-tool:       {"response": "text response"}

        The parser tries multiple strategies:
          1. Direct JSON parse (model outputs clean JSON)
          2. JSON extraction from text (model wraps JSON in explanation)
          3. Regex-based extraction (model uses informal format)
          4. Fallback to raw text response
        """
        text = raw_text.strip()

        # Strategy 1: Direct JSON parse
        parsed = self._try_json_parse(text)
        if parsed:
            return self._normalize_parsed(parsed)

        # Strategy 2: Extract JSON from text (find first { ... } block)
        parsed = self._try_extract_json(text)
        if parsed:
            return self._normalize_parsed(parsed)

        # Strategy 3: Extract JSON array (multi-tool)
        parsed = self._try_extract_json_array(text)
        if parsed:
            return {"tools": [self._normalize_parsed(p) for p in parsed]}

        # Fallback: treat as text response (model declined to call a tool)
        return {"response": text}

    def _try_json_parse(self, text: str) -> dict | None:
        """Try parsing the entire text as JSON."""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _try_extract_json(self, text: str) -> dict | None:
        """
        Extract the first JSON object from text using brace counting.

        WHY NOT REGEX:
          Same reason as in download.py — regex with non-greedy match fails
          on nested braces. We use the same brace-counting approach.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None

        return None

    def _try_extract_json_array(self, text: str) -> list[dict] | None:
        """Extract a JSON array from text."""
        start = text.find("[")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:i + 1])
                        if isinstance(result, list) and all(isinstance(r, dict) for r in result):
                            return result
                    except json.JSONDecodeError:
                        return None

        return None

    def _normalize_parsed(self, parsed: dict) -> dict[str, Any]:
        """
        Normalize different output formats to our canonical dict format.

        Models output tool calls in various formats:
          - {"name": "func", "arguments": {...}}     (OpenAI-style)
          - {"tool": "func", "arguments": {...}}     (our format)
          - {"function_call": {"name": ..., ...}}    (older OpenAI)
          - {"name": "func", "parameters": {...}}    (alternative)

        We normalize all to: {"tool": "func_name", "arguments": {...}}
        """
        # Already in our format
        if "tool" in parsed:
            return parsed

        # OpenAI-style: {"name": ..., "arguments": ...}
        if "name" in parsed:
            return {
                "tool": parsed["name"],
                "arguments": parsed.get("arguments", parsed.get("parameters", {})),
            }

        # Nested: {"function_call": {"name": ..., "arguments": ...}}
        if "function_call" in parsed:
            fc = parsed["function_call"]
            return {
                "tool": fc.get("name", ""),
                "arguments": fc.get("arguments", {}),
            }

        # Can't determine tool name — return as-is
        return parsed


# ============================================================
# Prompt Builder
# ============================================================


def build_eval_prompt(
    user_query: str,
    system_prompt: str,
    tool_schema: dict[str, Any],
) -> str:
    """
    Build a complete Llama 3.2 chat template prompt for evaluation.

    WHY BUILD PROMPT HERE (not in formatter.py):
      formatter.py works with ToolCallingExample objects (data pipeline).
      This function works with raw eval dataset fields (strings + dicts).
      Different inputs, same output format.

    Args:
        user_query: The user's natural language question
        system_prompt: System instructions (may be empty)
        tool_schema: Dict with "tools" key containing tool definitions

    Returns:
        Fully formatted Llama 3.2 chat template string (inference format)
    """
    BOS = "<|begin_of_text|>"
    EOT = "<|eot_id|>"
    HEADER_START = "<|start_header_id|>"
    HEADER_END = "<|end_header_id|>"

    # Build system content
    if system_prompt and system_prompt.strip():
        sys_content = system_prompt.strip()
    else:
        sys_content = (
            "You are a helpful assistant with access to tools. "
            "Use the provided tools when appropriate to answer the user's query. "
            "If no tool is needed, respond directly with text."
        )

    # Add tool definitions
    tools = tool_schema.get("tools", [])
    if tools:
        tools_json = json.dumps(
            [{"type": "function", "function": {"name": t.get("name", ""), "description": t.get("description", ""), "parameters": t.get("parameters", {})}} for t in tools],
            indent=2,
        )
        sys_content += (
            f"\n\nYou have access to the following tools:\n{tools_json}\n\n"
            "When you need to call a tool, respond ONLY with a JSON object in this exact format:\n"
            '{"name": "<tool_name>", "arguments": {<arg_name>: <arg_value>, ...}}\n\n'
            "If no tool is needed, respond normally with text. "
            "Do NOT add any explanation before or after the JSON."
        )

    return (
        f"{BOS}"
        f"{HEADER_START}system{HEADER_END}\n\n"
        f"{sys_content}{EOT}"
        f"{HEADER_START}user{HEADER_END}\n\n"
        f"{user_query}{EOT}"
        f"{HEADER_START}assistant{HEADER_END}\n\n"
    )


# ============================================================
# MLX Model Adapter (Apple Silicon)
# ============================================================


class MLXModelAdapter(BaseModelAdapter):
    """
    Model adapter using MLX for Apple Silicon native inference.

    WHY MLX:
      On M1/M2/M3 Macs, MLX uses the unified memory architecture to run
      LLMs at ~40-60 tokens/sec for 3B models. HuggingFace transformers
      would fall back to CPU (MPS support is spotty for LLMs), giving
      ~5 tokens/sec. MLX is 10x faster on this hardware.

    USAGE:
      adapter = MLXModelAdapter("mlx-community/Llama-3.2-3B-Instruct-4bit")
      result = adapter("What's the weather in Tokyo?")
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(model_id, **kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load the MLX model and tokenizer."""
        console.print(f"\n[bold blue]Loading model:[/bold blue] {self.model_id}")
        start = time.time()

        try:
            # Patch httpx for corporate proxy SSL (Netskope/Zscaler intercept)
            try:
                import httpx

                _orig_init = httpx.Client.__init__

                def _ssl_patched_init(self_client, *args, **kwargs):
                    kwargs.setdefault("verify", False)
                    _orig_init(self_client, *args, **kwargs)

                httpx.Client.__init__ = _ssl_patched_init
            except ImportError:
                pass

            from mlx_lm import load as mlx_load

            self._model, self._tokenizer = mlx_load(self.model_id)
            self._loaded = True

            elapsed = time.time() - start
            console.print(f"  [green]Model loaded in {elapsed:.1f}s[/green]")

        except ImportError:
            raise RuntimeError(
                "mlx-lm is not installed. Install with:\n"
                "  pip install mlx-lm\n"
                "MLX only works on Apple Silicon Macs."
            )

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate text using MLX inference."""
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        max_tok = max_tokens or self.max_tokens

        # mlx-lm >= 0.31 uses sampler objects instead of temp/top_p kwargs
        sampler = make_sampler(temp=self.temperature)

        output = mlx_generate(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_tokens=max_tok,
            sampler=sampler,
            verbose=False,
        )

        return output


# ============================================================
# Ollama Model Adapter
# ============================================================


class OllamaModelAdapter(BaseModelAdapter):
    """
    Model adapter using Ollama's local API.

    WHY OLLAMA:
      Quick experiments without managing GPU memory. Ollama handles
      model loading/unloading automatically. Good for comparing models
      (just change the model name) but slower than MLX for evaluation.
    """

    def __init__(
        self,
        model_id: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(model_id, **kwargs)
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def load(self) -> None:
        """Verify Ollama is running and model is available."""
        import urllib.request

        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m.get("name", "") for m in data.get("models", [])]
                if not any(self.model_id in m for m in models):
                    console.print(
                        f"[yellow]Model {self.model_id} not found in Ollama. "
                        f"Available: {models}. Will attempt to pull.[/yellow]"
                    )
                self._loaded = True
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}: {e}\n"
                "Start Ollama with: ollama serve"
            )

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate text using Ollama API."""
        import urllib.request

        url = f"{self.base_url}/api/generate"
        payload = json.dumps({
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens or self.max_tokens,
                "temperature": self.temperature,
            },
        }).encode()

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")


# ============================================================
# Dummy Adapter (for testing)
# ============================================================


class DummyModelAdapter(BaseModelAdapter):
    """
    Returns canned responses for testing the harness without a real model.

    WHY THIS EXISTS:
      You don't want to download a 2GB model just to test that the eval
      harness wiring works. This adapter returns deterministic responses
      so we can unit-test the full eval pipeline.
    """

    def __init__(self, model_id: str = "dummy", responses: list[str] | None = None, **kwargs: Any):
        super().__init__(model_id, **kwargs)
        self._responses = responses or []
        self._call_count = 0

    def load(self) -> None:
        self._loaded = True

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
        else:
            # Default: return a wrong tool call (simulates bad base model)
            response = '{"name": "unknown_tool", "arguments": {}}'
        self._call_count += 1
        return response


# ============================================================
# Factory
# ============================================================


def create_model_adapter(
    backend: str = "mlx",
    model_id: str | None = None,
    **kwargs: Any,
) -> BaseModelAdapter:
    """
    Factory function to create the right model adapter.

    Args:
        backend: One of "mlx", "ollama", "dummy"
        model_id: Model identifier (HF repo for MLX, model name for Ollama)
        **kwargs: Passed to the adapter constructor

    Returns:
        A BaseModelAdapter instance (not yet loaded — loads lazily)

    WHY A FACTORY:
      The CLI and harness don't need to know which adapter class to import.
      `create_model_adapter("mlx")` is cleaner than importing MLXModelAdapter.
    """
    adapters = {
        "mlx": (MLXModelAdapter, "mlx-community/Llama-3.2-3B-Instruct-4bit"),
        "ollama": (OllamaModelAdapter, "llama3.2:3b"),
        "dummy": (DummyModelAdapter, "dummy"),
    }

    if backend not in adapters:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {list(adapters.keys())}")

    adapter_cls, default_model = adapters[backend]
    return adapter_cls(model_id=model_id or default_model, **kwargs)
