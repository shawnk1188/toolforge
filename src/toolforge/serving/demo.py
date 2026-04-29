"""
Interactive Gradio demo for ToolForge.

WHY GRADIO:
  - Zero frontend code needed — pure Python
  - Auto-generates a shareable URL for demos
  - Built-in components for chat-style interfaces
  - Industry standard for ML model demos (HuggingFace Spaces)

USAGE:
  toolforge serve demo                     # Default model
  toolforge serve demo --adapter-path artifacts/dpo/adapters  # DPO model
"""

from __future__ import annotations

import json
import time
from typing import Any

# Default tool definitions for the demo
DEMO_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Number of results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"},
            },
            "required": ["ticker"],
        },
    },
]


def create_demo(
    model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    adapter_path: str | None = None,
    backend: str = "mlx",
) -> Any:
    """
    Create the Gradio demo interface.

    Returns a Gradio Blocks app ready to launch.
    """
    import gradio as gr

    from toolforge.eval.models import build_eval_prompt, create_model_adapter

    # Load model
    adapter = create_model_adapter(
        backend=backend,
        model_id=model_id,
        adapter_path=adapter_path,
    )
    adapter.load()

    model_label = model_id.split("/")[-1]
    if adapter_path:
        model_label += f" + {adapter_path.split('/')[-1]}"

    def predict(query: str, tools_json: str) -> str:
        """Run inference and format the result."""
        try:
            tools = json.loads(tools_json)
        except json.JSONDecodeError:
            return "❌ Invalid JSON in tools definition"

        tool_schema = {"tools": tools}
        prompt = build_eval_prompt(
            user_query=query,
            system_prompt="",
            tool_schema=tool_schema,
        )

        start = time.time()
        raw = adapter.generate(prompt)
        parsed = adapter.parse_output(raw)
        latency = (time.time() - start) * 1000

        # Format output
        lines = []
        lines.append(f"**Model:** {model_label}")
        lines.append(f"**Latency:** {latency:.0f}ms")
        lines.append("")

        if "tool" in parsed and parsed["tool"]:
            lines.append("### 🔧 Tool Call")
            lines.append(f"**Function:** `{parsed['tool']}`")
            lines.append(f"**Arguments:**")
            lines.append(f"```json\n{json.dumps(parsed.get('arguments', {}), indent=2)}\n```")
        elif "tools" in parsed:
            lines.append("### 🔧 Multi-Tool Calls")
            for i, tc in enumerate(parsed["tools"], 1):
                lines.append(f"\n**{i}. `{tc.get('tool', tc.get('name', '?'))}`**")
                lines.append(f"```json\n{json.dumps(tc.get('arguments', {}), indent=2)}\n```")
        elif "response" in parsed:
            lines.append("### 💬 Text Response")
            lines.append(parsed["response"])
        else:
            lines.append("### Raw Output")
            lines.append(f"```\n{raw}\n```")

        lines.append(f"\n---\n*Raw model output:*\n```\n{raw[:500]}\n```")
        return "\n".join(lines)

    # Example queries
    examples = [
        ["What's the weather in Tokyo?", json.dumps(DEMO_TOOLS, indent=2)],
        ["Search for the latest AI news", json.dumps(DEMO_TOOLS, indent=2)],
        ["What's 42 * 17 + 3?", json.dumps(DEMO_TOOLS, indent=2)],
        ["Tell me a joke about programming", json.dumps(DEMO_TOOLS, indent=2)],  # Should NOT call a tool
        ["Send an email to bob@example.com about the meeting tomorrow", json.dumps(DEMO_TOOLS, indent=2)],
        ["What's Apple's stock price?", json.dumps(DEMO_TOOLS, indent=2)],
    ]

    with gr.Blocks(
        title="ToolForge Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            f"""
            # 🔧 ToolForge — Precision Tool-Calling Demo

            Fine-tuned **Llama 3.2 3B** for accurate tool-calling.
            Model: `{model_label}`

            Enter a natural language query and the model will decide whether to call a tool
            or respond with text. Try the examples below!
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="User Query",
                    placeholder="e.g., What's the weather in Tokyo?",
                    lines=2,
                )
                tools_input = gr.Textbox(
                    label="Available Tools (JSON)",
                    value=json.dumps(DEMO_TOOLS, indent=2),
                    lines=15,
                )
                submit_btn = gr.Button("🚀 Run Inference", variant="primary")

            with gr.Column(scale=1):
                output = gr.Markdown(label="Model Output")

        gr.Examples(
            examples=examples,
            inputs=[query_input, tools_input],
            label="Example Queries",
        )

        submit_btn.click(
            fn=predict,
            inputs=[query_input, tools_input],
            outputs=output,
        )

    return demo


def run_demo(
    model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    adapter_path: str | None = None,
    backend: str = "mlx",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio demo."""
    demo = create_demo(
        model_id=model_id,
        adapter_path=adapter_path,
        backend=backend,
    )
    demo.launch(server_port=port, share=share)
