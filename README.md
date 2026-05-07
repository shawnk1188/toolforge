# ToolForge

Production-grade AI orchestration framework for reliable tool execution, structured streaming, replayable workflows, and scalable agentic AI systems.

> Built for deterministic AI orchestration, production-ready agent workflows, and enterprise-grade AI infrastructure.

---

## Overview

ToolForge is a production-focused AI orchestration framework designed to solve real-world challenges in building scalable LLM-powered systems.

The platform enables:

- Reliable tool execution
- Structured AI workflows
- Replayable orchestration pipelines
- Deterministic agent execution
- MCP/FastMCP compatible workflows
- Typed tool invocation
- Streaming AI interactions
- AI evaluation and observability

Unlike traditional chatbot wrappers, ToolForge focuses on engineering discipline, resiliency, and operational reliability for modern AI applications.

---

## Key Features

- Production-grade AI orchestration
- Reliable tool calling pipelines
- Structured streaming architecture
- MCP/FastMCP compatible execution
- Agentic AI workflows
- Replayable execution artifacts
- AI evaluation support
- Deterministic orchestration pipelines
- Async Python execution engine
- FastAPI-powered APIs
- Structured outputs with Pydantic
- Retry and repair workflows
- AI observability patterns
- Extensible plugin architecture
- Container-ready deployments

---

## Architecture

```text
User Request
     ↓
Planner / Router
     ↓
Tool Selection Engine
     ↓
Structured Execution Pipeline
     ↓
Validation & Repair Layer
     ↓
Streaming Response Engine
     ↓
Replayable Execution Artifacts
     ↓
Final AI Response
```

---

## Why ToolForge Exists

Modern LLM applications often struggle with:

- Unreliable tool execution
- Non-deterministic outputs
- Poor observability
- Difficult debugging
- Limited replayability
- Weak validation pipelines
- Inconsistent orchestration patterns

ToolForge was built to address these challenges by applying enterprise-grade engineering principles to AI systems and agent workflows.

The framework emphasizes:

- Reliability
- Determinism
- Scalability
- Observability
- Structured orchestration
- Production readiness

---

## Core Components

### Orchestration Engine

Coordinates execution flow across AI agents, tools, retries, and validation layers.

### Tool Execution Layer

Provides structured and typed tool invocation with validation, retries, and deterministic execution support.

### Streaming Pipeline

Supports structured streaming responses and incremental AI output processing.

### Validation & Repair Engine

Automatically validates outputs and applies retry/repair logic for resilient execution.

### Replayable Execution Artifacts

Stores execution metadata and orchestration traces for debugging and reproducibility.

### MCP/FastMCP Compatibility

Supports Model Context Protocol workflows for scalable AI tool interoperability.

---

## Tech Stack

### AI & Orchestration

- OpenAI APIs
- LangChain
- MCP
- FastMCP
- Agentic AI Workflows

### Backend

- Python
- FastAPI
- AsyncIO
- Pydantic v2

### Infrastructure

- Docker
- Kubernetes
- Redis
- CI/CD Pipelines

### Observability

- Structured Logging
- Metrics Pipelines
- AI Execution Tracing

---

## Installation

```bash
git clone https://github.com/shawnk1188/toolforge.git

cd toolforge

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
MODEL_NAME=gpt-4o
REDIS_URL=redis://localhost:6379
```

---

## Quick Start

```bash
uvicorn app.main:app --reload
```

API available at:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

---

## Example Workflow

```python
from toolforge.agent import Agent

agent = Agent()

response = agent.run(
    query="Summarize API failures from the last deployment"
)

print(response)
```

---

## Example Structured Output

```json
{
  "tool_used": "log_analyzer",
  "status": "success",
  "summary": "Deployment failures caused by authentication timeout",
  "retry_attempts": 1,
  "execution_time_ms": 842
}
```

---

## Production Features

- Structured retry pipelines
- Deterministic execution workflows
- Typed orchestration patterns
- AI workflow observability
- Replayable execution traces
- Async execution support
- Containerized deployment support
- CI/CD integration
- Structured logging
- Metrics instrumentation
- Failure recovery patterns
- AI reliability engineering

---

## Use Cases

### AI Agents

Build scalable agentic workflows with structured execution patterns.

### Enterprise AI Platforms

Enable production-ready AI orchestration and observability.

### RAG Systems

Integrate reliable retrieval and orchestration pipelines.

### Tool Calling Workflows

Implement deterministic and validated AI tool execution.

### AI Infrastructure

Support scalable cloud-native AI systems and inference workflows.

---

## Future Roadmap

- Multi-agent orchestration
- AI evaluation pipelines
- OpenTelemetry integration
- Distributed execution engine
- vLLM support
- DSPy integration
- Advanced tracing and replay tools
- Kubernetes-native orchestration
- Vector memory integrations

---

## Engineering Goals

ToolForge is designed around the philosophy that modern AI systems should be:

- Reliable
- Observable
- Deterministic
- Replayable
- Scalable
- Production-ready

The project focuses on bringing enterprise engineering discipline to Generative AI infrastructure and agent orchestration.

---

## Author

### Sushanth Kakarlapudi

Senior Lead Engineer specializing in:

- Generative AI
- Agentic AI
- AI Platform Engineering
- Distributed Systems
- Cloud-Native Infrastructure
- Production AI Systems

---

## License

MIT License
