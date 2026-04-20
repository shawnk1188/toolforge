"""
Data schema for tool-calling training examples.

WHY THIS MODULE EXISTS:
  Open-source datasets come in wildly different formats:
    - Glaive: chat-style with system/user/assistant messages
    - BFCL: prompt + function definitions + expected output
    - Gorilla: natural language query + API call

  We need ONE canonical format that all datasets are normalized into.
  This module defines that format using Pydantic, so every example
  is validated at conversion time — bad data is caught immediately,
  not discovered during training when a JSON parse fails at 3am.

SCHEMA DESIGN DECISIONS:
  1. Tool definitions use OpenAI-compatible function schema format —
     this is the de facto standard that most models are trained on.
  2. Each example is a complete conversation (system + user + assistant),
     not a standalone prompt. This matches how chat models are trained.
  3. The assistant response can be either a tool call OR a text response
     (for relevance_detection cases where no tool should be called).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Tool Definition Schema (OpenAI-compatible)
# ============================================================


class ParameterProperty(BaseModel):
    """A single parameter in a tool's input schema."""

    type: str = Field(..., description="JSON type: string, number, integer, boolean, array, object")
    description: str = Field(default="", description="Human-readable parameter description")
    enum: Optional[list[str]] = Field(None, description="Allowed values (for constrained params)")


class ToolParameters(BaseModel):
    """Parameters schema for a tool definition (JSON Schema subset)."""

    type: str = Field(default="object")
    properties: dict[str, ParameterProperty] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """
    A single tool available for the model to call.

    Uses OpenAI-compatible function calling format because:
    1. It's the most widely adopted format across model providers
    2. Most open-source tool-calling datasets use this format
    3. Llama 3.2's tool-calling fine-tuning uses this schema
    """

    name: str = Field(..., description="Function name (snake_case)")
    description: str = Field(..., description="What this tool does")
    parameters: ToolParameters = Field(default_factory=ToolParameters)

    @field_validator("name")
    @classmethod
    def name_must_be_valid_identifier(cls, v: str) -> str:
        """Tool names should be valid Python identifiers."""
        cleaned = v.strip()
        if not cleaned.replace("_", "").replace(".", "").isalnum():
            raise ValueError(f"Tool name must be alphanumeric with underscores, got: '{v}'")
        return cleaned


# ============================================================
# Tool Call Schema (what the model produces)
# ============================================================


class ToolCall(BaseModel):
    """A single tool call made by the model."""

    name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments to pass")


# ============================================================
# Training Example Schema
# ============================================================


class ExampleType(str, Enum):
    """
    Type of training example — determines which specs it's relevant for.

    WHY THIS MATTERS:
      When we generate eval datasets for each spec, we filter by type.
      tool_selection tests use SINGLE_TOOL examples.
      relevance_detection tests use NO_TOOL examples.
      multi_tool_sequencing tests use MULTI_TOOL examples.
      error_recovery tests use ERROR_HANDLING examples.
    """

    SINGLE_TOOL = "single_tool"       # User query → one tool call
    MULTI_TOOL = "multi_tool"         # User query → sequence of tool calls
    NO_TOOL = "no_tool"               # User query → text response (no tool needed)
    ERROR_HANDLING = "error_handling"  # Tool returns error → model responds gracefully


class ToolCallingExample(BaseModel):
    """
    A single training/evaluation example for tool-calling.

    This is the CANONICAL FORMAT that all datasets are normalized into.
    Every Glaive example, every BFCL example, every custom example
    passes through this schema before entering the training pipeline.

    Format:
      - system_prompt: instructions about available tools
      - user_query: what the user asked
      - available_tools: list of tool definitions
      - expected_tool_calls: what the model SHOULD produce (can be empty)
      - expected_response: text response (for no_tool cases)
      - example_type: categorization for spec routing
    """

    # Unique identifier for deduplication and tracing
    id: str = Field(..., description="Unique example ID (dataset_name:index)")

    # The conversation
    system_prompt: str = Field(default="", description="System instructions (may include tool list)")
    user_query: str = Field(..., min_length=1, description="The user's natural language query")

    # Available tools for this example
    available_tools: list[ToolDefinition] = Field(
        default_factory=list,
        description="Tools the model can choose from",
    )

    # Expected model output (at least one of these must be set)
    expected_tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Expected tool call(s) — empty for no_tool examples",
    )
    expected_response: Optional[str] = Field(
        None,
        description="Expected text response (for no_tool or error_handling cases)",
    )

    # Metadata
    example_type: ExampleType = Field(..., description="Type of example for spec routing")
    source_dataset: str = Field(..., description="Which dataset this came from")
    difficulty: Optional[str] = Field(None, description="easy, medium, hard (if available)")

    @model_validator(mode="after")
    def must_have_expected_output(self) -> "ToolCallingExample":
        """Every example must have either tool calls or a text response."""
        if not self.expected_tool_calls and not self.expected_response:
            raise ValueError(
                "Example must have at least one expected_tool_call or an expected_response. "
                f"ID: {self.id}"
            )
        return self

    @model_validator(mode="after")
    def tool_calls_must_reference_available_tools(self) -> "ToolCallingExample":
        """Every expected tool call must reference a tool in available_tools."""
        available_names = {t.name for t in self.available_tools}
        for call in self.expected_tool_calls:
            if call.name not in available_names:
                raise ValueError(
                    f"Tool call '{call.name}' not in available tools: {available_names}. "
                    f"ID: {self.id}"
                )
        return self

    def to_eval_format(self) -> dict:
        """
        Convert to the format expected by the eval harness.

        The eval harness expects:
          - "prompt": string to send to the model
          - "expected": dict with tool/arguments or tool=None
          - "tool_schema": available tools as dict
        """
        tool_schema = {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters.model_dump(),
                }
                for t in self.available_tools
            ]
        }

        if self.example_type == ExampleType.MULTI_TOOL:
            expected = {
                "tools": [
                    {"tool": tc.name, "arguments": tc.arguments}
                    for tc in self.expected_tool_calls
                ]
            }
        elif self.expected_tool_calls:
            tc = self.expected_tool_calls[0]
            expected = {"tool": tc.name, "arguments": tc.arguments}
        else:
            expected = {"tool": None, "response": self.expected_response or ""}

        return {
            "prompt": self.user_query,
            "expected": expected,
            "tool_schema": tool_schema,
            "system_prompt": self.system_prompt,
            "id": self.id,
        }


# ============================================================
# Dataset Split Schema
# ============================================================


class DatasetSplit(BaseModel):
    """Metadata for a processed dataset split."""

    split_name: str = Field(..., description="train, val, or test")
    num_examples: int = Field(..., ge=0)
    type_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each ExampleType in this split",
    )
    source_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count from each source dataset",
    )
    filepath: str = Field(..., description="Path to the .jsonl file")


class DatasetManifest(BaseModel):
    """
    Complete metadata for a processed dataset.

    Saved alongside the processed data so we always know:
    - How many examples per split
    - Distribution of example types
    - Which source datasets contributed
    - Processing timestamp
    """

    version: str = Field(default="1.0")
    total_examples: int
    splits: list[DatasetSplit]
    processing_timestamp: str
    source_datasets: list[str]
    dedup_removed: int = Field(default=0, description="Examples removed by deduplication")
    validation_failed: int = Field(default=0, description="Examples that failed schema validation")
