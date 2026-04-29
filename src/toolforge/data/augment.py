"""
Data augmentation for underrepresented training example types.

WHY THIS MODULE EXISTS:
  After Stage 4 SFT, three specs have critical data gaps:

  1. error_recovery (0.060): ZERO error_handling examples in SFT training.
     The synthesized error examples only went to eval datasets, not training.
     The model has literally never seen an error-recovery pattern.

  2. relevance_detection (0.186): Only 9% of training is no_tool (247/2757).
     The model learns to always call tools because 91% of training teaches
     tool-calling. We need more no_tool examples to balance the dataset.

  3. multi_tool_sequencing (0.000): 899 multi_tool examples exist but the
     model still can't output JSON arrays. We add reinforcement examples
     with clearer, shorter patterns to help the 3B model learn the format.

APPROACH:
  We generate synthetic training examples by:
  - Reusing tool definitions from existing examples (realistic tool schemas)
  - Creating diverse query patterns (not just templates)
  - Validating every example through Pydantic (same as real data)

  This is a standard data augmentation technique — the key insight is that
  behavioral skills (error handling, refusal, multi-step) can be taught
  through synthetic examples when real data is scarce.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rich.console import Console

from toolforge.data.schema import (
    ExampleType,
    ToolCall,
    ToolCallingExample,
    ToolDefinition,
)

console = Console()


# ============================================================
# Error Handling Example Generation
# ============================================================

# Diverse error messages that a tool might return
ERROR_MESSAGES = [
    "Service temporarily unavailable (503)",
    "Rate limit exceeded. Please try again later.",
    "Invalid API key. Authentication failed.",
    "Resource not found (404)",
    "Request timeout after 30 seconds",
    "Internal server error (500)",
    "Permission denied. Insufficient privileges.",
    "Invalid input format. Expected JSON.",
    "Connection refused. Service is down.",
    "Quota exceeded for this billing period.",
    "Database connection timeout",
    "SSL certificate verification failed",
]

# Templates for error acknowledgment responses (the model should learn these patterns)
ERROR_RESPONSE_TEMPLATES = [
    "I apologize, but I encountered an error when trying to use {tool_name}: {error}. Would you like me to try again or help with something else?",
    "I'm sorry, but the {tool_name} tool returned an error: {error}. I can try a different approach if you'd like.",
    "Unfortunately, I wasn't able to complete that request. The {tool_name} function encountered an issue: {error}. How would you like to proceed?",
    "I ran into a problem using {tool_name} — {error}. Would you like me to attempt an alternative solution?",
    "The tool {tool_name} was unable to process the request due to: {error}. I can try again or help with something else.",
    "I'm sorry, but there was an error with {tool_name}: {error}. Let me know if you'd like me to retry or take a different approach.",
]


def generate_error_handling_examples(
    source_examples: list[ToolCallingExample],
    count: int = 400,
    seed: int = 42,
) -> list[ToolCallingExample]:
    """
    Generate error_handling training examples from existing tool-calling examples.

    Takes normal single_tool examples and transforms them into error scenarios
    where the tool was called but returned an error. The model should learn to
    respond with a helpful text message acknowledging the error.

    Args:
        source_examples: Existing single_tool examples to base errors on
        count: Number of error examples to generate
        seed: Random seed for reproducibility

    Returns:
        List of validated ToolCallingExample objects with type=ERROR_HANDLING
    """
    rng = random.Random(seed)

    # Filter to single_tool examples that have tool calls
    candidates = [
        ex for ex in source_examples
        if ex.expected_tool_calls and ex.example_type == ExampleType.SINGLE_TOOL
    ]

    if not candidates:
        console.print("[yellow]No single_tool examples found for error augmentation[/yellow]")
        return []

    error_examples = []

    for i in range(count):
        source = rng.choice(candidates)
        tc = source.expected_tool_calls[0]
        error_msg = rng.choice(ERROR_MESSAGES)
        response_template = rng.choice(ERROR_RESPONSE_TEMPLATES)

        # Build query with error context (same format as eval data)
        error_query = (
            f"{source.user_query}\n\n"
            f"[System: The tool '{tc.name}' was called but returned an error: "
            f'"{error_msg}". Please respond appropriately.]'
        )

        # Build expected response
        expected_response = response_template.format(
            tool_name=tc.name,
            error=error_msg,
        )

        try:
            error_ex = ToolCallingExample(
                id=f"augmented_error:{i}",
                system_prompt=source.system_prompt,
                user_query=error_query,
                available_tools=source.available_tools,
                expected_tool_calls=[],  # No tool call — model should respond with text
                expected_response=expected_response,
                example_type=ExampleType.ERROR_HANDLING,
                source_dataset="augmented",
            )
            error_examples.append(error_ex)
        except Exception:
            continue  # Skip if validation fails

    console.print(f"  [green]Generated {len(error_examples)} error_handling examples[/green]")
    return error_examples


# ============================================================
# No-Tool (Relevance Detection) Example Generation
# ============================================================

# Queries that should NOT trigger a tool call, across different categories
NO_TOOL_QUERY_TEMPLATES = [
    # General knowledge (no tool needed)
    "What is the capital of {country}?",
    "Explain the concept of {concept} in simple terms.",
    "What are the main differences between {thing_a} and {thing_b}?",
    "Can you summarize the key points of {topic}?",
    "How does {process} work?",
    "What is the history of {subject}?",
    "Why is {phenomenon} important?",
    "Tell me about {person_or_place}.",

    # Conversational / greetings
    "Hello! How are you today?",
    "Thanks for your help earlier!",
    "What can you help me with?",
    "That makes sense, thank you for explaining.",
    "I appreciate the information.",
    "Good morning! Ready to get started?",
    "Can you tell me a joke?",
    "What's your opinion on {topic}?",

    # Clarification requests
    "What did you mean by that?",
    "Can you elaborate on the last point?",
    "I didn't quite understand — could you explain differently?",
    "What would you recommend for a beginner?",

    # Math / logic (model can answer directly)
    "What is {num_a} plus {num_b}?",
    "Is {number} a prime number?",
    "What percentage is {part} of {whole}?",

    # Advice / opinions
    "Should I learn Python or JavaScript first?",
    "What are some good practices for {activity}?",
    "How can I improve my {skill}?",
    "What are the pros and cons of {choice}?",
]

# Fill-in values for templates
TEMPLATE_FILLS = {
    "country": ["France", "Japan", "Brazil", "Australia", "Egypt", "Canada", "India", "Germany"],
    "concept": ["machine learning", "blockchain", "quantum computing", "photosynthesis", "relativity", "democracy", "supply chain"],
    "thing_a": ["Python", "REST APIs", "SQL", "Docker", "TCP", "Linux"],
    "thing_b": ["JavaScript", "GraphQL", "NoSQL", "Kubernetes", "UDP", "Windows"],
    "topic": ["renewable energy", "artificial intelligence", "space exploration", "cybersecurity", "climate change"],
    "process": ["DNS resolution", "HTTP handshake", "garbage collection", "neural network training", "photosynthesis"],
    "subject": ["the internet", "programming languages", "artificial intelligence", "the printing press"],
    "phenomenon": ["biodiversity", "network effects", "compound interest", "the greenhouse effect"],
    "person_or_place": ["the Great Wall of China", "Silicon Valley", "Marie Curie", "the Amazon rainforest"],
    "activity": ["software development", "public speaking", "time management", "project planning"],
    "skill": ["writing", "coding", "problem-solving", "communication"],
    "choice": ["remote work", "microservices", "open source", "cloud computing"],
    "num_a": ["15", "42", "100", "256", "1024"],
    "num_b": ["27", "58", "200", "128", "512"],
    "number": ["17", "51", "97", "143", "1009"],
    "part": ["25", "40", "75", "120"],
    "whole": ["200", "500", "1000", "300"],
}

# Responses for no-tool queries (the model should generate helpful text)
NO_TOOL_RESPONSE_PREFIXES = [
    "Based on my knowledge,",
    "That's a great question!",
    "Here's what I can tell you:",
    "Sure, I'd be happy to help.",
    "Let me explain:",
    "Great question —",
    "I can answer that directly.",
    "No tool is needed for this —",
]


def generate_no_tool_examples(
    source_examples: list[ToolCallingExample],
    count: int = 300,
    seed: int = 42,
) -> list[ToolCallingExample]:
    """
    Generate no_tool training examples to improve relevance detection.

    Creates examples where tools ARE available but the query doesn't need them.
    This teaches the model to NOT call tools when a text response is appropriate.

    Args:
        source_examples: Existing examples (we borrow their tool definitions)
        count: Number of no_tool examples to generate
        seed: Random seed for reproducibility

    Returns:
        List of validated ToolCallingExample objects with type=NO_TOOL
    """
    rng = random.Random(seed)

    # Collect diverse tool definitions from existing examples
    all_tool_sets = []
    for ex in source_examples:
        if ex.available_tools:
            all_tool_sets.append(ex.available_tools)

    if not all_tool_sets:
        console.print("[yellow]No tool definitions found for no_tool augmentation[/yellow]")
        return []

    no_tool_examples = []

    for i in range(count):
        # Pick a random query template and fill it
        template = rng.choice(NO_TOOL_QUERY_TEMPLATES)
        query = _fill_template(template, rng)

        # Pick random tools to be "available" (but irrelevant to the query)
        tools = rng.choice(all_tool_sets)

        # Generate a simple response
        prefix = rng.choice(NO_TOOL_RESPONSE_PREFIXES)
        response = f"{prefix} {query.rstrip('?').rstrip('!')}. I can provide more details if needed."

        system_prompt = (
            "You are a function calling AI model. You are provided with function "
            "signatures within <tools></tools> XML tags. You may call one or more "
            "functions to assist with the user query. Don't make assumptions about "
            "what values to plug into functions. If the user's question can be "
            "answered directly without using any tools, respond with a helpful text answer."
        )

        try:
            no_tool_ex = ToolCallingExample(
                id=f"augmented_no_tool:{i}",
                system_prompt=system_prompt,
                user_query=query,
                available_tools=tools,
                expected_tool_calls=[],
                expected_response=response,
                example_type=ExampleType.NO_TOOL,
                source_dataset="augmented",
            )
            no_tool_examples.append(no_tool_ex)
        except Exception:
            continue

    console.print(f"  [green]Generated {len(no_tool_examples)} no_tool examples[/green]")
    return no_tool_examples


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a template string with random values."""
    result = template
    for key, values in TEMPLATE_FILLS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, rng.choice(values), 1)
    return result


# ============================================================
# Multi-Tool Reinforcement Example Generation
# ============================================================


def generate_multi_tool_reinforcement(
    source_examples: list[ToolCallingExample],
    count: int = 100,
    seed: int = 42,
) -> list[ToolCallingExample]:
    """
    Generate simpler multi-tool examples to reinforce JSON array output format.

    The SFT model scored 0% on multi_tool_sequencing despite 899 training examples.
    This generates SHORTER, CLEARER multi-tool examples with 2 tools (not 3-5)
    to help the 3B model learn the array format.

    Args:
        source_examples: Existing multi_tool examples for tool definitions
        count: Number of reinforcement examples to generate
        seed: Random seed for reproducibility

    Returns:
        List of validated ToolCallingExample objects with type=MULTI_TOOL
    """
    rng = random.Random(seed)

    # Collect existing multi_tool examples
    multi_examples = [
        ex for ex in source_examples
        if ex.example_type == ExampleType.MULTI_TOOL
        and len(ex.expected_tool_calls) == 2  # Focus on 2-tool sequences (simplest)
    ]

    if not multi_examples:
        # Fall back to any multi_tool
        multi_examples = [
            ex for ex in source_examples
            if ex.example_type == ExampleType.MULTI_TOOL
        ]

    if not multi_examples:
        console.print("[yellow]No multi_tool examples found for reinforcement[/yellow]")
        return []

    reinforcement = []
    for i in range(count):
        source = rng.choice(multi_examples)

        # Take only the first 2 tool calls to keep it simple for the 3B model
        tool_calls = source.expected_tool_calls[:2]

        try:
            reinforced = ToolCallingExample(
                id=f"augmented_multi:{i}",
                system_prompt=source.system_prompt,
                user_query=source.user_query,
                available_tools=source.available_tools,
                expected_tool_calls=tool_calls,
                expected_response=source.expected_response,
                example_type=ExampleType.MULTI_TOOL,
                source_dataset="augmented",
            )
            reinforcement.append(reinforced)
        except Exception:
            continue

    console.print(f"  [green]Generated {len(reinforcement)} multi_tool reinforcement examples[/green]")
    return reinforcement


# ============================================================
# Main Augmentation Pipeline
# ============================================================


def run_augmentation(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/augmented",
    error_count: int = 400,
    no_tool_count: int = 300,
    multi_tool_count: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run the full data augmentation pipeline.

    Reads existing processed training data, generates synthetic examples
    for underrepresented types, and writes an augmented training dataset.

    The augmented data supplements (not replaces) the original training data.

    Args:
        processed_dir: Directory with processed train/val/test JSONL
        output_dir: Where to write augmented data
        error_count: Number of error_handling examples to generate
        no_tool_count: Number of additional no_tool examples
        multi_tool_count: Number of multi_tool reinforcement examples
        seed: Random seed for reproducibility

    Returns:
        Stats dict with counts per type
    """
    from rich.panel import Panel
    from rich.table import Table

    console.print(Panel(
        "[bold]ToolForge Data Augmentation[/bold]\n"
        "Generating synthetic training examples for underrepresented behaviors",
        title="Data Augmentation",
        border_style="yellow",
    ))

    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing training data
    train_path = processed_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run 'toolforge data prepare' first."
        )

    console.print(f"\n  Loading existing training data from {train_path}")
    existing = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                try:
                    existing.append(ToolCallingExample(**json.loads(line)))
                except Exception:
                    pass

    console.print(f"  Loaded {len(existing)} existing examples")

    # Count existing distribution
    from collections import Counter
    existing_dist = Counter(ex.example_type.value for ex in existing)
    console.print(f"  Current distribution: {dict(existing_dist)}")

    # Generate augmented examples
    console.print(f"\n[bold]Generating augmented examples:[/bold]")

    augmented = []

    # 1. Error handling (critical — zero in current training data)
    console.print(f"\n  [cyan]1. Error handling examples ({error_count} target)[/cyan]")
    error_examples = generate_error_handling_examples(existing, count=error_count, seed=seed)
    augmented.extend(error_examples)

    # 2. No-tool / relevance detection (severely underrepresented)
    console.print(f"\n  [cyan]2. No-tool examples ({no_tool_count} target)[/cyan]")
    no_tool_examples = generate_no_tool_examples(existing, count=no_tool_count, seed=seed)
    augmented.extend(no_tool_examples)

    # 3. Multi-tool reinforcement (model can't produce arrays)
    console.print(f"\n  [cyan]3. Multi-tool reinforcement ({multi_tool_count} target)[/cyan]")
    multi_examples = generate_multi_tool_reinforcement(existing, count=multi_tool_count, seed=seed)
    augmented.extend(multi_examples)

    # Combine with existing data
    combined = existing + augmented
    rng = random.Random(seed)
    rng.shuffle(combined)

    # Write augmented training data
    augmented_path = output_dir / "train_augmented.jsonl"
    with open(augmented_path, "w") as f:
        for ex in combined:
            f.write(ex.model_dump_json() + "\n")

    # Also write just the augmented examples (for analysis)
    aug_only_path = output_dir / "augmented_only.jsonl"
    with open(aug_only_path, "w") as f:
        for ex in augmented:
            f.write(ex.model_dump_json() + "\n")

    # Print summary
    new_dist = Counter(ex.example_type.value for ex in combined)

    table = Table(title="\nAugmented Dataset Distribution", show_lines=True)
    table.add_column("Type", style="cyan")
    table.add_column("Before", justify="right")
    table.add_column("Added", justify="right", style="green")
    table.add_column("After", justify="right", style="bold")
    table.add_column("% of Total", justify="right")

    total = len(combined)
    for etype in ["single_tool", "multi_tool", "no_tool", "error_handling"]:
        before = existing_dist.get(etype, 0)
        after = new_dist.get(etype, 0)
        added = after - before
        pct = (after / total * 100) if total > 0 else 0
        table.add_row(etype, str(before), f"+{added}", str(after), f"{pct:.1f}%")

    table.add_row("TOTAL", str(len(existing)), f"+{len(augmented)}", str(total), "100%")
    console.print(table)

    stats = {
        "original_count": len(existing),
        "augmented_count": len(augmented),
        "combined_count": len(combined),
        "error_handling_added": len(error_examples),
        "no_tool_added": len(no_tool_examples),
        "multi_tool_added": len(multi_examples),
        "output_path": str(augmented_path),
    }

    console.print(f"\n[bold green]Augmented data saved to {augmented_path}[/bold green]")
    console.print(f"  Combined: {len(combined)} examples ({len(augmented)} new)")

    return stats
