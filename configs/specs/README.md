# Behavioral Specs

Each YAML file in this directory defines a **behavioral specification** — a measurable
quality standard that the fine-tuned model must meet.

## Spec Schema

```yaml
name: string           # Unique identifier for this spec
description: string    # What this spec measures (human-readable)
metric: string         # Metric function name (from toolforge.eval.metrics)
threshold: float       # Minimum score to PASS (0.0 to 1.0)
dataset: string        # Path to evaluation dataset for this spec
num_samples: int       # Number of samples to evaluate (for speed)
priority: string       # critical | high | medium (determines training focus)
tags: list[string]     # Categories for grouping specs in reports
```

## How Specs Are Used

1. **Before training**: Run all specs against the base model → all FAIL (expected)
2. **After SFT**: Run again → some PASS, some still FAIL
3. **After DPO**: Target remaining failures → all PASS
4. **In CI**: Specs run as a gate — if any regress, the PR is blocked

## Current Specs

| Spec | Metric | Threshold | Priority |
|------|--------|-----------|----------|
| tool_selection | exact_match | 0.95 | critical |
| argument_accuracy | json_schema_match | 0.90 | critical |
| hallucination_resistance | tool_exists_check | 0.99 | critical |
| relevance_detection | correct_refusal_rate | 0.92 | high |
| multi_tool_sequencing | sequence_exact_match | 0.85 | high |
| error_recovery | graceful_error_rate | 0.88 | medium |
