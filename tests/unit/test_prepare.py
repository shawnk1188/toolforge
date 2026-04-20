"""
Tests for the data preparation module.

WHY THESE TESTS MATTER:
  The prepare module is the final step before training — if stratified splitting
  drops examples, eval datasets are malformed, or manifests have wrong counts,
  we get silently corrupted training data. These tests ensure every example is
  accounted for and every output file is well-formed.
"""

import json
from pathlib import Path

import pytest

from toolforge.data.prepare import (
    _synthesize_error_examples,
    generate_eval_datasets,
    save_splits,
    stratified_split,
)
from toolforge.data.schema import (
    ExampleType,
    ToolCall,
    ToolCallingExample,
    ToolDefinition,
    ToolParameters,
)


# ============================================================
# Helpers
# ============================================================


def _make_examples(n=50, types=None):
    """Create n test examples with mixed types."""
    types = types or [ExampleType.SINGLE_TOOL, ExampleType.NO_TOOL, ExampleType.MULTI_TOOL]
    examples = []
    for i in range(n):
        etype = types[i % len(types)]
        if etype == ExampleType.NO_TOOL:
            examples.append(ToolCallingExample(
                id=f"test:{i}",
                user_query=f"Query number {i} for testing purposes",
                available_tools=[ToolDefinition(name="get_weather", description="Get weather", parameters=ToolParameters())],
                expected_tool_calls=[],
                expected_response="No tool needed.",
                example_type=etype,
                source_dataset="test",
            ))
        else:
            examples.append(ToolCallingExample(
                id=f"test:{i}",
                user_query=f"Query number {i} for testing purposes",
                available_tools=[ToolDefinition(name="get_weather", description="Get weather", parameters=ToolParameters())],
                expected_tool_calls=[ToolCall(name="get_weather", arguments={"city": "SF"})],
                example_type=etype,
                source_dataset="test",
            ))
    return examples


# ============================================================
# Tests: stratified_split
# ============================================================


class TestStratifiedSplit:
    """Tests for the stratified_split function."""

    def test_ratios_must_sum_to_one(self):
        """Ratios that do not sum to 1.0 should raise an AssertionError."""
        examples = _make_examples(30)
        with pytest.raises(AssertionError, match="Ratios must sum to 1.0"):
            stratified_split(examples, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_each_split_gets_at_least_one_per_type(self):
        """Every example type should appear at least once in every split."""
        examples = _make_examples(60)
        splits = stratified_split(examples)

        for split_name, split_examples in splits.items():
            types_in_split = {ex.example_type for ex in split_examples}
            for etype in [ExampleType.SINGLE_TOOL, ExampleType.NO_TOOL, ExampleType.MULTI_TOOL]:
                assert etype in types_in_split, (
                    f"{etype} missing from {split_name} split"
                )

    def test_deterministic_with_same_seed(self):
        """Two calls with the same seed must produce identical splits."""
        examples = _make_examples(60)
        splits_a = stratified_split(examples, seed=123)
        splits_b = stratified_split(examples, seed=123)

        for key in ("train", "val", "test"):
            ids_a = [ex.id for ex in splits_a[key]]
            ids_b = [ex.id for ex in splits_b[key]]
            assert ids_a == ids_b, f"Split '{key}' differs between runs with same seed"

    def test_different_seed_produces_different_order(self):
        """Different seeds should (very likely) produce different orderings."""
        examples = _make_examples(60)
        splits_a = stratified_split(examples, seed=1)
        splits_b = stratified_split(examples, seed=999)

        ids_a = [ex.id for ex in splits_a["train"]]
        ids_b = [ex.id for ex in splits_b["train"]]
        assert ids_a != ids_b, "Different seeds produced identical train splits"

    def test_no_data_loss(self):
        """All input example IDs must appear in exactly one output split."""
        examples = _make_examples(60)
        splits = stratified_split(examples)

        output_ids = []
        for split_examples in splits.values():
            output_ids.extend(ex.id for ex in split_examples)

        input_ids = {ex.id for ex in examples}
        output_id_set = set(output_ids)

        # Every input appears in output (no loss)
        assert input_ids.issubset(output_id_set), "Some examples were lost during splitting"

    def test_types_represented_in_each_split(self):
        """Each split should contain examples from all provided types."""
        types = [ExampleType.SINGLE_TOOL, ExampleType.MULTI_TOOL]
        examples = _make_examples(40, types=types)
        splits = stratified_split(examples)

        for split_name, split_examples in splits.items():
            types_present = {ex.example_type for ex in split_examples}
            for etype in types:
                assert etype in types_present, (
                    f"{etype} missing from {split_name} split"
                )

    def test_few_examples_per_type_still_works(self):
        """Even with very few examples per type, splitting should not crash."""
        # 3 examples, one of each type
        examples = _make_examples(3)
        splits = stratified_split(examples)

        for key in ("train", "val", "test"):
            assert len(splits[key]) >= 1, f"Split '{key}' is empty"


# ============================================================
# Tests: generate_eval_datasets
# ============================================================


class TestGenerateEvalDatasets:
    """Tests for the generate_eval_datasets function."""

    def test_creates_expected_files(self, tmp_path):
        """All expected .jsonl eval files should be created."""
        examples = _make_examples(60)
        generate_eval_datasets(examples, output_dir=tmp_path)

        expected_files = [
            "tool_selection_test.jsonl",
            "argument_accuracy_test.jsonl",
            "hallucination_test.jsonl",
            "no_tool_needed_test.jsonl",
            "multi_tool_test.jsonl",
            "error_recovery_test.jsonl",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing eval file: {fname}"

    def test_returns_stats_with_correct_keys(self, tmp_path):
        """Returned stats dict should have a key for each eval dataset."""
        examples = _make_examples(60)
        stats = generate_eval_datasets(examples, output_dir=tmp_path)

        expected_keys = {
            "tool_selection_test",
            "argument_accuracy_test",
            "hallucination_test",
            "no_tool_needed_test",
            "multi_tool_test",
            "error_recovery_test",
        }
        assert set(stats.keys()) == expected_keys

    def test_stats_values_are_nonnegative_ints(self, tmp_path):
        """Every stat value should be a non-negative integer."""
        examples = _make_examples(60)
        stats = generate_eval_datasets(examples, output_dir=tmp_path)

        for key, count in stats.items():
            assert isinstance(count, int), f"stats['{key}'] is not int"
            assert count >= 0, f"stats['{key}'] is negative"

    def test_files_contain_valid_json_lines(self, tmp_path):
        """Each line in every output file should be valid JSON."""
        examples = _make_examples(60)
        generate_eval_datasets(examples, output_dir=tmp_path)

        for jsonl_file in tmp_path.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            pytest.fail(f"Invalid JSON at {jsonl_file.name}:{lineno}")

    def test_handles_empty_input(self, tmp_path):
        """Empty input should create files with 0 examples, not crash."""
        stats = generate_eval_datasets([], output_dir=tmp_path)

        for key, count in stats.items():
            assert count == 0, f"Expected 0 for '{key}' with empty input, got {count}"

    def test_file_line_counts_match_stats(self, tmp_path):
        """The number of lines in each file should match the reported stat."""
        examples = _make_examples(60)
        stats = generate_eval_datasets(examples, output_dir=tmp_path)

        for name, expected_count in stats.items():
            filepath = tmp_path / f"{name}.jsonl"
            with open(filepath) as f:
                actual_count = sum(1 for line in f if line.strip())
            assert actual_count == expected_count, (
                f"{name}: file has {actual_count} lines but stats says {expected_count}"
            )


# ============================================================
# Tests: save_splits
# ============================================================


class TestSaveSplits:
    """Tests for the save_splits function."""

    def test_creates_split_files(self, tmp_path):
        """train.jsonl, val.jsonl, and test.jsonl should be created."""
        examples = _make_examples(60)
        splits = stratified_split(examples)
        save_splits(splits, output_dir=tmp_path)

        for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
            assert (tmp_path / fname).exists(), f"Missing split file: {fname}"

    def test_creates_manifest(self, tmp_path):
        """manifest.json should be created in the output directory."""
        examples = _make_examples(60)
        splits = stratified_split(examples)
        save_splits(splits, output_dir=tmp_path)

        assert (tmp_path / "manifest.json").exists(), "Missing manifest.json"

    def test_manifest_has_correct_total(self, tmp_path):
        """Manifest total_examples should equal the sum of all split sizes."""
        examples = _make_examples(60)
        splits = stratified_split(examples)
        manifest = save_splits(splits, output_dir=tmp_path)

        total_in_splits = sum(len(v) for v in splits.values())
        assert manifest.total_examples == total_in_splits

    def test_manifest_json_is_valid(self, tmp_path):
        """The manifest.json file should be parseable JSON with expected fields."""
        examples = _make_examples(60)
        splits = stratified_split(examples)
        save_splits(splits, output_dir=tmp_path)

        with open(tmp_path / "manifest.json") as f:
            data = json.load(f)

        assert "total_examples" in data
        assert "splits" in data
        assert "processing_timestamp" in data

    def test_returns_dataset_manifest(self, tmp_path):
        """save_splits should return a DatasetManifest object."""
        from toolforge.data.schema import DatasetManifest

        examples = _make_examples(60)
        splits = stratified_split(examples)
        result = save_splits(splits, output_dir=tmp_path)

        assert isinstance(result, DatasetManifest)

    def test_split_files_have_correct_line_counts(self, tmp_path):
        """Each .jsonl file should have exactly as many lines as its split."""
        examples = _make_examples(60)
        splits = stratified_split(examples)
        save_splits(splits, output_dir=tmp_path)

        for split_name, split_examples in splits.items():
            filepath = tmp_path / f"{split_name}.jsonl"
            with open(filepath) as f:
                line_count = sum(1 for line in f if line.strip())
            assert line_count == len(split_examples), (
                f"{split_name}.jsonl has {line_count} lines, expected {len(split_examples)}"
            )

    def test_source_datasets_in_manifest(self, tmp_path):
        """Provided source_datasets list should appear in the manifest."""
        examples = _make_examples(30)
        splits = stratified_split(examples)
        manifest = save_splits(splits, output_dir=tmp_path, source_datasets=["ds_a", "ds_b"])

        assert manifest.source_datasets == ["ds_a", "ds_b"]


# ============================================================
# Tests: _synthesize_error_examples
# ============================================================


class TestSynthesizeErrorExamples:
    """Tests for the _synthesize_error_examples function."""

    def test_generates_error_examples(self):
        """Should produce error examples from source examples with tool calls."""
        import random

        source = _make_examples(20, types=[ExampleType.SINGLE_TOOL])
        rng = random.Random(42)
        result = _synthesize_error_examples(source, rng)

        assert len(result) > 0, "No error examples generated"

    def test_error_examples_have_error_handling_type(self):
        """Every synthesized example should have ERROR_HANDLING type."""
        import random

        source = _make_examples(20, types=[ExampleType.SINGLE_TOOL])
        rng = random.Random(42)
        result = _synthesize_error_examples(source, rng)

        for ex in result:
            assert ex.example_type == ExampleType.ERROR_HANDLING

    def test_error_examples_have_empty_tool_calls(self):
        """Error examples should have no expected tool calls (model should not retry)."""
        import random

        source = _make_examples(20, types=[ExampleType.SINGLE_TOOL])
        rng = random.Random(42)
        result = _synthesize_error_examples(source, rng)

        for ex in result:
            assert ex.expected_tool_calls == [], (
                f"Error example {ex.id} has non-empty expected_tool_calls"
            )

    def test_max_200_examples(self):
        """Should return at most 200 examples regardless of input size."""
        import random

        source = _make_examples(300, types=[ExampleType.SINGLE_TOOL])
        rng = random.Random(42)
        result = _synthesize_error_examples(source, rng)

        assert len(result) <= 200

    def test_skips_no_tool_examples(self):
        """Source examples without tool calls should be skipped."""
        import random

        source = _make_examples(10, types=[ExampleType.NO_TOOL])
        rng = random.Random(42)
        result = _synthesize_error_examples(source, rng)

        assert len(result) == 0, "Should not synthesize errors from NO_TOOL examples"

    def test_error_query_contains_error_context(self):
        """The synthesized query should include an error message context."""
        import random

        source = _make_examples(5, types=[ExampleType.SINGLE_TOOL])
        rng = random.Random(42)
        result = _synthesize_error_examples(source, rng)

        for ex in result:
            assert "returned an error" in ex.user_query, (
                f"Error context missing from query of {ex.id}"
            )

    def test_error_examples_have_expected_response(self):
        """Each error example should have a non-empty expected_response."""
        import random

        source = _make_examples(10, types=[ExampleType.SINGLE_TOOL])
        rng = random.Random(42)
        result = _synthesize_error_examples(source, rng)

        for ex in result:
            assert ex.expected_response, f"Error example {ex.id} has no expected_response"
