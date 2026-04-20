"""
Tests for the spec loader and validator.

WHY THESE TESTS MATTER:
  If the spec loader silently accepts invalid YAML, we'll run evaluations
  against broken specs and get meaningless results. These tests ensure
  the validation catches common mistakes before any GPU cycles are wasted.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from toolforge.eval.specs import (
    BehavioralSpec,
    Priority,
    load_all_specs,
    load_spec,
    validate_all_specs,
)


# ============================================================
# Fixtures: reusable test data
# ============================================================


@pytest.fixture
def valid_spec_dict() -> dict:
    """A minimal valid spec as a dictionary."""
    return {
        "name": "test_metric",
        "description": "A test spec for unit testing",
        "metric": "exact_match",
        "threshold": 0.90,
        "dataset": "data/eval/test.jsonl",
        "num_samples": 100,
        "priority": "high",
        "tags": ["test"],
        "baseline_expected": 0.50,
    }


@pytest.fixture
def valid_spec_file(valid_spec_dict, tmp_path) -> Path:
    """Write a valid spec to a temporary YAML file."""
    filepath = tmp_path / "test_spec.yaml"
    with open(filepath, "w") as f:
        yaml.dump(valid_spec_dict, f)
    return filepath


@pytest.fixture
def specs_dir_with_multiple(valid_spec_dict, tmp_path) -> Path:
    """Create a directory with multiple spec files at different priorities."""
    specs = [
        {**valid_spec_dict, "name": "critical_spec", "priority": "critical"},
        {**valid_spec_dict, "name": "high_spec", "priority": "high"},
        {**valid_spec_dict, "name": "medium_spec", "priority": "medium"},
    ]
    for spec in specs:
        filepath = tmp_path / f"{spec['name']}.yaml"
        with open(filepath, "w") as f:
            yaml.dump(spec, f)
    return tmp_path


# ============================================================
# Test: Loading valid specs
# ============================================================


class TestLoadSpec:
    def test_loads_valid_spec(self, valid_spec_file):
        """A well-formed YAML file should load without errors."""
        spec = load_spec(valid_spec_file)
        assert spec.name == "test_metric"
        assert spec.threshold == 0.90
        assert spec.priority == Priority.HIGH

    def test_spec_fields_are_typed(self, valid_spec_file):
        """Loaded spec should have correct Python types, not raw strings."""
        spec = load_spec(valid_spec_file)
        assert isinstance(spec.threshold, float)
        assert isinstance(spec.num_samples, int)
        assert isinstance(spec.tags, list)

    def test_file_not_found_raises(self):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_spec("/nonexistent/path/spec.yaml")


# ============================================================
# Test: Validation catches invalid specs
# ============================================================


class TestSpecValidation:
    def test_threshold_above_one_rejected(self, valid_spec_dict, tmp_path):
        """Threshold must be 0.0 to 1.0 — catch accidental '95' instead of '0.95'."""
        valid_spec_dict["threshold"] = 1.5
        filepath = tmp_path / "bad_spec.yaml"
        with open(filepath, "w") as f:
            yaml.dump(valid_spec_dict, f)
        with pytest.raises(Exception):  # Pydantic ValidationError
            load_spec(filepath)

    def test_threshold_negative_rejected(self, valid_spec_dict, tmp_path):
        """Negative thresholds make no sense."""
        valid_spec_dict["threshold"] = -0.1
        filepath = tmp_path / "bad_spec.yaml"
        with open(filepath, "w") as f:
            yaml.dump(valid_spec_dict, f)
        with pytest.raises(Exception):
            load_spec(filepath)

    def test_invalid_metric_name_rejected(self, valid_spec_dict, tmp_path):
        """Typos in metric names should be caught immediately."""
        valid_spec_dict["metric"] = "excat_match"  # Intentional typo
        filepath = tmp_path / "bad_spec.yaml"
        with open(filepath, "w") as f:
            yaml.dump(valid_spec_dict, f)
        with pytest.raises(Exception):
            load_spec(filepath)

    def test_invalid_priority_rejected(self, valid_spec_dict, tmp_path):
        """Priority must be critical/high/medium."""
        valid_spec_dict["priority"] = "urgent"  # Not a valid priority
        filepath = tmp_path / "bad_spec.yaml"
        with open(filepath, "w") as f:
            yaml.dump(valid_spec_dict, f)
        with pytest.raises(Exception):
            load_spec(filepath)

    def test_missing_required_field_rejected(self, valid_spec_dict, tmp_path):
        """Missing 'description' (required) should fail validation."""
        del valid_spec_dict["description"]
        filepath = tmp_path / "bad_spec.yaml"
        with open(filepath, "w") as f:
            yaml.dump(valid_spec_dict, f)
        with pytest.raises(Exception):
            load_spec(filepath)

    def test_zero_samples_rejected(self, valid_spec_dict, tmp_path):
        """num_samples must be positive."""
        valid_spec_dict["num_samples"] = 0
        filepath = tmp_path / "bad_spec.yaml"
        with open(filepath, "w") as f:
            yaml.dump(valid_spec_dict, f)
        with pytest.raises(Exception):
            load_spec(filepath)

    def test_non_snake_case_name_rejected(self, valid_spec_dict, tmp_path):
        """Spec names must be snake_case for consistency."""
        valid_spec_dict["name"] = "My Spec Name!"  # Has spaces and punctuation
        filepath = tmp_path / "bad_spec.yaml"
        with open(filepath, "w") as f:
            yaml.dump(valid_spec_dict, f)
        with pytest.raises(Exception):
            load_spec(filepath)


# ============================================================
# Test: Loading multiple specs
# ============================================================


class TestLoadAllSpecs:
    def test_loads_all_yaml_files(self, specs_dir_with_multiple):
        """Should find and load all .yaml files in the directory."""
        specs = load_all_specs(specs_dir_with_multiple)
        assert len(specs) == 3

    def test_sorted_by_priority(self, specs_dir_with_multiple):
        """Critical specs should appear first, then high, then medium."""
        specs = load_all_specs(specs_dir_with_multiple)
        priorities = [s.priority for s in specs]
        assert priorities == [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM]

    def test_nonexistent_dir_raises(self):
        """Loading from a nonexistent directory should raise."""
        with pytest.raises(NotADirectoryError):
            load_all_specs("/nonexistent/dir")

    def test_empty_dir_returns_empty_list(self, tmp_path):
        """A directory with no YAML files should return an empty list."""
        specs = load_all_specs(tmp_path)
        assert specs == []

    def test_skips_underscore_prefixed_files(self, valid_spec_dict, tmp_path):
        """Files starting with _ should be skipped (convention for disabled specs)."""
        # Write a normal spec
        filepath = tmp_path / "normal_spec.yaml"
        with open(filepath, "w") as f:
            yaml.dump(valid_spec_dict, f)

        # Write a disabled spec (underscore prefix)
        disabled = tmp_path / "_disabled_spec.yaml"
        with open(disabled, "w") as f:
            yaml.dump({**valid_spec_dict, "name": "disabled_spec"}, f)

        specs = load_all_specs(tmp_path)
        assert len(specs) == 1
        assert specs[0].name == "test_metric"
