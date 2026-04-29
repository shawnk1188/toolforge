"""
Tests for DPO configuration, loss function, and preference pair utilities.

WHY THESE TESTS MATTER:
  A misconfigured DPO run can produce a model that's WORSE than SFT.
  Common failure modes:
    - Wrong beta → model ignores preferences or forgets SFT skills
    - Wrong log prob computation → gradients are garbage
    - Wrong loss → reward hacking, model games the metric
  These tests catch these issues before spending GPU hours.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
import yaml

from toolforge.training.dpo import (
    DPOConfig,
    DPODataset,
    collate_dpo_batch,
    dpo_loss,
    _clip_grad_norm,
    _compute_grad_norm,
)


# ============================================================
# Tests: DPOConfig defaults
# ============================================================


class TestDPOConfigDefaults:
    """Verify sensible defaults."""

    def test_default_beta(self):
        config = DPOConfig()
        assert config.beta == 0.1

    def test_default_learning_rate(self):
        config = DPOConfig()
        assert config.learning_rate == 5e-6

    def test_default_iters(self):
        config = DPOConfig()
        assert config.iters == 500

    def test_default_batch_size(self):
        config = DPOConfig()
        assert config.batch_size == 1

    def test_default_label_smoothing(self):
        config = DPOConfig()
        assert config.label_smoothing == 0.0

    def test_default_grad_checkpoint(self):
        config = DPOConfig()
        assert config.grad_checkpoint is True

    def test_default_sft_adapter_path(self):
        config = DPOConfig()
        assert config.sft_adapter_path == "artifacts/sft/adapters"

    def test_default_adapter_path(self):
        config = DPOConfig()
        assert config.adapter_path == "artifacts/dpo/adapters"

    def test_default_lora_rank(self):
        config = DPOConfig()
        assert config.lora_rank == 8

    def test_default_grad_clip_norm(self):
        config = DPOConfig()
        assert config.grad_clip_norm == 1.0

    def test_default_early_stopping_patience(self):
        config = DPOConfig()
        assert config.early_stopping_patience == 50


# ============================================================
# Tests: DPOConfig.from_yaml
# ============================================================


class TestDPOConfigFromYaml:
    """Tests for YAML config loading."""

    def test_loads_from_yaml(self, tmp_path):
        config_data = {
            "beta": 0.2,
            "iters": 300,
            "learning_rate": 1e-5,
            "lora_parameters": {"rank": 16, "dropout": 0.1, "scale": 32.0},
        }
        yaml_path = tmp_path / "dpo.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = DPOConfig.from_yaml(yaml_path)
        assert config.beta == 0.2
        assert config.iters == 300
        assert config.learning_rate == 1e-5
        assert config.lora_rank == 16

    def test_missing_yaml_returns_defaults(self, tmp_path):
        config = DPOConfig.from_yaml(tmp_path / "nonexistent.yaml")
        assert config.beta == 0.1
        assert config.iters == 500

    def test_partial_yaml_fills_defaults(self, tmp_path):
        config_data = {"beta": 0.05}
        yaml_path = tmp_path / "dpo.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = DPOConfig.from_yaml(yaml_path)
        assert config.beta == 0.05
        assert config.iters == 500  # default

    def test_production_config_loads(self):
        path = Path("configs/training/dpo.yaml")
        if not path.exists():
            pytest.skip("Production DPO config not found")
        config = DPOConfig.from_yaml(path)
        # v2 config has more conservative settings
        assert config.beta == 0.5
        assert config.learning_rate == 1e-6
        assert config.grad_clip_norm == 1.0
        assert config.early_stopping_patience == 50


# ============================================================
# Tests: DPO Loss
# ============================================================


class TestDPOLoss:
    """Tests for the DPO loss function."""

    def test_loss_is_scalar(self):
        """DPO loss should be a scalar."""
        import mlx.core as mx

        loss, metrics = dpo_loss(
            policy_chosen_logps=mx.array([-1.0, -2.0]),
            policy_rejected_logps=mx.array([-3.0, -4.0]),
            ref_chosen_logps=mx.array([-1.5, -2.5]),
            ref_rejected_logps=mx.array([-3.5, -4.5]),
            beta=0.1,
        )
        assert loss.shape == ()

    def test_loss_when_policy_prefers_chosen(self):
        """
        When policy strongly prefers chosen over rejected,
        loss should be LOW (this is the desired state).
        """
        import mlx.core as mx

        # Policy assigns much higher log prob to chosen than rejected
        loss, metrics = dpo_loss(
            policy_chosen_logps=mx.array([-0.5]),   # high prob chosen
            policy_rejected_logps=mx.array([-5.0]),  # low prob rejected
            ref_chosen_logps=mx.array([-1.0]),
            ref_rejected_logps=mx.array([-1.0]),
            beta=0.1,
        )
        # Loss should be small (model already prefers chosen)
        assert metrics["loss"] < 1.0
        assert metrics["reward_margin"] > 0

    def test_loss_when_policy_prefers_rejected(self):
        """
        When policy prefers rejected, loss should be HIGH
        (this is what we're trying to fix).
        """
        import mlx.core as mx

        # Policy assigns higher log prob to rejected (bad!)
        loss, metrics = dpo_loss(
            policy_chosen_logps=mx.array([-5.0]),    # low prob chosen
            policy_rejected_logps=mx.array([-0.5]),   # high prob rejected
            ref_chosen_logps=mx.array([-1.0]),
            ref_rejected_logps=mx.array([-1.0]),
            beta=0.1,
        )
        # Margin should be negative (model prefers wrong answer)
        assert metrics["reward_margin"] < 0
        assert metrics["accuracy"] < 0.5

    def test_accuracy_metric(self):
        """Accuracy = fraction where reward margin > 0."""
        import mlx.core as mx

        _, metrics = dpo_loss(
            policy_chosen_logps=mx.array([-1.0, -5.0]),
            policy_rejected_logps=mx.array([-5.0, -1.0]),
            ref_chosen_logps=mx.array([-2.0, -2.0]),
            ref_rejected_logps=mx.array([-2.0, -2.0]),
            beta=0.1,
        )
        # First example: policy prefers chosen (correct)
        # Second example: policy prefers rejected (wrong)
        assert metrics["accuracy"] == 0.5

    def test_beta_controls_loss_scale(self):
        """Higher beta should amplify the loss signal."""
        import mlx.core as mx

        args = dict(
            policy_chosen_logps=mx.array([-2.0]),
            policy_rejected_logps=mx.array([-3.0]),
            ref_chosen_logps=mx.array([-2.5]),
            ref_rejected_logps=mx.array([-2.5]),
        )

        _, metrics_low = dpo_loss(**args, beta=0.01)
        _, metrics_high = dpo_loss(**args, beta=1.0)

        # Higher beta → larger absolute reward margin
        assert abs(metrics_high["reward_margin"]) > abs(metrics_low["reward_margin"])

    def test_label_smoothing(self):
        """Label smoothing > 0 should produce different loss than standard DPO."""
        import mlx.core as mx

        args = dict(
            policy_chosen_logps=mx.array([-1.0]),
            policy_rejected_logps=mx.array([-3.0]),
            ref_chosen_logps=mx.array([-2.0]),
            ref_rejected_logps=mx.array([-2.0]),
            beta=0.1,
        )

        _, m_standard = dpo_loss(**args, label_smoothing=0.0)
        _, m_smooth = dpo_loss(**args, label_smoothing=0.1)

        # Label smoothing should increase the loss slightly
        # (it adds a penalty for being TOO confident in preferences)
        assert m_standard["loss"] != m_smooth["loss"]

    def test_returns_all_metrics(self):
        """Metrics dict should have all expected keys."""
        import mlx.core as mx

        _, metrics = dpo_loss(
            policy_chosen_logps=mx.array([-1.0]),
            policy_rejected_logps=mx.array([-2.0]),
            ref_chosen_logps=mx.array([-1.5]),
            ref_rejected_logps=mx.array([-1.5]),
            beta=0.1,
        )
        assert "loss" in metrics
        assert "chosen_reward" in metrics
        assert "rejected_reward" in metrics
        assert "reward_margin" in metrics
        assert "accuracy" in metrics


# ============================================================
# Tests: collate_dpo_batch
# ============================================================


class TestCollateDpoBatch:
    """Tests for batch collation with padding."""

    def test_pads_to_max_length(self):
        """Sequences should be padded to the longest in the batch."""
        import mlx.core as mx

        batch = [
            {
                "chosen_ids": mx.array([1, 2, 3]),
                "rejected_ids": mx.array([4, 5]),
                "chosen_length": 3,
                "rejected_length": 2,
            },
            {
                "chosen_ids": mx.array([6, 7]),
                "rejected_ids": mx.array([8, 9, 10, 11]),
                "chosen_length": 2,
                "rejected_length": 4,
            },
        ]

        result = collate_dpo_batch(batch)
        assert result["chosen_ids"].shape == (2, 3)    # max chosen length
        assert result["rejected_ids"].shape == (2, 4)   # max rejected length

    def test_preserves_lengths(self):
        """Original lengths should be preserved (for masking)."""
        import mlx.core as mx

        batch = [
            {
                "chosen_ids": mx.array([1, 2, 3]),
                "rejected_ids": mx.array([4, 5]),
                "chosen_length": 3,
                "rejected_length": 2,
            },
        ]

        result = collate_dpo_batch(batch)
        assert result["chosen_lengths"].tolist() == [3]
        assert result["rejected_lengths"].tolist() == [2]

    def test_single_item_batch(self):
        """Single-item batch should work without padding."""
        import mlx.core as mx

        batch = [
            {
                "chosen_ids": mx.array([1, 2]),
                "rejected_ids": mx.array([3, 4, 5]),
                "chosen_length": 2,
                "rejected_length": 3,
            },
        ]

        result = collate_dpo_batch(batch)
        assert result["chosen_ids"].shape == (1, 2)
        assert result["rejected_ids"].shape == (1, 3)


# ============================================================
# Tests: Preference pair format
# ============================================================


class TestPreferencePairFormat:
    """Tests for the preference pair JSONL format."""

    def test_pair_has_required_fields(self, tmp_path):
        """Each preference pair must have prompt, chosen, rejected."""
        pair = {
            "prompt": "What's the weather?",
            "chosen": '{"name": "get_weather", "arguments": {"city": "NYC"}}',
            "rejected": '{"name": "unknown_tool", "arguments": {}}',
            "spec": "tool_selection",
        }

        path = tmp_path / "pairs.jsonl"
        path.write_text(json.dumps(pair) + "\n")

        # Verify it loads
        with open(path) as f:
            loaded = json.loads(f.readline())
        assert "prompt" in loaded
        assert "chosen" in loaded
        assert "rejected" in loaded

    def test_chosen_is_ground_truth(self, tmp_path):
        """Chosen should be the correct answer from ground truth."""
        pair = {
            "prompt": "Tell me a joke",
            "chosen": "Sure! Why did the scarecrow win an award? He was outstanding in his field!",
            "rejected": '{"name": "search_jokes", "arguments": {"topic": "general"}}',
            "spec": "relevance_detection",
        }

        # The key insight: for relevance_detection, chosen is TEXT (not a tool call)
        # and rejected IS a tool call (which is wrong — model shouldn't call tools here)
        assert not pair["chosen"].startswith("{")  # text response
        assert pair["rejected"].startswith("{")  # unwanted tool call


# ============================================================
# Tests: Gradient Clipping
# ============================================================


class TestGradientClipping:
    """Tests for gradient clipping to prevent catastrophic forgetting."""

    def test_compute_grad_norm_simple(self):
        """Compute norm of a simple gradient tree."""
        import mlx.core as mx

        grads = {"weight": mx.array([3.0, 4.0])}
        norm = _compute_grad_norm(grads)
        assert abs(norm - 5.0) < 1e-5  # sqrt(9+16) = 5

    def test_compute_grad_norm_nested(self):
        """Compute norm across nested gradient tree."""
        import mlx.core as mx

        grads = {
            "layer1": {"weight": mx.array([1.0, 0.0])},
            "layer2": {"weight": mx.array([0.0, 1.0])},
        }
        norm = _compute_grad_norm(grads)
        assert abs(norm - math.sqrt(2.0)) < 1e-5

    def test_clip_does_nothing_under_threshold(self):
        """Gradient with norm below max_norm should be unchanged."""
        import mlx.core as mx

        grads = {"weight": mx.array([0.1, 0.1])}
        clipped = _clip_grad_norm(grads, max_norm=10.0)
        assert clipped["weight"].tolist() == pytest.approx([0.1, 0.1], abs=1e-5)

    def test_clip_scales_down_large_gradients(self):
        """Gradient with norm above max_norm should be scaled down."""
        import mlx.core as mx

        grads = {"weight": mx.array([3.0, 4.0])}  # norm = 5.0
        clipped = _clip_grad_norm(grads, max_norm=1.0)
        clipped_norm = _compute_grad_norm(clipped)
        assert abs(clipped_norm - 1.0) < 1e-4

    def test_clip_preserves_direction(self):
        """Clipping should scale uniformly — gradient direction is preserved."""
        import mlx.core as mx

        grads = {"w": mx.array([6.0, 8.0])}  # norm = 10, ratio 3:4
        clipped = _clip_grad_norm(grads, max_norm=5.0)
        vals = clipped["w"].tolist()
        # Should still have 3:4 ratio
        assert abs(vals[0] / vals[1] - 0.75) < 1e-5

    def test_clip_handles_nested_tree(self):
        """Clipping should work on nested gradient trees."""
        import mlx.core as mx

        grads = {
            "layers": [
                {"weight": mx.array([3.0, 0.0])},
                {"weight": mx.array([0.0, 4.0])},
            ]
        }
        clipped = _clip_grad_norm(grads, max_norm=1.0)
        clipped_norm = _compute_grad_norm(clipped)
        assert abs(clipped_norm - 1.0) < 1e-4

    def test_clip_zero_gradient(self):
        """Zero gradients should remain zero after clipping."""
        import mlx.core as mx

        grads = {"weight": mx.array([0.0, 0.0])}
        clipped = _clip_grad_norm(grads, max_norm=1.0)
        assert clipped["weight"].tolist() == [0.0, 0.0]
