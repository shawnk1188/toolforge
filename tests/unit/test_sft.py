"""
Tests for SFT training configuration and pipeline orchestration.

WHY THESE TESTS MATTER:
  A misconfigured training run wastes 4+ hours of GPU time. These tests
  verify that config loading, YAML parsing, and mlx-lm config conversion
  all work correctly BEFORE spending compute.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from toolforge.training.sft import SFTConfig


# ============================================================
# Tests: SFTConfig defaults
# ============================================================


class TestSFTConfigDefaults:
    """Verify sensible defaults when no YAML is provided."""

    def test_default_model(self):
        config = SFTConfig()
        assert config.model == "mlx-community/Llama-3.2-3B-Instruct-4bit"

    def test_default_lora_rank(self):
        config = SFTConfig()
        assert config.lora_rank == 8

    def test_default_lora_dropout(self):
        config = SFTConfig()
        assert config.lora_dropout == 0.05

    def test_default_lora_scale(self):
        config = SFTConfig()
        assert config.lora_scale == 20.0

    def test_default_iters(self):
        config = SFTConfig()
        assert config.iters == 1000

    def test_default_batch_size(self):
        config = SFTConfig()
        assert config.batch_size == 2

    def test_default_learning_rate(self):
        config = SFTConfig()
        assert config.learning_rate == 1e-5

    def test_default_max_seq_length(self):
        config = SFTConfig()
        assert config.max_seq_length == 2048

    def test_default_grad_checkpoint(self):
        config = SFTConfig()
        assert config.grad_checkpoint is True

    def test_default_mask_prompt(self):
        config = SFTConfig()
        assert config.mask_prompt is True

    def test_default_seed(self):
        config = SFTConfig()
        assert config.seed == 42

    def test_default_num_layers(self):
        config = SFTConfig()
        assert config.num_layers == 16

    def test_default_lr_schedule_none(self):
        config = SFTConfig()
        assert config.lr_schedule is None

    def test_default_save_every(self):
        config = SFTConfig()
        assert config.save_every == 200

    def test_default_val_batches(self):
        config = SFTConfig()
        assert config.val_batches == 10


# ============================================================
# Tests: SFTConfig.from_yaml
# ============================================================


class TestSFTConfigFromYaml:
    """Tests for YAML config loading."""

    def test_loads_from_yaml(self, tmp_path):
        """YAML values should override defaults."""
        config_data = {
            "model": "custom-model/test",
            "data": "data/custom",
            "iters": 500,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "lora_parameters": {
                "rank": 16,
                "dropout": 0.1,
                "scale": 32.0,
            },
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = SFTConfig.from_yaml(yaml_path)
        assert config.model == "custom-model/test"
        assert config.data_dir == "data/custom"
        assert config.iters == 500
        assert config.batch_size == 4
        assert config.learning_rate == 5e-5
        assert config.lora_rank == 16
        assert config.lora_dropout == 0.1
        assert config.lora_scale == 32.0

    def test_missing_yaml_returns_defaults(self, tmp_path):
        """If the YAML file doesn't exist, return all defaults."""
        config = SFTConfig.from_yaml(tmp_path / "nonexistent.yaml")
        assert config.model == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert config.iters == 1000

    def test_partial_yaml_fills_defaults(self, tmp_path):
        """YAML with only some fields should fill in defaults for the rest."""
        config_data = {"iters": 200}
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = SFTConfig.from_yaml(yaml_path)
        assert config.iters == 200  # overridden
        assert config.batch_size == 2  # default
        assert config.lora_rank == 8  # default

    def test_empty_yaml_returns_defaults(self, tmp_path):
        """Empty YAML file should not crash — return defaults."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("")

        config = SFTConfig.from_yaml(yaml_path)
        assert config.iters == 1000

    def test_lr_schedule_loaded_as_dict(self, tmp_path):
        """lr_schedule should be loaded as a raw dict for mlx-lm."""
        config_data = {
            "lr_schedule": {
                "name": "cosine_decay",
                "arguments": [1e-5, 1000],
                "warmup": 50,
                "warmup_init": 1e-7,
            }
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = SFTConfig.from_yaml(yaml_path)
        assert isinstance(config.lr_schedule, dict)
        assert config.lr_schedule["name"] == "cosine_decay"
        assert config.lr_schedule["arguments"] == [1e-5, 1000]
        assert config.lr_schedule["warmup"] == 50

    def test_adapter_path_loaded(self, tmp_path):
        """adapter_path should be loaded from YAML."""
        config_data = {"adapter_path": "my/custom/adapters"}
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        config = SFTConfig.from_yaml(yaml_path)
        assert config.adapter_path == "my/custom/adapters"


# ============================================================
# Tests: SFTConfig.to_mlx_config
# ============================================================


class TestSFTConfigToMlxConfig:
    """Tests for conversion to mlx-lm config format."""

    def test_has_required_mlx_fields(self):
        """MLX config must have all fields mlx_lm.lora expects."""
        config = SFTConfig()
        mlx = config.to_mlx_config()

        required = [
            "model", "data", "train", "fine_tune_type", "seed",
            "num_layers", "batch_size", "iters", "learning_rate",
            "steps_per_report", "steps_per_eval", "save_every",
            "val_batches", "max_seq_length", "grad_checkpoint",
            "mask_prompt", "adapter_path", "lora_parameters",
        ]
        for key in required:
            assert key in mlx, f"Missing key: {key}"

    def test_train_is_true(self):
        """MLX config should always set train=True."""
        mlx = SFTConfig().to_mlx_config()
        assert mlx["train"] is True

    def test_fine_tune_type_is_lora(self):
        """Fine-tune type should be lora."""
        mlx = SFTConfig().to_mlx_config()
        assert mlx["fine_tune_type"] == "lora"

    def test_lora_parameters_nested(self):
        """LoRA params should be nested under lora_parameters key."""
        config = SFTConfig(lora_rank=16, lora_dropout=0.1, lora_scale=32.0)
        mlx = config.to_mlx_config()
        lora = mlx["lora_parameters"]
        assert lora["rank"] == 16
        assert lora["dropout"] == 0.1
        assert lora["scale"] == 32.0

    def test_values_propagated(self):
        """Config values should propagate to mlx config."""
        config = SFTConfig(
            iters=500,
            batch_size=4,
            learning_rate=5e-5,
            max_seq_length=4096,
        )
        mlx = config.to_mlx_config()
        assert mlx["iters"] == 500
        assert mlx["batch_size"] == 4
        assert mlx["learning_rate"] == 5e-5
        assert mlx["max_seq_length"] == 4096

    def test_lr_schedule_passed_through(self):
        """lr_schedule dict should be passed through as-is."""
        schedule = {
            "name": "cosine_decay",
            "arguments": [1e-5, 1000],
            "warmup": 50,
            "warmup_init": 1e-7,
        }
        config = SFTConfig(lr_schedule=schedule)
        mlx = config.to_mlx_config()
        assert mlx["lr_schedule"] == schedule

    def test_lr_schedule_none_when_not_set(self):
        """lr_schedule should be None when not configured."""
        config = SFTConfig()
        mlx = config.to_mlx_config()
        assert mlx["lr_schedule"] is None

    def test_data_dir_maps_to_data(self):
        """Our data_dir field maps to mlx-lm's 'data' key."""
        config = SFTConfig(data_dir="data/custom_mlx")
        mlx = config.to_mlx_config()
        assert mlx["data"] == "data/custom_mlx"

    def test_grad_checkpoint_propagated(self):
        """grad_checkpoint should be propagated."""
        config = SFTConfig(grad_checkpoint=True)
        mlx = config.to_mlx_config()
        assert mlx["grad_checkpoint"] is True


# ============================================================
# Tests: Production config file
# ============================================================


class TestProductionConfig:
    """Verify the actual production sft.yaml loads correctly."""

    @pytest.fixture
    def prod_config(self):
        """Load the production config if it exists."""
        path = Path("configs/training/sft.yaml")
        if not path.exists():
            pytest.skip("Production config not found")
        return SFTConfig.from_yaml(path)

    def test_prod_config_loads(self, prod_config):
        """Production config should load without errors."""
        assert prod_config is not None

    def test_prod_model_is_quantized(self, prod_config):
        """Production model should be 4-bit quantized (for M1 memory)."""
        assert "4bit" in prod_config.model

    def test_prod_lora_rank_reasonable(self, prod_config):
        """LoRA rank should be between 4 and 64."""
        assert 4 <= prod_config.lora_rank <= 64

    def test_prod_lr_reasonable(self, prod_config):
        """Learning rate should be in the LoRA fine-tuning range."""
        assert 1e-6 <= prod_config.learning_rate <= 1e-3

    def test_prod_has_lr_schedule(self, prod_config):
        """Production config should have an lr_schedule."""
        assert prod_config.lr_schedule is not None
        assert prod_config.lr_schedule["name"] == "cosine_decay"

    def test_prod_lr_schedule_has_arguments(self, prod_config):
        """lr_schedule must have arguments for mlx-lm build_schedule."""
        assert "arguments" in prod_config.lr_schedule
        args = prod_config.lr_schedule["arguments"]
        assert len(args) == 2  # [init_lr, decay_steps]

    def test_prod_lr_schedule_has_warmup(self, prod_config):
        """Production schedule should include warmup."""
        assert "warmup" in prod_config.lr_schedule
        assert prod_config.lr_schedule["warmup"] > 0

    def test_prod_mask_prompt_true(self, prod_config):
        """mask_prompt should be true (only train on assistant responses)."""
        assert prod_config.mask_prompt is True

    def test_prod_grad_checkpoint_true(self, prod_config):
        """grad_checkpoint should be true for Apple Silicon memory safety."""
        assert prod_config.grad_checkpoint is True

    def test_prod_mlx_config_valid(self, prod_config):
        """MLX config conversion should work for production config."""
        mlx = prod_config.to_mlx_config()
        assert mlx["train"] is True
        assert mlx["fine_tune_type"] == "lora"
        assert mlx["lr_schedule"] is not None
