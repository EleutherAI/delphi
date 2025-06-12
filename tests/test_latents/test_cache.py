import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


def test_latent_locations(cache_setup: dict[str, Any]):
    """
    Test that the latent locations generated in memory have the expected
    shape and values.
    """
    cache = cache_setup["cache"]
    locations = cache.cache.latent_locations["layers.1"]
    max_values, _ = locations.max(axis=0)
    # Expected values based on the cache run
    assert max_values[0] == 5, "Expected first dimension max value to be 5"
    assert max_values[1] == 15, "Expected token ids to go up to 15"
    assert max_values[2] > 32700, "Expected latent dimension around 32768"


def test_split_files_created(cache_setup: dict[str, Any]):
    """
    Test that exactly 5 cache split files have been created.
    """
    save_dir = cache_setup["temp_dir"] / "layers.1"
    cache_files = [f for f in os.listdir(save_dir) if f.endswith(".safetensors")]
    assert len(cache_files) == 5, "Expected 5 split files in the cache directory"


def test_split_file_contents(cache_setup: dict[str, Any]):
    """
    Test that one of the split files (loaded via safetensors) holds convincing data:
    - latent locations and activations have the same number of entries,
    - tokens were correctly stored and match the input tokens.
    - latent max values are as expected.
    """
    save_dir = cache_setup["temp_dir"] / "layers.1"
    tokens = cache_setup["tokens"]
    # Choose one file to verify
    cache_files = os.listdir(save_dir)
    file_path = Path(save_dir) / cache_files[0]
    saved_cache = load_file(str(file_path))

    locations = saved_cache["locations"]
    activations = saved_cache["activations"]
    tokens = saved_cache["tokens"]

    assert len(locations) == len(
        activations
    ), "Mismatch between locations & activations entries"

    np.testing.assert_array_equal(
        tokens,
        tokens[:12, :],
        err_msg="Tokens saved do not match the input tokens",
    )
    max_values = locations.max(axis=0)
    assert max_values[0] == 5, "Max batch index mismatch in saved file"
    assert max_values[1] == 15, "Max token value mismatch in saved file"
    assert max_values[2] > 6499, "Latent dimension mismatch in saved file"


def test_config_file(cache_setup: dict[str, Any]):
    """
    Test that the saved configuration file contains the correct parameters.
    """
    config_path = cache_setup["temp_dir"] / "layers.1" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    cache_cfg = cache_setup["cache_cfg"]
    assert config["batch_size"] == cache_cfg.batch_size, "Config batch_size mismatch"
    assert config["cache_ctx_len"] == cache_cfg.cache_ctx_len, "ctx_len mismatch"
    assert config["n_tokens"] == cache_cfg.n_tokens, "Config n_tokens mismatch"
