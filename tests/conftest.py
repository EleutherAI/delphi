from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.latents import LatentCache
from delphi.sparse_coders import load_hooks_sparse_coders

random_text = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Suspendisse dapibus elementum tellus, ut efficitur lorem fringilla",
    "consequat. Curabitur luctus iaculis cursus. Aliquam erat volutpat.",
    "Nam porttitor vulputate arcu, nec rutrum magna malesuada eget.",
    "Vivamus ultrices lacus quam, quis malesuada augue iaculis et.",
    "Proin a egestas urna, ac sollicitudin orci. Suspendisse sem mi,",
    "vulputate vitae egestas sed, ullamcorper vel arcu.",
    "Phasellus in ornare tellus.Fusce bibendum purus dolor,",
    "quis ornare sem congue eget.",
    "Aenean et lectus nibh. Nunc ac sapien a mauris facilisis",
    "aliquam sed vitae velit. Sed porttitor a diam id rhoncus.",
    "Mauris viverra laoreet ex, vitae pulvinar diam pellentesque nec.",
    "Vivamus quis maximus tellus, vel consectetur lorem.",
]


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def model() -> PreTrainedModel:
    model = AutoModel.from_pretrained("EleutherAI/pythia-70m")
    return model


@pytest.fixture(scope="module")
def mock_dataset(tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    tokens = tokenizer(
        random_text, return_tensors="pt", truncation=True, max_length=16, padding=True
    )["input_ids"]
    tokens = cast(Tensor, tokens)
    return tokens


@pytest.fixture(scope="module")
def cache_setup(tmp_path_factory, mock_dataset: torch.Tensor, model: PreTrainedModel):
    """
    This fixture creates a temporary directory, loads the model,
    initializes the cache, runs the cache once, saves the cache splits
    and configuration, and returns all the relevant objects.
    """
    # Create a temporary directory for saving cache files and config
    temp_dir = tmp_path_factory.mktemp("test_cache")

    # Load model and set run configuration
    cache_cfg = CacheConfig(batch_size=1, cache_ctx_len=16, n_tokens=100)

    run_cfg_gemma = RunConfig(
        constructor_cfg=ConstructorConfig(),
        sampler_cfg=SamplerConfig(),
        cache_cfg=cache_cfg,
        model="EleutherAI/pythia-160m",
        sparse_model="EleutherAI/sae-pythia-70m-32k",
        hookpoints=["layers.1"],
    )
    hookpoint_to_sparse_encode, _ = load_hooks_sparse_coders(model, run_cfg_gemma)
    # Define cache config and initialize cache
    log_path = Path.cwd() / "results" / "test" / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        log_path=log_path,
    )

    # Generate mock tokens and run the cache
    tokens = mock_dataset
    cache.run(cache_cfg.n_tokens, tokens)

    # Save splits to temporary directory (the layer key is "gpt_neox.layers.1")

    cache.save_splits(n_splits=5, save_dir=temp_dir, save_tokens=True)

    # Save the cache config

    cache.save_config(temp_dir, cache_cfg, "EleutherAI/pythia-70m")
    hookpoint_firing_counts = torch.load(
        log_path / "hookpoint_firing_counts.pt", weights_only=True
    )
    return {
        "cache": cache,
        "tokens": tokens,
        "cache_cfg": cache_cfg,
        "temp_dir": temp_dir,
        "firing_counts": hookpoint_firing_counts,
    }


def test_hookpoint_firing_counts_initialization(cache_setup):
    """
    Ensure that hookpoint_firing_counts is initialized as an empty dictionary.
    """
    cache = cache_setup["cache"]
    assert isinstance(cache.hookpoint_firing_counts, dict)
    assert len(cache.hookpoint_firing_counts) == 0  # Should be empty before run()


def test_hookpoint_firing_counts_updates(cache_setup):
    """
    Ensure that hookpoint_firing_counts is properly updated after running the cache.
    """
    cache = cache_setup["cache"]
    tokens = cache_setup["tokens"]
    cache.run(cache_setup["cache_cfg"].n_tokens, tokens)

    assert (
        len(cache.hookpoint_firing_counts) > 0
    ), "hookpoint_firing_counts should not be empty after run()"
    for hookpoint, counts in cache.hookpoint_firing_counts.items():
        assert isinstance(
            counts, torch.Tensor
        ), f"Counts for {hookpoint} should be a torch.Tensor"
        assert counts.ndim == 1, f"Counts for {hookpoint} should be a 1D tensor"
        assert (counts >= 0).all(), f"Counts for {hookpoint} should be non-negative"


def test_hookpoint_firing_counts_persistence(cache_setup):
    """
    Ensure that hookpoint_firing_counts are correctly saved and loaded.
    """
    cache = cache_setup["cache"]
    cache.save_firing_counts()

    firing_counts_path = Path.cwd() / "results" / "log" / "hookpoint_firing_counts.pt"
    assert firing_counts_path.exists(), "Firing counts file should exist after saving"

    loaded_counts = torch.load(firing_counts_path, weights_only=True)
    assert isinstance(
        loaded_counts, dict
    ), "Loaded firing counts should be a dictionary"
    assert (
        loaded_counts.keys() == cache.hookpoint_firing_counts.keys()
    ), "Loaded firing counts keys should match saved keys"

    for hookpoint, counts in loaded_counts.items():
        assert torch.equal(
            counts, cache.hookpoint_firing_counts[hookpoint]
        ), f"Mismatch in firing counts for {hookpoint}"
