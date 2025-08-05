from pathlib import Path

import torch
from transformers import AutoTokenizer

from delphi.utils import load_tokenized_data


def test_dataset_is_array():
    tokens = load_tokenized_data(
        ctx_len=16,
        tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m"),
        dataset_repo="NeelNanda/pile-10k",
        dataset_split="train",
        dataset_name="",
        column_name="text",
        seed=42,
    )
    assert isinstance(tokens, torch.Tensor)
    assert tokens.ndim == 2
    assert tokens.shape[1] == 16
    assert tokens.dtype in (torch.int64, torch.int32)
    assert tokens.min() >= 0
    assert tokens.max() < 50304


def test_hookpoint_firing_counts_initialization(cache_setup):
    """
    Ensure that hookpoint_firing_counts is initialized as an empty dictionary.
    """
    cache = cache_setup["empty_cache"]
    assert isinstance(cache.hookpoint_firing_counts, dict)
    assert len(cache.hookpoint_firing_counts) == 0  # Should be empty before run()


def test_hookpoint_firing_counts_updates(cache_setup):
    """
    Ensure that hookpoint_firing_counts is properly updated after running the cache.
    """
    cache = cache_setup["empty_cache"]
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
    cache = cache_setup["empty_cache"]
    cache.save_firing_counts()

    firing_counts_path = (
        Path.cwd() / "results" / "test" / "log" / "hookpoint_firing_counts.pt"
    )
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
