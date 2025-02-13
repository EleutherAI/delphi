import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from delphi.config import CacheConfig, RunConfig
from delphi.latents import LatentCache
from delphi.sparse_coders import load_sparse_coders

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
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def model():
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    return model


@pytest.fixture(scope="module")
def mock_dataset(tokenizer: AutoTokenizer) -> torch.Tensor:
    tokens = tokenizer(
        random_text, return_tensors="pt", truncation=True, max_length=16, padding=True
    )["input_ids"]
    return tokens


@pytest.fixture(scope="module")
def cache_setup(
    tmp_path_factory, mock_dataset: torch.Tensor, model: AutoModelForCausalLM
):
    """
    This fixture creates a temporary directory, loads the model,
    initializes the cache, runs the cache once, saves the cache splits
    and configuration, and returns all the relevant objects.
    """
    # Create a temporary directory for saving cache files and config
    temp_dir = tmp_path_factory.mktemp("test_cache")

    # Load model and set run configuration
    run_cfg_gemma = RunConfig(
        model="EleutherAI/pythia-70m",
        sparse_model="EleutherAI/sae-pythia-70m-32k",
        hookpoints=["layers.1"],
    )
    hookpoint_to_sae_encode = load_sparse_coders(model, run_cfg_gemma)

    # Define cache config and initialize cache
    cache_cfg = CacheConfig(batch_size=1, ctx_len=16, n_tokens=100)
    cache = LatentCache(model, hookpoint_to_sae_encode, batch_size=cache_cfg.batch_size)

    # Generate mock tokens and run the cache
    tokens = mock_dataset
    cache.run(cache_cfg.n_tokens, tokens)

    # Save splits to temporary directory (the layer key is "gpt_neox.layers.1")

    cache.save_splits(n_splits=5, save_dir=temp_dir, save_tokens=True)

    # Save the cache config

    cache.save_config(temp_dir, cache_cfg, "EleutherAI/pythia-70m")

    dict_to_return = {
        "cache": cache,
        "tokens": tokens,
        "cache_cfg": cache_cfg,
        "temp_dir": temp_dir,
    }

    return dict_to_return
