from pathlib import Path

import torch
from sae_lens import HookedSAETransformer

from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.hsae_utils import load_tokenized_data
from delphi.latents import LatentCache
from delphi.sparse_coders import load_hooks_sparse_coders

print("!!! it worked !!!")

"""
*** delphi applied to hsae ***

- uses TransformerLens for the tokenization and the model
"""

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# Path to save the latent activations
latents_path = Path().cwd().parent / "latents_save"

# use get_pretrained_saes_directory to get access to all pretrained sae names
model = HookedSAETransformer.from_pretrained("tiny-stories-1L-21M", device=device)

# Load model and set run configuration
cache_cfg = CacheConfig(
    dataset_repo="EleutherAI/rpj-v2-sample",
    dataset_split="train[:1%]",
    n_splits=5,
    batch_size=128,
    cache_ctx_len=256,
    n_tokens=1_000_000,
)

run_cfg_hsae = RunConfig(
    constructor_cfg=ConstructorConfig(),
    sampler_cfg=SamplerConfig(),
    cache_cfg=cache_cfg,
    model="tiny-stories-1L-21M",
    sparse_model="hsae-16k-22_03_2025",
    hookpoints=["blocks.0.hook_mlp_out"],
)

hookpoint_to_sparse_encode, _ = load_hooks_sparse_coders(model, run_cfg_hsae)

tokens = load_tokenized_data(
    cache_cfg.cache_ctx_len,
    model.tokenizer,
    cache_cfg.dataset_repo,
    cache_cfg.dataset_split,
)

if run_cfg_hsae.filter_bos:
    if model.tokenizer.bos_token_id is None:
        print("Tokenizer does not have a BOS token, skipping BOS filtering")
    else:
        flattened_tokens = tokens.flatten()
        mask = ~torch.isin(
            flattened_tokens, torch.tensor([model.tokenizer.bos_token_id])
        )
        masked_tokens = flattened_tokens[mask]
        truncated_tokens = masked_tokens[
            : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
        ]
        tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

# Create a log path within the run directory
log_path = latents_path.parent / "log"
log_path.mkdir(parents=True, exist_ok=True)

cache = LatentCache(
    model,
    hookpoint_to_sparse_encode,
    batch_size=cache_cfg.batch_size,
    transcode=False,
    log_path=log_path,
)
cache.run(cache_cfg.n_tokens, tokens)

# Save firing counts to the run-specific log directory
if run_cfg_hsae.verbose:
    cache.save_firing_counts()
    cache.generate_statistics_cache()

cache.save_splits(
    # Split the activation and location indices into different files to make
    # loading faster
    n_splits=cache_cfg.n_splits,
    save_dir=latents_path,
)

cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg_hsae.model)

print("!!! it really worked !!!")
