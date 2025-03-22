from pathlib import Path

import torch
from sae_lens import HookedSAETransformer

from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.hsae_utils import populate_cache
from delphi.sparse_coders import load_hooks_sparse_coders

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
latents_path = Path().cwd().parent / "latents"

# use get_pretrained_saes_directory to get access to all pretrained sae names
model = HookedSAETransformer.from_pretrained("tiny-stories-1L-21M", device=device)

# Load model and set run configuration
cache_cfg = CacheConfig(
    dataset_repo="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
    dataset_split="train",
    n_splits=5,
    batch_size=128,
    cache_ctx_len=256,
    n_tokens=10_000_000,
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

populate_cache(
    run_cfg_hsae, model, hookpoint_to_sparse_encode, latents_path, model.tokenizer
)

# def populate_cache(
#     run_cfg: RunConfig,
#     model: PreTrainedModel,
#     hookpoint_to_sparse_encode: dict[str, Callable],
#     latents_path: Path,
#     tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
#     transcode: bool,
# ):
#     """
#     Populates an on-disk cache in `latents_path` with SAE latent activations.
#     """
#     latents_path.mkdir(parents=True, exist_ok=True)

#     # Create a log path within the run directory
#     log_path = latents_path.parent / "log"
#     log_path.mkdir(parents=True, exist_ok=True)

#     cache_cfg = run_cfg.cache_cfg
#     tokens = load_tokenized_data(
#         cache_cfg.cache_ctx_len,
#         tokenizer,
#         cache_cfg.dataset_repo,
#         cache_cfg.dataset_split,
#         cache_cfg.dataset_name,
#         run_cfg.seed,
#     )

#     if run_cfg.filter_bos:
#         if tokenizer.bos_token_id is None:
#             print("Tokenizer does not have a BOS token, skipping BOS filtering")
#         else:
#             flattened_tokens = tokens.flatten()
#             mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
#             masked_tokens = flattened_tokens[mask]
#             truncated_tokens = masked_tokens[
#                 : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
#             ]
#             tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

#     cache = LatentCache(
#         model,
#         hookpoint_to_sparse_encode,
#         batch_size=cache_cfg.batch_size,
#         transcode=transcode,
#         log_path=log_path,
#     )
#     cache.run(cache_cfg.n_tokens, tokens)

#     # Save firing counts to the run-specific log directory
#     if run_cfg.verbose:
#         cache.save_firing_counts()
#         cache.generate_statistics_cache()

#     cache.save_splits(
#         # Split the activation and location indices into different files to make
#         # loading faster
#         n_splits=cache_cfg.n_splits,
#         save_dir=latents_path,
#     )

#     cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


# def load_tokenized_data(
#     ctx_len: int,
#     tokenizer: AutoTokenizer,
#     dataset_repo: str,
#     dataset_split: str,
#     dataset_name: str = "",
#     seed: int = 22,
# ):
#     """
#     Load a huggingface dataset, tokenize it, and shuffle using transformer_lens
#     """
#     from datasets import load_dataset
#     from transformer_lens import utils
#     print(dataset_repo,dataset_name,dataset_split)
#     data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
#     if "rpj" in dataset_repo:
#         tokens = utils.tokenize_and_concatenate(data, tokenizer, max_length=ctx_len,column_name="raw_content")
#     else:
#         tokens = utils.tokenize_and_concatenate(data, tokenizer, max_length=ctx_len,column_name="text")

#     tokens = tokens.shuffle(seed)["tokens"]

#     return tokens
