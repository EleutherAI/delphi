from pathlib import Path
from typing import Callable

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi.config import RunConfig
from delphi.latents import LatentCache


def populate_cache(
    run_cfg: RunConfig,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    """
    Populates an on-disk cache in `latents_path` with SAE latent activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_tokenized_data(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        cache_cfg.dataset_name,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens)

    # Save firing counts to the run-specific log directory
    if run_cfg.verbose:
        cache.save_firing_counts()
        cache.generate_statistics_cache()

    cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


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


def load_tokenized_data(
    ctx_len: int,
    tokenizer: AutoTokenizer,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    seed: int = 22,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    """
    from datasets import load_dataset
    from transformer_lens import utils

    print(dataset_repo, dataset_name, dataset_split)
    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    if "rpj" in dataset_repo:
        tokens = utils.tokenize_and_concatenate(
            data, tokenizer, max_length=ctx_len, column_name="raw_content"
        )
    else:
        tokens = utils.tokenize_and_concatenate(
            data, tokenizer, max_length=ctx_len, column_name="text"
        )

    tokens = tokens.shuffle(seed)["tokens"]

    return tokens
