from typing import Optional

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..config import ConstructorConfig
from .latents import (
    ActivatingExample,
    ActivationData,
    LatentRecord,
    NonActivatingExample,
)


def prepare_non_activating_examples(
    tokens: Int[Tensor, "examples ctx_len"],
    distance: float,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> list[NonActivatingExample]:
    """
    Prepare a list of non-activating examples from input tokens and distance.

    Args:
        tokens: Tokenized input sequences.
        distance: The distance from the neighbouring latent.
    """
    return [
        NonActivatingExample(
            tokens=toks,
            activations=torch.zeros_like(toks, dtype=torch.float),
            normalized_activations=None,
            distance=distance,
            str_tokens=tokenizer.batch_decode(toks),
        )
        for toks in tokens
    ]


def _top_k_pools(
    max_buffer: Float[Tensor, "batch"],
    split_activations: Float[Tensor, "activations ctx_len"],
    buffer_tokens: Int[Tensor, "batch ctx_len"],
    max_examples: int,
) -> tuple[Int[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    Get the top k activation pools.

    Args:
        max_buffer: The maxima of each context window's activations.
        split_activations: The split activations.
        buffer_tokens: The buffer tokens.
        max_examples: The maximum number of examples.

    Returns:
        The token windows and activation windows.
    """
    k = min(max_examples, len(max_buffer))
    top_values, top_indices = torch.topk(max_buffer, k, sorted=True)

    activation_windows = torch.stack([split_activations[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows


def pool_max_activation_windows(
    activations: Float[Tensor, "examples"],
    tokens: Int[Tensor, "windows seq"],
    ctx_indices: Int[Tensor, "examples"],
    index_within_ctx: Int[Tensor, "examples"],
    ctx_len: int,
    max_examples: int,
) -> tuple[Int[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    Pool max activation windows from the buffer output and update the latent record.

    Args:
        activations : The activations.
        tokens : The input tokens.
        ctx_indices : The context indices.
        index_within_ctx : The index within the context.
        ctx_len : The context length.
        max_examples : The maximum number of examples.
    Returns:
        The token windows and activation windows.
    """
    # unique_ctx_indices: array of distinct context window indices in order of first
    # appearance. sequential integers from 0 to batch_size * cache_token_length//ctx_len
    # inverses: maps each activation back to its index in unique_ctx_indices
    # (can be used to dereference the context window idx of each activation)
    # lengths: the number of activations per unique context window index
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(activations, "max", lengths=lengths)

    # Deduplicate the context windows
    new_tensor = torch.zeros(len(unique_ctx_indices), ctx_len, dtype=activations.dtype)
    new_tensor[inverses, index_within_ctx] = activations

    tokens = tokens[unique_ctx_indices]

    token_windows, activation_windows = _top_k_pools(
        max_buffer, new_tensor, tokens, max_examples
    )

    return token_windows, activation_windows


def constructor(
    record: LatentRecord,
    activation_data: ActivationData,
    constructor_cfg: ConstructorConfig,
    tokens: Int[Tensor, "batch seq"],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    all_data: Optional[dict[int, ActivationData]] = None,
    seed: int = 42,
) -> LatentRecord | None:
    cache_ctx_len = tokens.shape[1]
    example_ctx_len = constructor_cfg.example_ctx_len
    source_non_activating = constructor_cfg.non_activating_source
    n_not_active = constructor_cfg.n_non_activating
    max_examples = constructor_cfg.max_examples
    min_examples = constructor_cfg.min_examples

    # Get all positions where the latent is active
    flat_indices = (
        activation_data.locations[:, 0] * cache_ctx_len
        + activation_data.locations[:, 1]
    )
    ctx_indices = flat_indices // example_ctx_len
    index_within_ctx = flat_indices % example_ctx_len
    reshaped_tokens = tokens.reshape(-1, example_ctx_len)
    n_windows = reshaped_tokens.shape[0]

    unique_batch_pos = ctx_indices.unique()

    mask = torch.ones(n_windows, dtype=torch.bool)
    mask[unique_batch_pos] = False
    # Indices where the latent is not active
    non_active_indices = mask.nonzero(as_tuple=False).squeeze()
    activations = activation_data.activations

    # Add activation examples to the record in place
    token_windows, act_windows = pool_max_activation_windows(
        activations=activations,
        tokens=reshaped_tokens,
        ctx_indices=ctx_indices,
        index_within_ctx=index_within_ctx,
        ctx_len=example_ctx_len,
        max_examples=max_examples,
    )
    # TODO: We might want to do this in the sampler
    # we are tokenizing examples that are not going to be used
    record.examples = [
        ActivatingExample(
            tokens=toks,
            activations=acts,
            normalized_activations=None,
            str_tokens=tokenizer.batch_decode(toks),
        )
        for toks, acts in zip(token_windows, act_windows)
    ]

    if len(record.examples) < min_examples:
        # Not enough examples to explain the latent
        return None

    if source_non_activating == "random":
        # Add random non-activating examples to the record in place
        non_activating_examples = random_non_activating_windows(
            available_indices=non_active_indices,
            reshaped_tokens=reshaped_tokens,
            n_not_active=n_not_active,
            seed=seed,
            tokenizer=tokenizer,
        )
    elif source_non_activating == "neighbours":
        assert all_data is not None, "All data is required for neighbour constructor"
        non_activating_examples = neighbour_non_activation_windows(
            record,
            not_active_mask=mask,
            tokens=tokens,
            all_data=all_data,
            ctx_len=example_ctx_len,
            n_not_active=n_not_active,
            seed=seed,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Invalid non-activating source: {source_non_activating}")

    record.not_active = non_activating_examples
    return record


def neighbour_non_activation_windows(
    record: LatentRecord,
    not_active_mask: Bool[Tensor, "windows"],
    tokens: Int[Tensor, "batch seq"],
    all_data: dict[int, ActivationData],
    ctx_len: int,
    n_not_active: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    seed: int = 42,
):
    """
    Generate random activation windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        not_active_mask (TensorType["n_windows"]): The mask of the non-active windows.
        tokens (TensorType["batch", "seq"]): The input tokens.
        all_data (AllData): The all data containing activations and locations.
        ctx_len (int): The context length.
        n_not_active (int): The number of non-activating examples per latent.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer.
        seed (int): The random seed.
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    assert (
        record.neighbours is not None
    ), "Neighbours are not set, please precompute them"

    cache_token_length = tokens.shape[1]
    reshaped_tokens = tokens.reshape(-1, ctx_len)
    n_windows = reshaped_tokens.shape[0]
    # TODO: For now we use at most 10 examples per neighbour, we may want to allow a
    # variable number of examples per neighbour
    n_examples_per_neighbour = 10

    number_examples = 0
    all_examples = []
    for neighbour in record.neighbours:
        if number_examples >= n_not_active:
            break
        # get the locations of the neighbour
        if neighbour.latent_index not in all_data:
            continue
        locations = all_data[neighbour.latent_index].locations
        activations = all_data[neighbour.latent_index].activations
        # get the active window indices
        flat_indices = locations[:, 0] * cache_token_length + locations[:, 1]
        ctx_indices = flat_indices // ctx_len
        index_within_ctx = flat_indices % ctx_len
        # Set the mask to True for the unique locations
        unique_batch_pos_active = ctx_indices.unique()

        mask = torch.zeros(n_windows, dtype=torch.bool)
        mask[unique_batch_pos_active] = True

        # Get the indices where mask and not_active_mask are True
        mask = mask & not_active_mask

        available_indices = mask.nonzero().flatten()

        mask_ctx = torch.isin(ctx_indices, available_indices)
        available_ctx_indices = ctx_indices[mask_ctx]
        available_index_within_ctx = index_within_ctx[mask_ctx]
        activations = activations[mask_ctx]
        # If there are no available indices, skip this neighbour
        if activations.numel() == 0:
            continue
        token_windows, _ = pool_max_activation_windows(
            activations=activations,
            tokens=reshaped_tokens,
            ctx_indices=available_ctx_indices,
            index_within_ctx=available_index_within_ctx,
            max_examples=n_examples_per_neighbour,
            ctx_len=ctx_len,
        )
        # use the first n_examples_per_neighbour examples,
        # which will be the most active examples
        examples_used = len(token_windows)
        all_examples.extend(
            prepare_non_activating_examples(
                token_windows, -neighbour.distance, tokenizer
            )
        )
        number_examples += examples_used
    if len(all_examples) == 0:
        print("No examples found")
    return all_examples


def random_non_activating_windows(
    available_indices: Int[Tensor, "windows"],
    reshaped_tokens: Int[Tensor, "windows ctx_len"],
    n_not_active: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    seed: int = 42,
) -> list[NonActivatingExample]:
    """
    Generate random non-activating sequence windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        available_indices (TensorType["n_windows"]): The indices of the windows where
        the latent is not active.
        reshaped_tokens (TensorType["n_windows", "ctx_len"]): The tokens reshaped
        to the context length.
        n_not_active (int): The number of non activating examples to generate.
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    # If this happens it means that the latent is active in every window,
    # so it is a bad latent
    if available_indices.numel() < n_not_active:
        print("No available randomly sampled non-activating sequences")
        return []
    else:
        random_indices = torch.randint(
            0, available_indices.shape[0], size=(n_not_active,)
        )
        selected_indices = available_indices[random_indices]

    toks = reshaped_tokens[selected_indices]

    return prepare_non_activating_examples(
        toks,
        -1.0,
        tokenizer,
    )
