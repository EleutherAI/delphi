import hashlib
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..config import ConstructorConfig
from .latents import (
    ActivatingExample,
    ActivationData,
    LatentRecord,
    NonActivatingExample,
)

model_cache: dict[tuple[str, str], SentenceTransformer] = {}


def get_model(name: str, device: str = "cuda") -> SentenceTransformer:
    global model_cache
    if (name, device) not in model_cache:
        print(f"Loading model {name} on device {device}")
        model_cache[(name, device)] = SentenceTransformer(name, device=device)
    return model_cache[(name, device)]


def prepare_non_activating_examples(
    tokens: Int[Tensor, "examples ctx_len"],
    activations: Float[Tensor, "examples ctx_len"],
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
            activations=acts,
            normalized_activations=None,
            distance=distance,
            str_tokens=tokenizer.batch_decode(toks),
        )
        for toks, acts in zip(tokens, activations)
    ]


def _top_k_pools(
    max_buffer: Float[Tensor, "batch"],
    split_activations: Float[Tensor, "activations ctx_len"],
    buffer_tokens: Int[Tensor, "batch ctx_len"],
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
    sorted_indices = torch.argsort(max_buffer, descending=True)
    activation_windows = split_activations[sorted_indices]
    token_windows = buffer_tokens[sorted_indices]

    return token_windows, activation_windows


def pool_max_activation_windows(
    activations: Float[Tensor, "examples"],
    tokens: Int[Tensor, "windows seq"],
    ctx_indices: Int[Tensor, "examples"],
    index_within_ctx: Int[Tensor, "examples"],
    ctx_len: int,
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

    token_windows, activation_windows = _top_k_pools(max_buffer, new_tensor, tokens)

    return token_windows, activation_windows


def pool_centered_activation_windows(
    activations: Float[Tensor, "examples"],
    tokens: Float[Tensor, "windows seq"],
    n_windows_per_batch: int,
    ctx_indices: Float[Tensor, "examples"],
    index_within_ctx: Float[Tensor, "examples"],
    ctx_len: int,
) -> tuple[Float[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    Similar to pool_max_activation_windows. Doesn't use the ctx_indices that were
    at the start of the batch or the end of the batch, because it always tries
    to have a buffer of ctx_len*5//6 on the left and ctx_len*1//6 on the right.
    To do this, for each window, it joins the contexts from the other windows
    of the same batch, to form a new context, which is then cut to the correct shape,
    centered on the max activation.

    Args:
        activations : The activations.
        tokens : The input tokens.
        ctx_indices : The context indices.
        index_within_ctx : The index within the context.
        ctx_len : The context length.
        max_examples : The maximum number of examples.
    """

    # Get unique context indices and their counts like in pool_max_activation_windows
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(activations, "max", lengths=lengths)

    # Get the top max_examples windows
    sorted_values, sorted_indices = torch.sort(max_buffer, descending=True)

    # this tensor has the correct activations for each context window
    temp_tensor = torch.zeros(len(unique_ctx_indices), ctx_len, dtype=activations.dtype)
    temp_tensor[inverses, index_within_ctx] = activations

    unique_ctx_indices = unique_ctx_indices[sorted_indices]
    temp_tensor = temp_tensor[sorted_indices]

    # if a element in unique_ctx_indices is divisible by n_windows_per_batch it
    # the start of a new batch, so we discard it
    modulo = unique_ctx_indices % n_windows_per_batch
    not_first_position = modulo != 0
    # remove also the elements that are at the end of the batch
    not_last_position = modulo != n_windows_per_batch - 1
    mask = not_first_position & not_last_position
    unique_ctx_indices = unique_ctx_indices[mask]
    temp_tensor = temp_tensor[mask]
    if len(unique_ctx_indices) == 0:
        return torch.zeros(0, ctx_len), torch.zeros(0, ctx_len)

    # Vectorized operations for all windows at once
    n_windows = len(unique_ctx_indices)

    # Create indices for previous, current, and next windows
    prev_indices = unique_ctx_indices - 1
    next_indices = unique_ctx_indices + 1

    # Create a tensor to hold all concatenated tokens
    all_tokens = torch.cat(
        [tokens[prev_indices], tokens[unique_ctx_indices], tokens[next_indices]], dim=1
    )  # Shape: [n_windows, ctx_len*3]

    # Create tensor for all activations
    final_tensor = torch.zeros((n_windows, ctx_len * 3), dtype=activations.dtype)
    final_tensor[:, ctx_len : ctx_len * 2] = (
        temp_tensor  # Set current window activations
    )

    # Set previous window activations where available
    prev_mask = torch.isin(prev_indices, unique_ctx_indices)
    if prev_mask.any():
        prev_locations = torch.where(
            unique_ctx_indices.unsqueeze(1) == prev_indices.unsqueeze(0)
        )[1]
        final_tensor[prev_mask, :ctx_len] = temp_tensor[prev_locations]

    # Set next window activations where available
    next_mask = torch.isin(next_indices, unique_ctx_indices)
    if next_mask.any():
        next_locations = torch.where(
            unique_ctx_indices.unsqueeze(1) == next_indices.unsqueeze(0)
        )[1]
        final_tensor[next_mask, ctx_len * 2 :] = temp_tensor[next_locations]

    # Find max activation indices
    max_activation_indices = torch.argmax(temp_tensor, dim=1) + ctx_len

    # Calculate left for all windows
    left_positions = max_activation_indices - (ctx_len - ctx_len // 4)

    # Create index tensors for gathering
    batch_indices = torch.arange(n_windows).unsqueeze(1)
    pos_indices = torch.arange(ctx_len).unsqueeze(0)
    gather_indices = left_positions.unsqueeze(1) + pos_indices

    # Gather the final windows
    token_windows = all_tokens[batch_indices, gather_indices]
    activation_windows = final_tensor[batch_indices, gather_indices]
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
    min_examples = constructor_cfg.min_examples
    # Get all positions where the latent is active
    flat_indices = (
        activation_data.locations[:, 0] * cache_ctx_len
        + activation_data.locations[:, 1]
    )
    ctx_indices = flat_indices // example_ctx_len
    index_within_ctx = flat_indices % example_ctx_len
    n_windows_per_batch = tokens.shape[1] // example_ctx_len
    reshaped_tokens = tokens.reshape(-1, example_ctx_len)
    n_windows = reshaped_tokens.shape[0]
    unique_batch_pos = ctx_indices.unique()
    mask = torch.ones(n_windows, dtype=torch.bool)
    mask[unique_batch_pos] = False
    # Indices where the latent is not active
    non_active_indices = mask.nonzero(as_tuple=False).squeeze()
    activations = activation_data.activations
    # per context frequency
    record.per_context_frequency = len(unique_batch_pos) / n_windows

    # Add activation examples to the record in place
    if constructor_cfg.center_examples:
        token_windows, act_windows = pool_max_activation_windows(
            activations=activations,
            tokens=reshaped_tokens,
            ctx_indices=ctx_indices,
            index_within_ctx=index_within_ctx,
            ctx_len=example_ctx_len,
        )
    else:
        token_windows, act_windows = pool_centered_activation_windows(
            activations=activations,
            tokens=reshaped_tokens,
            n_windows_per_batch=n_windows_per_batch,
            ctx_indices=ctx_indices,
            index_within_ctx=index_within_ctx,
            ctx_len=example_ctx_len,
        )
    record.examples = [
        ActivatingExample(
            tokens=toks,
            activations=acts,
            normalized_activations=None,
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
    elif source_non_activating == "FAISS":
        non_activating_examples = faiss_non_activation_windows(
            available_indices=non_active_indices,
            record=record,
            tokens=tokens,
            ctx_len=example_ctx_len,
            tokenizer=tokenizer,
            n_not_active=n_not_active,
            embedding_model=constructor_cfg.faiss_embedding_model,
            seed=seed,
            cache_enabled=constructor_cfg.faiss_embedding_cache_enabled,
            cache_dir=constructor_cfg.faiss_embedding_cache_dir,
        )
    else:
        raise ValueError(f"Invalid non-activating source: {source_non_activating}")
    record.not_active = non_activating_examples
    return record


def create_token_key(tokens_tensor, ctx_len):
    """
    Create a file key based on token tensors without detokenization.

    Args:
        tokens_tensor: Tensor of tokens
        ctx_len: Context length

    Returns:
        A string key
    """
    h = hashlib.md5()
    total_tokens = 0

    # Process a sample of elements (first, middle, last)
    num_samples = len(tokens_tensor)
    indices_to_hash = (
        [0, num_samples // 2, -1] if num_samples >= 3 else range(num_samples)
    )

    for idx in indices_to_hash:
        if 0 <= idx < num_samples or (idx == -1 and num_samples > 0):
            # Convert tensor to bytes and hash it
            token_bytes = tokens_tensor[idx].cpu().numpy().tobytes()
            h.update(token_bytes)
            total_tokens += len(tokens_tensor[idx])

    # Add collection shape to make collisions less likely
    shape_str = f"{tokens_tensor.shape}"
    h.update(shape_str.encode())

    return f"{h.hexdigest()[:12]}_items{num_samples}_{ctx_len}"


def faiss_non_activation_windows(
    available_indices: Float[Tensor, "windows"],
    record: LatentRecord,
    tokens: Float[Tensor, "batch seq"],
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    n_not_active: int,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    cache_enabled: bool = True,
    cache_dir: str = ".embedding_cache",
) -> list[NonActivatingExample]:
    """
    Generate hard negative examples using FAISS similarity search based
    on text embeddings.

    This function builds a FAISS index over non-activating examples and
    finds examples that are semantically similar to the activating examples
    based on text embeddings.

    Args:
        available_indices: Indices of windows where the latent is not active
        record: The latent record containing activating examples
        tokens: The input tokens
        ctx_len: The context length for examples
        tokenizer: The tokenizer to decode tokens
        n_not_active: Number of non-activating examples to generate
        embedding_model: Model used for text embeddings
        seed: Random seed
        cache_enabled: Whether to cache embeddings
        cache_dir: Directory to store cached embeddings

    Returns:
        A list of non-activating examples that are semantically similar to
            activating examples
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    # Check if we have enough non-activating examples
    if available_indices.numel() < n_not_active:
        print("Not enough non-activating examples available")
        return []

    # Reshape tokens to get context windows
    reshaped_tokens = tokens.reshape(-1, ctx_len)

    # Get non-activating token windows
    non_activating_tokens = reshaped_tokens[available_indices]

    # Define cache directory structure
    cache_dir = os.environ.get("DELPHI_CACHE_DIR", cache_dir)
    embedding_model_name = embedding_model.split("/")[-1]
    cache_path = Path(cache_dir) / embedding_model_name

    # Get activating example texts

    activating_texts = [
        "".join(tokenizer.batch_decode(example.tokens))
        for example in record.examples[: min(10, len(record.examples))]
    ]

    if not activating_texts:
        print("No activating examples available")
        return []

    # Create unique cache keys for both activating and non-activating texts
    # Use the hash of the concatenated texts to ensure uniqueness
    non_activating_cache_key = create_token_key(non_activating_tokens, ctx_len)
    activating_cache_key = create_token_key(
        torch.stack(
            [
                example.tokens
                for example in record.examples[: min(10, len(record.examples))]
            ]
        ),
        ctx_len,
    )

    # Cache files for activating and non-activating embeddings
    non_activating_cache_file = cache_path / f"{non_activating_cache_key}.faiss"
    activating_cache_file = cache_path / f"{activating_cache_key}.npy"

    # Try to load cached non-activating embeddings
    index = None
    if cache_enabled and non_activating_cache_file.exists():
        try:
            index = faiss.read_index(str(non_activating_cache_file), faiss.IO_FLAG_MMAP)
            print(f"Loaded non-activating index from {non_activating_cache_file}")
        except Exception as e:
            print(f"Error loading cached embeddings: {repr(e)}")

    if index is None:
        print("Decoding non-activating tokens...")
        non_activating_texts = [
            "".join(tokenizer.batch_decode(tokens)) for tokens in non_activating_tokens
        ]

        print("Computing non-activating embeddings...")
        non_activating_embeddings = get_model(embedding_model).encode(
            non_activating_texts, show_progress_bar=False
        )
        dim = non_activating_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)

        index.add(non_activating_embeddings)  # type: ignore
        if cache_enabled:
            os.makedirs(cache_path, exist_ok=True)
            faiss.write_index(index, str(non_activating_cache_file))
            print(f"Cached non-activating embeddings to {non_activating_cache_file}")

    activating_embeddings = None
    if cache_enabled and activating_cache_file.exists():
        try:
            activating_embeddings = np.load(activating_cache_file)
            print(f"Loaded cached activating embeddings from {activating_cache_file}")
        except Exception as e:
            print(f"Error loading cached embeddings: {repr(e)}")
    # Compute embeddings for activating examples if not cached
    if activating_embeddings is None:
        print("Computing activating embeddings...")
        activating_embeddings = get_model(embedding_model).encode(
            activating_texts, show_progress_bar=False
        )
        # Cache the embeddings
        if cache_enabled:
            os.makedirs(cache_path, exist_ok=True)
            np.save(activating_cache_file, activating_embeddings)
            print(f"Cached activating embeddings to {activating_cache_file}")

    # Search for the nearest neighbors to each activating example
    collected_indices = set()
    hard_negative_indices = []

    # For each activating example, find the closest non-activating examples
    for embedding in activating_embeddings:
        # Skip if we already have enough examples
        if len(hard_negative_indices) >= n_not_active:
            break

        # Search for similar non-activating examples
        distances, indices = index.search(embedding.reshape(1, -1), n_not_active)  # type: ignore

        # Add new indices that haven't been collected yet
        for idx in indices[0]:
            if (
                idx not in collected_indices
                and len(hard_negative_indices) < n_not_active
            ):
                hard_negative_indices.append(idx)
                collected_indices.add(idx)

    # Get the token windows for the selected hard negatives
    selected_tokens = non_activating_tokens[hard_negative_indices]
    # Create non-activating examples
    return prepare_non_activating_examples(
        selected_tokens,
        torch.zeros_like(selected_tokens),
        -1.0,  # Using -1.0 as the distance since these are not neighbour-based
        tokenizer,
    )


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
        token_windows, token_activations = pool_max_activation_windows(
            activations=activations,
            tokens=reshaped_tokens,
            ctx_indices=available_ctx_indices,
            index_within_ctx=available_index_within_ctx,
            ctx_len=ctx_len,
        )
        token_windows = token_windows[:n_examples_per_neighbour]
        token_activations = token_activations[:n_examples_per_neighbour]
        # use the first n_examples_per_neighbour examples,
        # which will be the most active examples
        examples_used = len(token_windows)
        all_examples.extend(
            prepare_non_activating_examples(
                token_windows,
                token_activations,  # activations of neighbour
                -neighbour.distance,
                tokenizer,
            )
        )
        number_examples += examples_used
    if len(all_examples) == 0:
        print("No examples found, falling back to random non-activating examples")
        non_active_indices = not_active_mask.nonzero(as_tuple=False).squeeze()

        return random_non_activating_windows(
            available_indices=non_active_indices,
            reshaped_tokens=reshaped_tokens,
            n_not_active=n_not_active,
            tokenizer=tokenizer,
        )
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
        torch.zeros_like(toks),  # there is no way to define these activations
        -1.0,
        tokenizer,
    )
