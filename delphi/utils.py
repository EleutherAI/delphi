from typing import Any, TypeVar, cast

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def load_tokenized_data(
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    column_name: str = "text",
    seed: int = 22,
    convert_to_tensor_chunk_size: int = 2**18,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    Using this function ensures we are using the same tokens everywhere.

    Args:
        ctx_len: The context length of the tokens.
        tokenizer: The tokenizer to use.
        dataset_repo: The repository of the dataset.
        dataset_split: The split of the dataset.
        dataset_name: The name of the dataset.
        column_name: The name of the column to tokenize.
        seed: The seed to use for shuffling the dataset.
        convert_to_tensor_chunk_size: The chunk size to use when converting the dataset
        from Huggingface's Table format to a tensor. Values around 2**17-2**18 seem to
        be the fastest.
    """
    from datasets import load_dataset
    from sparsify.data import chunk_and_tokenize

    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    data = data.shuffle(seed)
    tokens_ds = chunk_and_tokenize(
        data,  # type: ignore
        tokenizer,
        max_seq_len=ctx_len,
        text_key=column_name,
    )

    tokens = tokens_ds["input_ids"]

    try:
        from datasets import Column

        if isinstance(tokens, Column):
            from datasets.table import table_iter

            tokens = torch.cat(
                [
                    torch.from_numpy(
                        np.stack(table_chunk["input_ids"].to_numpy(), axis=0)
                    )
                    for table_chunk in table_iter(
                        tokens.source._data, convert_to_tensor_chunk_size
                    )
                ]
            )
    except ImportError:
        assert len(tokens.shape) == 2

    return tokens


T = TypeVar("T")


def assert_type(typ: type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore


def to_int64_tensor(tensor: np.ndarray) -> Tensor:
    assert tensor.dtype in (
        np.uint16,
        np.int16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    )
    if tensor.dtype in (np.uint64, np.int64):
        return torch.from_numpy(tensor).to(torch.int64)
    og_shape = tensor.shape
    if tensor.dtype in (np.uint16, np.int16):
        signed_np_dtype, signed_torch_dtype = np.int16, torch.int16
        multiplier = 4
    else:
        signed_np_dtype, signed_torch_dtype = np.int32, torch.int32
        multiplier = 2
    t = torch.tensor(tensor.ravel().view(signed_np_dtype))
    result = torch.zeros(t.shape[0] * multiplier, dtype=signed_torch_dtype)
    result[::multiplier] = t
    return result.view(torch.int64).view(og_shape)
