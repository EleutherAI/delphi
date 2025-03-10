from typing import Any, Type, TypeVar, cast

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def load_tokenized_data(
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    column_name: str = "text",
    seed: int = 22,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    Using this function ensures we are using the same tokens everywhere.
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

    return tokens


T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore
