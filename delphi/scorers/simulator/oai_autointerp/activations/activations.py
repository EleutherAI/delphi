# Dataclasses and enums for storing neuron-indexed information about activations. Also,
# related helper functions.

from dataclasses import dataclass
from typing import Optional

from simple_parsing import Serializable


@dataclass
class ActivationRecord(Serializable):
    """Collated lists of tokens and their activations for a single neuron."""

    tokens: list[str]
    """Tokens in the text sequence, represented as strings."""
    activations: list[int | float]
    """Raw activation values for the neuron on each token in the text sequence."""


def _check_slices(
    slices_by_split: dict[str, slice],
    expected_num_values: int,
) -> None:
    """Assert that the slices are disjoint and fully cover the intended range."""
    indices = set()
    sum_of_slice_lengths = 0
    n_splits = len(slices_by_split.keys())
    for s in slices_by_split.values():
        subrange = range(expected_num_values)[s]
        sum_of_slice_lengths += len(subrange)
        indices |= set(subrange)
    assert (
        sum_of_slice_lengths == expected_num_values
    ), f"{sum_of_slice_lengths=} != {expected_num_values=}"
    stride = n_splits
    expected_indices = set.union(
        *[
            set(range(start_index, expected_num_values, stride))
            for start_index in range(n_splits)
        ]
    )
    assert indices == expected_indices, f"{indices=} != {expected_indices=}"


def get_slices_for_splits(
    splits: list[str],
    num_activation_records_per_split: int,
) -> dict[str, slice]:
    """
    Get equal-sized interleaved subsets for each of a list of splits, given the number
    of elements to include in each split.
    """

    stride = len(splits)
    num_activation_records_for_even_splits = num_activation_records_per_split * stride
    slices_by_split = {
        split: slice(split_index, num_activation_records_for_even_splits, stride)
        for split_index, split in enumerate(splits)
    }
    _check_slices(
        slices_by_split=slices_by_split,
        expected_num_values=num_activation_records_for_even_splits,
    )
    return slices_by_split


@dataclass
class ActivationRecordSliceParams:
    """How to select splits (train, valid, etc.) of activation records."""

    n_examples_per_split: Optional[int]
    """The number of examples to include in each split."""
