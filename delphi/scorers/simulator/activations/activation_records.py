"""Utilities for formatting activation records into prompts."""

import math
from dataclasses import dataclass
from typing import Sequence

from simple_parsing import Serializable


@dataclass
class ActivationRecord(Serializable):
    """Collated lists of tokens and their activations for a single neuron."""

    tokens: list[str]
    """Tokens in the text sequence, represented as strings."""
    activations: list[float]
    """Raw activation values for the neuron on each token in the text sequence."""


def relu(x: float) -> float:
    return max(0.0, x)


def calculate_max_activation(activation_records: Sequence[ActivationRecord]) -> float:
    """Return the maximum activation value of the neuron across all the activation
    records."""
    flattened = [
        # Relu is used to assume any values less than 0 are indicating the neuron is in
        # the resting state. This is a simplifying assumption that works with relu/gelu.
        max(relu(x) for x in activation_record.activations)
        for activation_record in activation_records
    ]
    return max(flattened)


def normalize_activations(
    activation_record: list[float], max_activation: float
) -> list[int]:
    """Convert raw neuron activations to integers on the range [0, 10]."""
    if max_activation <= 0:
        return [0 for x in activation_record]
    # Relu is used to assume any values less than 0 are indicating the neuron is in the
    # resting state. This is a simplifying assumption that works with relu/gelu.
    return [
        min(10, math.floor(10 * relu(x) / max_activation)) for x in activation_record
    ]


def non_zero_activation_proportion(
    activation_records: Sequence[ActivationRecord], max_activation: float
) -> float:
    """Return the proportion of activation values that aren't zero."""
    total_activations_count = sum(
        [len(activation_record.activations) for activation_record in activation_records]
    )
    normalized_activations = [
        normalize_activations(activation_record.activations, max_activation)
        for activation_record in activation_records
    ]
    non_zero_activations_count = sum(
        [
            len([x for x in activations if x != 0])
            for activations in normalized_activations
        ]
    )
    return non_zero_activations_count / total_activations_count
