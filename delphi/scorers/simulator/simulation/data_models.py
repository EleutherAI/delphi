"""Data models and utilities for simulator operations.

This module contains:
- ActivationRecord: Core data structure for neuron activation data
- Utility functions for processing ActivationRecord instances
- Configuration constants used throughout the simulator
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from simple_parsing import Serializable

# === CONFIGURATION CONSTANTS ===

# Activation value constraints
MAX_NORMALIZED_ACTIVATION = 10
MIN_NORMALIZED_ACTIVATION = 0

# LLM prompt parameters
PROMPT_LOGPROBS_COUNT = 15
"""Number of top log-probabilities to request from the LLM for logprob-based
simulation."""

JSON_MAX_TOKENS = 1000
"""Maximum number of tokens to generate when requesting JSON responses from the LLM."""

# Token processing
END_OF_TEXT_TOKEN = "<|endoftext|>"
"""The end-of-text token that may appear in sequences."""

END_OF_TEXT_TOKEN_REPLACEMENT = "<|not_endoftext|>"
"""Replacement string for end-of-text tokens to avoid LLM confusion."""

# Prompt formatting
EXPLANATION_PREFIX = "the main thing this neuron does is find"
"""Standard prefix used when presenting neuron explanations in prompts."""

# Validation sets for parsing
VALID_ACTIVATION_TOKENS = set(
    str(i) for i in range(MIN_NORMALIZED_ACTIVATION, MAX_NORMALIZED_ACTIVATION + 1)
)
"""Set of valid activation value tokens for parsing LLM responses."""

# === DATA STRUCTURES ===


@dataclass
class ActivationRecord(Serializable):
    """Collated lists of tokens and their activations for a single neuron."""

    tokens: list[str]
    """Tokens in the text sequence, represented as strings."""

    activations: list[float | int]
    """Raw activation values for the neuron on each token in the text sequence."""

    quantile: int
    """
    Quantile index for this record. Used for grouping records in simulation scoring.
    """


# === UTILITY FUNCTIONS ===

# ReLU is used throughout to assume any values less than 0 indicate the neuron
# is in the resting state - a simplifying assumption that works with relu/gelu.

UNKNOWN_ACTIVATION_STRING = "unknown"


def relu(x: float) -> float:
    """Apply ReLU activation function (max(0, x))."""
    return max(0.0, x)


def calculate_max_activation(activation_records: Sequence[ActivationRecord]) -> float:
    """Return the maximum activation value of the neuron across all the activation
    records."""
    flattened = [
        max(relu(x) for x in activation_record.activations)
        for activation_record in activation_records
    ]
    return max(flattened)


def normalize_activations(
    activation_record: list[float], max_activation: float
) -> list[int]:
    """Convert raw neuron activations to integers on the range
    [0, MAX_NORMALIZED_ACTIVATION]."""
    if max_activation <= 0:
        return [0 for x in activation_record]
    return [
        min(
            MAX_NORMALIZED_ACTIVATION,
            math.floor(MAX_NORMALIZED_ACTIVATION * relu(x) / max_activation),
        )
        for x in activation_record
    ]


def _format_activation_record(
    activation_record: ActivationRecord,
    max_activation: float,
    omit_zeros: bool,
    hide_activations: bool = False,
    start_index: int = 0,
) -> str:
    """Format neuron activations into a string, suitable for use in prompts."""
    tokens = activation_record.tokens
    normalized_activations = normalize_activations(
        activation_record.activations, max_activation
    )
    if omit_zeros:
        assert (
            not hide_activations
        ) and start_index == 0, "Can't hide activations and omit zeros"
        tokens = [
            token
            for token, activation in zip(tokens, normalized_activations)
            if activation > 0
        ]
        normalized_activations = [x for x in normalized_activations if x > 0]

    entries = []
    assert len(tokens) == len(normalized_activations)
    for index, token, activation in zip(
        range(len(tokens)), tokens, normalized_activations
    ):
        activation_string = str(int(activation))
        if hide_activations or index < start_index:
            activation_string = UNKNOWN_ACTIVATION_STRING
        entries.append(f"{token}\t{activation_string}")
    return "\n".join(entries)


def format_activation_records(
    activation_records: Sequence[ActivationRecord],
    max_activation: float,
    *,
    omit_zeros: bool = False,
    start_indices: Optional[list[int]] = None,
    hide_activations: bool = False,
) -> str:
    """Format a list of activation records into a string."""
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join(
            [
                _format_activation_record(
                    activation_record,
                    max_activation,
                    omit_zeros=omit_zeros,
                    hide_activations=hide_activations,
                    start_index=0 if start_indices is None else start_indices[i],
                )
                for i, activation_record in enumerate(activation_records)
            ]
        )
        + "\n<end>\n"
    )


def _format_tokens_for_simulation(tokens: Sequence[str]) -> str:
    """
    Format tokens into a string with each token marked as having an "unknown" activation
    for use in prompts.
    """
    entries = []
    for token in tokens:
        entries.append(f"{token}\t{UNKNOWN_ACTIVATION_STRING}")
    return "\n".join(entries)


def format_sequences_for_simulation(
    all_tokens: Sequence[Sequence[str]],
) -> str:
    """
    Format a list of lists of tokens into a string with each token marked as having
    an "unknown" activation, suitable for use in prompts.
    """
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join(
            [_format_tokens_for_simulation(tokens) for tokens in all_tokens]
        )
        + "\n<end>\n"
    )
