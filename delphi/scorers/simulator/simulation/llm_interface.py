"""LLM interface utilities for prompt building and response parsing.

This module contains:
- PromptBuilder: Constructs structured prompts for LLM requests
- PromptFormat: Enumeration of supported prompt formats
- Response parsing functions for both logprob and JSON modes
- LLM-specific utility functions
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from enum import Enum
from typing import Any, TypedDict

import tiktoken

# Import data models from our consolidated module
from .data_models import MAX_NORMALIZED_ACTIVATION, MIN_NORMALIZED_ACTIVATION
from .types import SequenceSimulation

# === PROMPT BUILDING ===

HarmonyMessage = TypedDict(
    "HarmonyMessage",
    {
        "role": str,
        "content": str,
    },
)


class PromptFormat(str, Enum):
    """
    Prompt format for the Harmony models that use a structured turn-taking
    role+content format.
    Generates a list of HarmonyMessage dicts that can be sent to the
    /chat/completions endpoint.
    """

    HARMONY_V4 = "harmony_v4"
    """
    Suitable for Harmony models that use a structured turn-taking role+content format.
    Generates a list of HarmonyMessage dicts that can be sent to the /chat/completions
    endpoint.
    """

    @classmethod
    def from_string(cls, s: str) -> PromptFormat:
        for prompt_format in cls:
            if prompt_format.value == s:
                return prompt_format
        raise ValueError(f"{s} is not a valid PromptFormat")


class Role(str, Enum):
    """See https://platform.openai.com/docs/guides/chat"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class PromptBuilder:
    """Class for accumulating components of a prompt and then formatting
    them into an output."""

    def __init__(self) -> None:
        self._messages: list[HarmonyMessage] = []

    def add_message(self, role: Role, message: str) -> None:
        self._messages.append(HarmonyMessage(role=role, content=message))

    def prompt_length_in_tokens(self, prompt_format: PromptFormat) -> int:
        # TODO(sbills): Make the model/encoding configurable. This implementation
        #  assumes GPT-4.
        encoding = tiktoken.get_encoding("cl100k_base")
        assert (
            prompt_format == PromptFormat.HARMONY_V4
        ), f"Only HARMONY_V4 format is supported, got {prompt_format}"

        # Approximately-correct implementation adapted from this documentation:
        # https://platform.openai.com/docs/guides/chat/introduction
        num_tokens = 0
        for message in self._messages:
            num_tokens += (
                4  # every message follows
                # <|im_start|>{role/name}\n{content}<|im_end|>\n
            )
            num_tokens += len(
                encoding.encode(message["content"], allowed_special="all")
            )
        num_tokens += 2  # every reply is primed with <|im_start|>assistant
        return num_tokens

    def build(
        self, prompt_format: PromptFormat, *, allow_extra_system_messages: bool = False
    ) -> list[HarmonyMessage]:
        """
        Validates the messages added so far (reasonable alternation of assistant
        vs. user, etc.) and returns a list of HarmonyMessages suitable for use
        with the /chat/completions endpoint.

        The `allow_extra_system_messages` parameter allows the caller to specify
        that the prompt should be allowed to contain system messages after the very
        first one.
        """
        assert (
            prompt_format == PromptFormat.HARMONY_V4
        ), f"Only HARMONY_V4 format is supported, got {prompt_format}"

        # Create a deep copy of the messages so we can modify it and so that the
        # caller can't modify the internal state of this object.
        messages = [message.copy() for message in self._messages]

        expected_next_role = "system"
        for message in messages:
            role = message["role"]
            assert role == expected_next_role or (
                allow_extra_system_messages and role == "system"
            ), f"Expected message from {expected_next_role} but got message from {role}"
            if role == "system":
                expected_next_role = "user"
            elif role == "user":
                expected_next_role = "assistant"
            elif role == "assistant":
                expected_next_role = "user"

        return messages


# === RESPONSE PARSING ===


def compute_expected_value(
    probabilities_by_distribution_value: OrderedDict[int, float],
) -> float:
    """
    Compute the expected value of a distribution over activation values.
    """
    expected = 0.0
    for distribution_value, probability in probabilities_by_distribution_value.items():
        expected += distribution_value * probability
    return expected


def parse_top_logprobs(top_logprobs: dict[str, float]) -> OrderedDict[int, float]:
    """
    Parse the top logprobs from the API response, extracting activation values.
    """
    probabilities_by_distribution_value = OrderedDict()
    for token, logprob in top_logprobs.items():
        try:
            distribution_value = int(token)
            if (
                MIN_NORMALIZED_ACTIVATION
                <= distribution_value
                <= MAX_NORMALIZED_ACTIVATION
            ):
                # Handle both float and Logprob objects
                logprob_value = (
                    float(logprob) if hasattr(logprob, "__float__") else logprob
                )
                probabilities_by_distribution_value[distribution_value] = math.exp(
                    logprob_value
                )
        except (ValueError, TypeError):
            # Skip non-integer tokens or invalid logprobs
            continue
    return probabilities_by_distribution_value


def compute_predicted_activation_stats_for_token(
    top_logprobs: dict[str, float],
) -> tuple[OrderedDict[int, float], float]:
    probabilities_by_distribution_value = parse_top_logprobs(top_logprobs)
    total_p_of_distribution_values = sum(probabilities_by_distribution_value.values())
    norm_probabilities_by_distribution_value = OrderedDict(
        {
            distribution_value: p / total_p_of_distribution_values
            for distribution_value, p in probabilities_by_distribution_value.items()
        }
    )
    expected_value = compute_expected_value(norm_probabilities_by_distribution_value)
    return (
        norm_probabilities_by_distribution_value,
        expected_value,
    )


def parse_simulation_response(
    response: Any,
    tokenized_prompt: list[int],
    tab_token: int,
    tokens: Sequence[str],
) -> SequenceSimulation:
    """
    Parse an API response to a simulation prompt.

    Args:
        response: response from the API
        tokenized_prompt: tokenized version of the prompt
        tab_token: token ID for tab character
        tokens: list of tokens as strings in the sequence where the neuron
        is being simulated
    """
    logprobs = response.prompt_logprobs

    # Handle both old (dict) and new (list) API formats
    def logprobs_lookup_dict(idx):
        return logprobs[idx]

    def logprobs_lookup_list(idx):
        return logprobs[idx] if idx < len(logprobs) else None

    if isinstance(logprobs, dict):
        logprobs_lookup = logprobs_lookup_dict
    elif isinstance(logprobs, list):
        logprobs_lookup = logprobs_lookup_list
    else:
        # Return zeros if format is unexpected
        return SequenceSimulation(
            tokens=list(tokens),
            expected_activations=[0.0] * len(tokens),
            activation_scale=None,
            distribution_values=[[] for _ in tokens],
            distribution_probabilities=[[] for _ in tokens],
        )

    # Find penultimate assistant token (this works with llama template)
    assistant_token = tokenized_prompt[-3]
    assistant_token_indices = [
        i for i, token in enumerate(tokenized_prompt) if token == assistant_token
    ]
    start_index = assistant_token_indices[-2]

    # Find all the tab tokens after the start index, the next token is the one
    # we care about
    tab_tokens = [
        i + start_index + 1
        for i, token in enumerate(tokenized_prompt[start_index:])
        if token == tab_token
    ]

    expected_values = []
    distribution_values = []
    distribution_probabilities = []

    for tab_indice in tab_tokens:
        token_logprobs = logprobs_lookup(tab_indice)
        if token_logprobs is None:
            continue

        # Convert from new API format to expected format
        # From: {token_id: Logprob(logprob=X, decoded_token="Y")}
        # To: {"Y": X}
        converted_logprobs = {}
        for token_id, logprob_obj in token_logprobs.items():
            if hasattr(logprob_obj, "decoded_token") and hasattr(
                logprob_obj, "logprob"
            ):
                converted_logprobs[logprob_obj.decoded_token] = logprob_obj.logprob

        (
            p_by_distribution_value,
            expected_value,
        ) = compute_predicted_activation_stats_for_token(
            converted_logprobs,
        )
        distribution_values.append(list(p_by_distribution_value.keys()))
        distribution_probabilities.append(list(p_by_distribution_value.values()))
        expected_values.append(float(expected_value))

    # Trim to match token count if needed
    if len(expected_values) > len(tokens):
        expected_values = expected_values[: len(tokens)]
        distribution_values = distribution_values[: len(tokens)]
        distribution_probabilities = distribution_probabilities[: len(tokens)]

    return SequenceSimulation(
        tokens=list(tokens),
        expected_activations=expected_values,
        activation_scale=None,  # Not used in simulation version
        distribution_values=distribution_values,
        distribution_probabilities=distribution_probabilities,
    )
