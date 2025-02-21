import json
from typing import Optional, TypedDict

from ..activations.activation_records import (
    ActivationRecord,
    calculate_max_activation,
    normalize_activations,
)
from .few_shot_examples import FewShotExampleSet
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_JSON

UNKNOWN_ACTIVATION_STRING = "unknown"

HarmonyMessage = TypedDict(
    "HarmonyMessage",
    {
        "role": str,
        "content": str,
    },
)


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
    activation_records: list[ActivationRecord],
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


def _format_tokens_for_simulation(tokens: list[str]) -> str:
    """
    Format tokens into a string with each token marked as having an "unknown" activation
    for use in prompts.
    """
    entries = []
    for token in tokens:
        entries.append(f"{token}\t{UNKNOWN_ACTIVATION_STRING}")
    return "\n".join(entries)


def format_sequences_for_simulation(
    all_tokens: list[list[str]],
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


def build_prompt_all_at_once(
    tokens: list[str], explanation: str
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    few_shot_examples = FewShotExampleSet.ORIGINAL.get_examples()
    for i, example in enumerate(few_shot_examples):
        messages.append(
            {
                "role": "user",
                "content": f"\n\nNeuron {i + 1}\nExplanation of neuron {i + 1}"
                f" behavior: {example.explanation}",
            }
        )
        formatted_activation_records = format_activation_records(
            example.activation_records,
            calculate_max_activation(example.activation_records),
            start_indices=example.first_revealed_activation_indices,
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"\nActivations: {formatted_activation_records}\n",
            }
        )

    messages.append(
        {
            "role": "user",
            "content": f"\n\nNeuron {len(few_shot_examples) + 1}\nExplanation of neuron"
            f" {len(few_shot_examples) + 1} behavior: "
            f"{explanation.strip()}",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": f"\nActivations: {format_sequences_for_simulation([tokens])}",
        }
    )

    return messages


def _format_record_for_logprob_free_simulation_json(
    explanation: str,
    activation_record: ActivationRecord,
    include_activations: bool = False,
) -> str:
    if include_activations:
        assert len(activation_record.tokens) == len(
            activation_record.activations
        ), f"{len(activation_record.tokens)=}, {len(activation_record.activations)=}"
    return json.dumps(
        {
            "explanation": explanation,
            "document": "".join(activation_record.tokens),
            "activations": [
                {
                    "token": token,
                    "activation": (
                        activation_record.activations[i]
                        if include_activations
                        else UNKNOWN_ACTIVATION_STRING
                    ),
                }
                for i, token in enumerate(activation_record.tokens)
            ],
        }
    )


def build_simulation_prompt_json(
    tokens: list[str],
    explanation: str,
) -> list[dict[str, str]]:
    """Make a few-shot prompt for predicting the neuron's activations on a sequence."""

    assert explanation != ""
    messages = [{"role": "system", "content": SYSTEM_PROMPT_JSON}]

    few_shot_example = FewShotExampleSet.NEWER.get_single_token_prediction_example()

    # User with unknown activations
    messages.append(
        {
            "role": "user",
            "content": _format_record_for_logprob_free_simulation_json(
                explanation=few_shot_example.explanation,
                activation_record=few_shot_example.activation_records[0],
                include_activations=False,
            ),
        }
    )
    # Assistant with predicted activations
    messages.append(
        {
            "role": "assistant",
            "content": _format_record_for_logprob_free_simulation_json(
                explanation=few_shot_example.explanation,
                activation_record=few_shot_example.activation_records[0],
                include_activations=True,
            ),
        }
    )
    # Our request for the model to predict the activations for the given explanation
    messages.append(
        {
            "role": "user",
            "content": _format_record_for_logprob_free_simulation_json(
                explanation=explanation,
                activation_record=ActivationRecord(tokens=tokens, activations=[]),
                include_activations=False,
            ),
        }
    )
    return messages
