"""Uses API calls to simulate neuron activations based on an explanation."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel

from delphi.clients.client import Client
from ..activations.activation_records import (
    calculate_max_activation,
    format_activation_records,
    format_sequences_for_simulation,
    normalize_activations,
)
from ..activations.activations import ActivationRecord
from .explanations import (
    ActivationScale,
    SequenceSimulation,
)
from .few_shot_examples import FewShotExampleSet
from .prompt_builder import (
    HarmonyMessage,
    PromptBuilder,
    PromptFormat,
    Role,
)

logger = logging.getLogger(__name__)

# Our prompts use normalized activation values, which map any range of positive
# activations to the integers from 0 to 10.
MAX_NORMALIZED_ACTIVATION = 10
VALID_ACTIVATION_TOKENS_ORDERED = list(
    str(i) for i in range(MAX_NORMALIZED_ACTIVATION + 1)
)
VALID_ACTIVATION_TOKENS = set(VALID_ACTIVATION_TOKENS_ORDERED)

# Edge Case #3: The chat-based simulator is confused by end token. Replace it with
# a "not end token"
END_OF_TEXT_TOKEN = "<|endoftext|>"
END_OF_TEXT_TOKEN_REPLACEMENT = "<|not_endoftext|>"
EXPLANATION_PREFIX = "the main thing this neuron does is find"


class Activation(BaseModel):
    token: str
    activation: float


class ResponseModel(BaseModel):
    to_find: str
    document: str
    activations: list[Activation]


class SimulationType(str, Enum):
    """How to simulate neuron activations.
    Values correspond to subclasses of NeuronSimulator."""

    ALL_AT_ONCE = "all_at_once"
    """
    Use a single prompt with unknown tokens; calculate EVs using logprobs.

    Implemented by ExplanationNeuronSimulator.
    """

    ONE_AT_A_TIME = "one_at_a_time"
    """
    Use a separate prompt for each token being simulated; calculate EVs using logprobs.

    Implemented by ExplanationTokenByTokenSimulator.
    """

    @classmethod
    def from_string(cls, s: str) -> SimulationType:
        for simulation_type in SimulationType:
            if simulation_type.value == s:
                return simulation_type
        raise ValueError(f"Invalid simulation type: {s}")


def compute_expected_value(
    norm_probabilities_by_distribution_value: OrderedDict[int, float],
) -> float:
    """
    Given a map from distribution values (integers on the range [0, 10]) to normalized
    probabilities, return an expected value for the distribution.
    """
    return np.dot(
        np.array(list(norm_probabilities_by_distribution_value.keys())),
        np.array(list(norm_probabilities_by_distribution_value.values())),
    )


def parse_top_logprobs(top_logprobs: dict[str, float]) -> OrderedDict[int, float]:
    """
    Given a map from tokens to logprobs, return a map from distribution values
    (integers on the range [0, 10]) to unnormalized probabilities
    (in the sense that they may not sum to 1).
    """
    probabilities_by_distribution_value = OrderedDict()
    for token, contents in top_logprobs.items():
        logprob = contents.logprob  # type: ignore
        decoded_token = contents.decoded_token  # type: ignore
        if decoded_token in VALID_ACTIVATION_TOKENS:
            token_as_int = int(decoded_token)
            probabilities_by_distribution_value[token_as_int] = np.exp(logprob)
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
        prompt_format: how the prompt was formatted
        tokens: list of tokens as strings in the sequence where the neuron
        is being simulated
    """
    logprobs = response.prompt_logprobs
    # print(logprobs)
    # print(logprobs[1])
    # (gpaulo) this should be done in a smarter way, it really only works with the llama template
    assistant_token = tokenized_prompt[-3]
    # find penultimate assistant token
    assistant_token_indices = [
        i for i, token in enumerate(tokenized_prompt) if token == assistant_token
    ]
    start_index = assistant_token_indices[-2]
    # find all the tab tokens after the start index, the next token is the one we care to do logprobs over
    tab_tokens = [
        i + start_index + 1
        for i, token in enumerate(tokenized_prompt[start_index:])
        if token == tab_token
    ]

    expected_values = []
    distribution_values = []
    distribution_probabilities = []

    for tab_indice in tab_tokens:
        (
            p_by_distribution_value,
            expected_value,
        ) = compute_predicted_activation_stats_for_token(
            logprobs[tab_indice],
        )
        distribution_values.append(list(p_by_distribution_value.keys()))
        distribution_probabilities.append(list(p_by_distribution_value.values()))
        expected_values.append(float(expected_value))

    # (gpaulo) I don't know if this is the best way to handle this
    if len(expected_values) > len(tokens):
        expected_values = expected_values[: len(tokens)]
        distribution_values = distribution_values[: len(tokens)]
        distribution_probabilities = distribution_probabilities[: len(tokens)]
    if len(expected_values) < len(tokens):
        expected_values = expected_values + [0] * (len(tokens) - len(expected_values))
        distribution_values = distribution_values + [[0]] * (
            len(tokens) - len(distribution_values)
        )
        distribution_probabilities = distribution_probabilities + [[0]] * (
            len(tokens) - len(distribution_probabilities)
        )

    return SequenceSimulation(
        tokens=list(tokens),
        expected_activations=expected_values,
        activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
        distribution_values=distribution_values,
        distribution_probabilities=distribution_probabilities,
    )


class NeuronSimulator(ABC):
    """Abstract base class for simulating neuron behavior."""

    @abstractmethod
    async def simulate(self, tokens: Sequence[str]) -> SequenceSimulation:
        """Simulate the behavior of a neuron based on an explanation."""
        ...


class ExplanationNeuronSimulator(NeuronSimulator):
    """
    Simulate neuron behavior based on an explanation.

    This class uses a few-shot prompt with examples of other explanations
    and activations.
    This prompt allows us to score all of the tokens at once using a nifty trick
    involving logprobs.
    """

    def __init__(
        self,
        client: Client,
        explanation: str,
        few_shot_example_set: FewShotExampleSet = FewShotExampleSet.ORIGINAL,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
    ):
        self.client = client
        self.explanation = explanation
        self.few_shot_example_set = few_shot_example_set
        self.prompt_format = prompt_format

    async def simulate(
        self,
        tokens: Sequence[str],
    ) -> SequenceSimulation:
        prompt = self.make_simulation_prompt(tokens)
        sampling_params: dict[str, Any] = {
            "max_tokens": 1,
            "prompt_logprobs": 15,
        }
        if self.prompt_format == PromptFormat.HARMONY_V4:
            assert isinstance(prompt, list)
            assert isinstance(prompt[0], dict)  # Really a HarmonyMessage

        else:
            assert isinstance(prompt, str)

        response = await self.client.generate(prompt, **sampling_params)  # type: ignore
        tokenized_prompt = self.client.tokenizer.apply_chat_template(  # type: ignore
            prompt, add_generation_prompt=True
        )
        tab_token = self.client.tokenizer.encode("\t")[1]  # type: ignore
        logger.debug("response in score_explanation_by_activations is %s", response)
        try:
            result = parse_simulation_response(
                response, tokenized_prompt, tab_token, tokens
            )
            logger.debug("result in score_explanation_by_activations is %s", result)
            return result
        except Exception as e:
            logger.error(f"Simulation response parsing failed: {e}")
            return SequenceSimulation(
                tokens=list(tokens),
                expected_activations=[0] * len(tokens),
                activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
                distribution_values=[],
                distribution_probabilities=[],
            )

    # TODO(sbills): The current token<tab>activation format can result in
    # improper tokenization. In particular, if the token is itself a tab,
    # we may get a single "\t\t" token rather than two "\t" tokens.
    # Consider using a separator that does not appear in any multi-character tokens.
    def make_simulation_prompt(
        self, tokens: Sequence[str]
    ) -> Union[str, list[HarmonyMessage]]:
        """Create a few-shot prompt for predicting neuron activations for
        the given tokens."""

        # TODO(sbills): The prompts in this file are subtly different from
        # the ones in explainer.py.
        # Consider reconciling them.
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            "system",  # type: ignore
            """We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.
Look at summary of what the neuron does, and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to 10, "unknown" indicates an unknown activation. Most activations will be 0.
""",
        )

        few_shot_examples = self.few_shot_example_set.get_examples()
        for i, example in enumerate(few_shot_examples):
            prompt_builder.add_message(
                "user",  # type: ignore
                f"\n\nNeuron {i + 1}\nExplanation of neuron {i + 1} behavior: {EXPLANATION_PREFIX}"
                f"{example.explanation}",
            )
            formatted_activation_records = format_activation_records(
                example.activation_records,
                calculate_max_activation(example.activation_records),
                start_indices=example.first_revealed_activation_indices,
            )
            prompt_builder.add_message(
                "assistant", f"\nActivations: {formatted_activation_records}\n"  # type: ignore
            )

        prompt_builder.add_message(
            "user",  # type: ignore
            f"\n\nNeuron {len(few_shot_examples) + 1}\nExplanation of neuron "
            f"{len(few_shot_examples) + 1} behavior: {EXPLANATION_PREFIX} "
            f"{self.explanation.strip()}",
        )
        prompt_builder.add_message(
            "assistant",  # type: ignore
            f"\nActivations: {format_sequences_for_simulation([tokens])}",
        )
        return prompt_builder.build(self.prompt_format)


def _format_record_for_logprob_free_simulation(
    activation_record: ActivationRecord,
    include_activations: bool = False,
    max_activation: Optional[float] = None,
) -> str:
    response = ""
    normalized_activations = None
    if include_activations:
        assert max_activation is not None
        assert len(activation_record.tokens) == len(
            activation_record.activations
        ), f"{len(activation_record.tokens)=}, {len(activation_record.activations)=}"
        normalized_activations = normalize_activations(
            activation_record.activations, max_activation=max_activation
        )

    for i, token in enumerate(activation_record.tokens):
        # Edge Case #3: End tokens confuse the chat-based simulator. Replace end token with "not end token".
        if token.strip() == END_OF_TEXT_TOKEN:
            token = END_OF_TEXT_TOKEN_REPLACEMENT
        # We use a weird unicode character here to make it easier to parse the response (can split on "༗\n").
        if include_activations:
            assert normalized_activations is not None
            response += f"{token}\t{normalized_activations[i]}༗\n"
        else:
            response += f"{token}\t༗\n"
    return response


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
            "to_find": explanation,
            "document": "".join(activation_record.tokens),
            "activations": [
                {
                    "token": token,
                    "activation": (
                        activation_record.activations[i]
                        if include_activations
                        else None
                    ),
                }
                for i, token in enumerate(activation_record.tokens)
            ],
        }
    )


def _parse_no_logprobs_completion_json(
    completion,
    tokens: Sequence[str],
) -> Sequence[float]:
    """
    Parse a completion into a list of simulated activations. If the model did not faithfully
    reproduce the token sequence, return a list of 0s. If the model's activation for a token
    is not a number between 0 and 10 (inclusive), substitute 0.

    Args:
        completion: completion from the API
        tokens: list of tokens as strings in the sequence where the neuron is being simulated
    """

    logger.debug("for tokens:\n%s", tokens)
    logger.debug("received completion:\n%s", completion)

    zero_prediction = [0] * len(tokens)

    try:
        # TODO: we need to standardize the output format of the clients
        completion = json.loads(completion.text)
        if "activations" not in completion:
            logger.error(
                "The key 'activations' is not in the completion:\n%s\nExpected Tokens:\n%s",
                json.dumps(completion),
                tokens,
            )
            return zero_prediction
        activations = completion["activations"]
        if len(activations) != len(tokens):
            logger.error(
                "Tokens and activations length did not match:\n%s\nExpected Tokens:\n%s",
                json.dumps(completion),
                tokens,
            )
            return zero_prediction
        predicted_activations = []
        # check that there is a token and activation value
        # no need to double check the token matches exactly
        for i, activation in enumerate(activations):
            if "token" not in activation:
                logger.error(
                    "The key 'token' is not in activation:\n%s\nCompletion:%s\nExpected Tokens:\n%s",
                    activation,
                    json.dumps(completion),
                    tokens,
                )
                predicted_activations.append(0)
                continue
            if "activation" not in activation:
                logger.error(
                    "The key 'activation' is not in activation:\n%s\nCompletion:%s\nExpected Tokens:\n%s",
                    activation,
                    json.dumps(completion),
                    tokens,
                )
                predicted_activations.append(0)
                continue
            # Ensure activation value is between 0-10 inclusive
            try:
                predicted_activation_float = float(activation["activation"])
                if (
                    predicted_activation_float < 0
                    or predicted_activation_float > MAX_NORMALIZED_ACTIVATION
                ):
                    logger.error(
                        "activation value out of range: %s\nCompletion:%s\nExpected Tokens:\n%s",
                        predicted_activation_float,
                        json.dumps(completion),
                        tokens,
                    )
                    predicted_activations.append(0)
                else:
                    predicted_activations.append(predicted_activation_float)
            except ValueError:
                logger.error(
                    "activation value invalid: %s\nCompletion:%s\nExpected Tokens:\n%s",
                    activation["activation"],
                    json.dumps(completion),
                    tokens,
                )
                predicted_activations.append(0)
            except TypeError:
                logger.error(
                    "activation value incorrect type: %s\nCompletion:%s\nExpected Tokens:\n%s",
                    activation["activation"],
                    json.dumps(completion),
                    tokens,
                )
                predicted_activations.append(0)
        logger.debug("predicted activations: %s", predicted_activations)
        return predicted_activations

    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse completion JSON:\n%s\nExpected Tokens:\n%s",
            completion,
            tokens,
        )
        return zero_prediction


class LogprobFreeExplanationTokenSimulator(NeuronSimulator):
    """
    Simulate neuron behavior based on an explanation.

    Unlike ExplanationNeuronSimulator and ExplanationTokenByTokenSimulator, this class does not rely on
    logprobs to calculate expected activations. Instead, it uses a few-shot prompt that displays all of the
    tokens at once, and request that the model repeat the tokens with the activations appended. Sampling
    is with temperature = 0. Thus, the activations are deterministic. Also, each activation for a token
    is a function of all the activations that came previously and all of the tokens in the sequence, not
    just the current and previous tokens. In the case where the model does not faithfully reproduce the
    token sequence, the simulator will return a response where every predicted activation is 0. Example prompt as follows:

    Explanation: Explanation 1

    Sequence 1 Tokens Without Activations:

    A\t_
    B\t_
    C\t_

    Sequence 1 Tokens With Activations:

    A\t4_
    B\t10_
    C\t0_

    Sequence 2 Tokens Without Activations:

    D\t_
    E\t_
    F\t_

    Sequence 2 Tokens With Activations:

    D\t3_
    E\t6_
    F\t9_

    Explanation: Explanation 2

    Sequence 1 Tokens Without Activations:

    G\t_
    H\t_
    I\t_

    Sequence 1 Tokens With Activations:
    <start sampling here>

    G\t2_
    H\t0_
    I\t3_

    """

    def __init__(
        self,
        client,
        explanation: str,
        max_concurrent: Optional[int] = 10,
        json_mode: Optional[bool] = True,
        few_shot_example_set: FewShotExampleSet = FewShotExampleSet.NEWER,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
        cache: bool = False,
    ):
        assert (
            few_shot_example_set != FewShotExampleSet.ORIGINAL
        ), "This simulator doesn't support the ORIGINAL few-shot example set."
        # self.api_client = ApiClient(
        #     model_name=model_name, max_concurrent=max_concurrent, cache=cache
        # )

        self.client = client
        self.json_mode = json_mode
        self.explanation = explanation
        self.few_shot_example_set = few_shot_example_set
        self.prompt_format = prompt_format

    async def simulate(self, tokens: Sequence[str]) -> SequenceSimulation:
        if self.json_mode:
            prompt = self._make_simulation_prompt_json(
                tokens,
                self.explanation,
            )
            try:
                response = await self.client.generate(prompt, max_tokens=1000)

                predicted_activations = _parse_no_logprobs_completion_json(
                    response, tokens
                )
            except Exception as e:
                predicted_activations = []
        else:
            prompt = self._make_simulation_prompt_json(
                tokens,
                self.explanation,
            )
            try:
                response = await self.client.generate(
                    prompt, max_tokens=1000, temperature=0.0
                )
                predicted_activations = []
                # assert len(response["choices"]) == 1
                choice = response["choices"][0]
                completion = choice["message"]["content"]
                predicted_activations = _parse_no_logprobs_completion_json(
                    completion, tokens
                )
            except Exception as e:
                predicted_activations = []

        result = SequenceSimulation(
            activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
            expected_activations=predicted_activations,  # type: ignore
            # Since the predicted activation is just a sampled token, we don't have a distribution.
            distribution_values=[],
            distribution_probabilities=[],
            tokens=list(tokens),  # SequenceSimulation expects list type
        )
        logger.debug("result in score_explanation_by_activations is %s", result)
        return result

    def _make_simulation_prompt_json(
        self,
        tokens: Sequence[str],
        explanation: str,
    ) -> Union[str, list[HarmonyMessage]]:
        """Make a few-shot prompt for predicting the neuron's activations on a sequence."""
        """NOTE: The JSON version does not give GPT multiple sequence examples per neuron."""
        assert explanation != ""
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            "system",  # type: ignore
            """We're studying neurons in a neural network. Each neuron looks for certain things in a short document. Your task is to read the explanation of what the neuron does, and predict the neuron's activations for each token in the document.

For each document, you will see the full text of the document, then the tokens in the document with the activation left blank. You will print, in valid json, the exact same tokens verbatim, but with the activation values filled in according to the explanation. Pay special attention to the explanation's description of the context and order of tokens or words.

Fill out the activation values with integer values from 0 to 10. Don't use negative numbers. Please think carefully.";
""",
        )

        few_shot_examples = self.few_shot_example_set.get_examples()
        for example in few_shot_examples:
            """
            {
                "to_find": "hello",
                "document": "The",
                "activations": [
                    {
                        "token": "The",
                        "activation": null
                    }
                ]
            }
            """
            prompt_builder.add_message(
                "user",  # type: ignore
                _format_record_for_logprob_free_simulation_json(
                    explanation=example.explanation,
                    activation_record=example.activation_records[0],
                    include_activations=False,
                ),
            )
            """
            {
                "to_find": "hello",
                "document": "The",
                "activations": [
                    {
                        "token": "The",
                        "activation": 10
                    }
                ]
            }
            """
            prompt_builder.add_message(
                "assistant",  # type: ignore
                _format_record_for_logprob_free_simulation_json(
                    explanation=example.explanation,
                    activation_record=example.activation_records[0],
                    include_activations=True,
                ),
            )
        """
        {
            "to_find": "hello",
            "document": "The",
            "activations": [
                {
                    "token": "The",
                    "activation": null
                }
            ]
        }
        """
        prompt_builder.add_message(
            "user",  # type: ignore
            _format_record_for_logprob_free_simulation_json(
                explanation=explanation,
                activation_record=ActivationRecord(tokens=tokens, activations=[]),  # type: ignore
                include_activations=False,
            ),
        )
        return prompt_builder.build(
            self.prompt_format, allow_extra_system_messages=True
        )

    def _make_simulation_prompt(
        self,
        tokens: Sequence[str],
        explanation: str,
    ) -> Union[str, list[HarmonyMessage]]:
        """Make a few-shot prompt for predicting the neuron's activations on a sequence."""
        assert explanation != ""
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            "system",  # type: ignore
            """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at an explanation of what the neuron does, and try to predict its activations on a particular token.

The activation format is token<tab>activation, and activations range from 0 to 10. Most activations will be 0.
For each sequence, you will see the tokens in the sequence where the activations are left blank. You will print the exact same tokens verbatim, but with the activations filled in according to the explanation.
""",
        )

        few_shot_examples = self.few_shot_example_set.get_examples()
        for i, example in enumerate(few_shot_examples):
            few_shot_example_max_activation = calculate_max_activation(
                example.activation_records
            )

            tokens_without_activations = _format_record_for_logprob_free_simulation(
                example.activation_records[0], include_activations=False
            )
            prompt_builder.add_message(
                "user",  # type: ignore
                f"Neuron {i + 1}\nExplanation of neuron {i + 1} behavior: {EXPLANATION_PREFIX} "
                f"{example.explanation}\n\n"
                f"Sequence 1 Tokens without Activations:\n{tokens_without_activations}\n\n"
                f"Sequence 1 Tokens with Activations:\n",
            )
            tokens_with_activations = _format_record_for_logprob_free_simulation(
                example.activation_records[0],
                include_activations=True,
                max_activation=few_shot_example_max_activation,
            )
            prompt_builder.add_message(
                "assistant",  # type: ignore
                f"{tokens_with_activations}\n\n",
            )

            for record_index, record in enumerate(example.activation_records[1:]):
                tks_without = _format_record_for_logprob_free_simulation(
                    record, include_activations=False
                )
                prompt_builder.add_message(
                    "user",  # type: ignore
                    f"Sequence {record_index + 2} Tokens without Activations:\n{tks_without}\n\n"
                    f"Sequence {record_index + 2} Tokens with Activations:\n",
                )
                tokens_with_activations = _format_record_for_logprob_free_simulation(
                    record,
                    include_activations=True,
                    max_activation=few_shot_example_max_activation,
                )
                prompt_builder.add_message(
                    "assistant",  # type: ignore
                    f"{tokens_with_activations}\n\n",
                )

        neuron_index = len(few_shot_examples) + 1
        tokens_without_activations = _format_record_for_logprob_free_simulation(
            ActivationRecord(tokens=tokens, activations=[]), include_activations=False  # type: ignore
        )
        prompt_builder.add_message(
            "user",  # type: ignore
            f"Neuron {neuron_index}\nExplanation of neuron {neuron_index} behavior: {EXPLANATION_PREFIX} "
            f"{explanation}\n\n"
            f"Sequence 1 Tokens without Activations:\n{tokens_without_activations}\n\n"
            f"Sequence 1 Tokens with Activations:\n",
        )
        return prompt_builder.build(
            self.prompt_format, allow_extra_system_messages=True
        )
