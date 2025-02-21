import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np

from delphi.clients.client import Client

from .dataclasses import SequenceSimulation

from ..prompts import (
    build_prompt_all_at_once,
    build_simulation_prompt_json,
)

logger = logging.getLogger(__name__)

# Edge Case #3: The chat-based simulator is confused by end token. Replace it with
# a "not end token"
END_OF_TEXT_TOKEN = "<|endoftext|>"
END_OF_TEXT_TOKEN_REPLACEMENT = "<|not_endoftext|>"


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
    ).item()


def parse_top_logprobs(top_logprobs: dict[str, float]) -> OrderedDict[int, float]:
    """
    Given a map from tokens to logprobs, return a map from distribution values
    (integers on the range [0, 10]) to unnormalized probabilities
    (in the sense that they may not sum to 1).
    """
    probabilities_by_distribution_value = OrderedDict()
    for token, contents in top_logprobs.items():
        logprob = contents.logprob
        decoded_token = contents.decoded_token
        # check if token is a number
        str_nums = [str(i) for i in range(0, 10)]
        if decoded_token in str_nums:
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
            distribution_value: float(p / total_p_of_distribution_values)
            for distribution_value, p in probabilities_by_distribution_value.items()
        }
    )
    expected_value = compute_expected_value(norm_probabilities_by_distribution_value)
    return (
        norm_probabilities_by_distribution_value,
        expected_value,
    )


def pad_expected_values(
    expected_values: list[float],
    distribution_values: list[list[int]],
    distribution_probabilities: list[list[float]],
    tokens: list[str],
) -> tuple[list[float], list[list[int]], list[list[float]]]:
    if len(expected_values) > len(tokens):
        expected_values = expected_values[: len(tokens)]
        distribution_values = distribution_values[: len(tokens)]
        distribution_probabilities = distribution_probabilities[: len(tokens)]
    if len(expected_values) < len(tokens):
        expected_values = expected_values + [0] * (len(tokens) - len(expected_values))
    return expected_values, distribution_values, distribution_probabilities


def get_tab_positions_from_prompt_logprobs(prompt_logprobs: list, tokens: list[str]):
    # The token from the prompt is always the first token of the dict
    # we want to count each tab after the first <start>
    candidate = ""
    target_sentence = "<start>"
    tab_token_positions = []
    # We are going to do two loops because it is easier
    for i, logprob_dict in enumerate(prompt_logprobs):
        if logprob_dict is None:
            continue
        list_logprobs = list(logprob_dict.keys())
        token_id = list_logprobs[0]
        decoded_token = logprob_dict[token_id].decoded_token
        # remove any special tokens
        candidate += decoded_token.replace("\n", "").replace("\t", "")

        if candidate not in target_sentence:
            candidate = ""
        if candidate == target_sentence:
            # we only care about the tabs after the last <start>
            tab_token_positions = []
        if decoded_token == "\t":
            tab_token_positions.append(i)

    return tab_token_positions


def parse_simulation_response(
    response: dict[str, Any],
    tokens: list[str],
) -> SequenceSimulation:
    """
    Parse an API response to a simulation prompt.

    Args:
        response: response from the client
        tokens: list of tokens as strings in the sequence where the neuron
        is being simulated
    """
    logprobs = response.prompt_logprobs
    tab_token_positions = get_tab_positions_from_prompt_logprobs(logprobs, tokens)
    expected_values = []
    distribution_values = []
    distribution_probabilities = []

    for tab_indice in tab_token_positions:
        (
            p_by_distribution_value,
            expected_value,
        ) = compute_predicted_activation_stats_for_token(
            logprobs[tab_indice + 1],
        )
        distribution_values.append(list(p_by_distribution_value.keys()))
        distribution_probabilities.append(list(p_by_distribution_value.values()))
        expected_values.append(expected_value)

    # (gpaulo) If there are more values than tokens we truncate the values
    # and if there are less values than tokens we pad the values with 0s
    (expected_values, distribution_values, distribution_probabilities) = (
        pad_expected_values(
            expected_values, distribution_values, distribution_probabilities, tokens
        )
    )

    return SequenceSimulation(
        tokens=list(tokens),
        expected_activations=expected_values,
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
    ):
        self.client = client
        self.explanation = explanation

    async def simulate(
        self,
        tokens: list[str],
    ) -> SequenceSimulation:
        prompt = build_prompt_all_at_once(tokens, self.explanation)
        sampling_params: dict[str, Any] = {
            "max_tokens": 1,
            "prompt_logprobs": 15,
        }

        response = await self.client.generate(prompt, **sampling_params)

        logger.debug("response in score_explanation_by_activations is %s", response)
        try:
            result = parse_simulation_response(response, tokens)
            logger.debug("result in score_explanation_by_activations is %s", result)
            return result
        except Exception as e:
            logger.error(f"Simulation response parsing failed: {e}")
            return SequenceSimulation(
                tokens=list(tokens),
                expected_activations=[0] * len(tokens),
                distribution_values=[],
                distribution_probabilities=[],
            )


def _parse_no_logprobs_completion_json(
    completion,
    tokens: list[str],
) -> list[float]:
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
        # The model likes to bable. Get completion only after the first {
        completion = "{" + "".join(completion.split("{")[1:])
        json_completion = json.loads(completion)
        if "activations" not in json_completion:
            logger.error(
                "The key 'activations' is not in the completion:\n%s",
                json.dumps(completion),
            )
            return zero_prediction
        activations = json_completion["activations"]
        if len(activations) != len(tokens):
            logger.error(
                "Tokens and activations length did not match:\n%s\n%s",
                len(activations),
                len(tokens),
            )
            # Pad with zeros or truncate to match length
            if len(activations) < len(tokens):
                activations.extend(
                    [{"token": "", "activation": 0}] * (len(tokens) - len(activations))
                )
            else:
                activations = activations[: len(tokens)]
        predicted_activations = []
        # check that there is a token and activation value
        # no need to double check the token matches exactly
        for i, activation in enumerate(activations):
            if "token" not in activation:
                logger.error(
                    "The key 'token' is not in activation:\n%s",
                    activation,
                )
                predicted_activations.append(0)
                continue
            if "activation" not in activation:
                logger.error(
                    "The key 'activation' is not in activation:\n%s",
                    activation,
                )
                predicted_activations.append(0)
                continue
            # Ensure activation value is between 0-10 inclusive
            try:
                predicted_activation_float = float(activation["activation"])
                if predicted_activation_float < 0 or predicted_activation_float > 10:
                    logger.error(
                        "activation value out of range: %s",
                        predicted_activation_float,
                    )
                    predicted_activations.append(
                        max(0, min(10, predicted_activation_float))
                    )
                else:
                    predicted_activations.append(predicted_activation_float)
            except ValueError:
                logger.error(
                    "activation value invalid: %s",
                    activation["activation"],
                )
                predicted_activations.append(0)
            except TypeError:
                logger.error(
                    "activation value incorrect type: %s",
                    activation["activation"],
                )
                predicted_activations.append(0)
        logger.debug("predicted activations: %s", predicted_activations)
        return predicted_activations

    except json.JSONDecodeError:
        # Json error -> return empty list so that the simulation is discarded
        logger.warning(
            "Failed to parse completion JSON:\n%s",
            completion,
        )
        return []


class LogprobFreeExplanationTokenSimulator(NeuronSimulator):
    """
    Simulate neuron behavior based on an explanation.

    Unlike ExplanationNeuronSimulator and ExplanationTokenByTokenSimulator, this class does not rely on
    logprobs to calculate expected activations. Instead, it uses a few-shot prompt that displays all of the
    tokens at once, and request that the model replies in a json format. Also, each activation for a token
    is a function of all the activations that came previously and all of the tokens in the sequence, not
    just the current and previous tokens.
    """

    def __init__(
        self,
        client,
        explanation: str,
        temperature: float = 0.0,
    ):

        self.client = client
        self.explanation = explanation
        self.temperature = temperature

    async def simulate(self, tokens: list[str]) -> SequenceSimulation:

        prompt = build_simulation_prompt_json(
            tokens,
            self.explanation,
        )
        try:
            response = await self.client.generate(
                prompt, max_tokens=1000, temperature=self.temperature
            )

            predicted_activations = _parse_no_logprobs_completion_json(
                response.text, tokens
            )
        except Exception as e:
            logger.error(f"Error simulating neuron behavior: {e}")
            predicted_activations = []

        result = SequenceSimulation(
            expected_activations=predicted_activations,
            # We are getting a single prediction for each token, so we don't have a distribution.
            distribution_values=[],
            distribution_probabilities=[],
            tokens=tokens,
        )
        logger.debug("result in score_explanation_by_activations is %s", result)
        return result
