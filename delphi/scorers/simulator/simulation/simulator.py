"""
Unified neuron simulator.

This module implements a single simulator class capable of producing token-level
activation predictions from a natural-language explanation using two modes:

- Logprob mode: builds a few-shot prompt and derives per-token expected
  activations from prompt log-probabilities.
- JSON mode: builds a few-shot prompt that requests a JSON structure containing
  tokens with activation values; the JSON response is validated and parsed into
  numeric activations.

Both modes return a list of activation values aligned with the provided tokens
and include robust error handling to produce a well-formed result even when the
model output is malformed.
"""

import json
import logging
from typing import List, Dict, Any, Union, Optional

from .llm_interface import (
    Role,
    PromptBuilder,
    HarmonyMessage,
    PromptFormat,
    parse_simulation_response,
)
from .data_models import (
    format_activation_records,
    calculate_max_activation,
    format_sequences_for_simulation,
)
from .few_shot_examples import FewShotExampleSet
from .types import SequenceSimulation, ActivationScale

logger = logging.getLogger(__name__)

from .data_models import (
    MAX_NORMALIZED_ACTIVATION,
    VALID_ACTIVATION_TOKENS,
    END_OF_TEXT_TOKEN,
    END_OF_TEXT_TOKEN_REPLACEMENT,
    EXPLANATION_PREFIX,
    PROMPT_LOGPROBS_COUNT,
    JSON_MAX_TOKENS,
)


class NeuronSimulator:
    """Simulator for token-level activation prediction.

    The simulator infers activations for a sequence of tokens given a textual
    explanation of the neuron's behavior. The behavior is controlled by a mode
    flag:

    - ``use_logprobs=True``: compute expected activations from
      prompt log-probabilities over discretized activation tokens.
    - ``use_logprobs=False``: request a JSON response that directly provides
      activation values per token.

    The returned predictions have the same length and order as the input tokens.
    """

    def __init__(self, client: Any, explanation: str, use_logprobs: bool = True):
        """Initialize the simulator.

        Args:
            client: LLM client for inference
            explanation: Text explanation of neuron behavior
            use_logprobs: If True, use logprob-based simulation; if False, use JSON mode
        """
        self.client = client
        self.explanation = explanation
        self.use_logprobs = use_logprobs
        self.few_shot_example_set = (
            FewShotExampleSet.ORIGINAL if use_logprobs else FewShotExampleSet.NEWER
        )
        self.prompt_format = PromptFormat.HARMONY_V4

    async def simulate(self, tokens: List[str]):
        """Simulate neuron activations for the given tokens.

        Args:
            tokens: List of token strings to simulate

        Returns:
            SequenceSimulation object with predicted activations
        """
        try:
            if self.use_logprobs:
                return await self._simulate_with_logprobs(tokens)
            else:
                return await self._simulate_with_json(tokens)
        except Exception as e:
            logger.warning(
                f"Simulation failed: {e}, returning zeros for tokens: {tokens}"
            )
            return SequenceSimulation(
                tokens=list(tokens),
                expected_activations=[0.0] * len(tokens),
                activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
                distribution_values=[],
                distribution_probabilities=[],
                uncalibrated_simulation=None,
            )

    async def _simulate_with_logprobs(self, tokens: List[str]):
        """Simulate using log-probabilities from a few-shot prompt.

        Builds a prompt containing few-shot examples and the target token
        sequence with unknown activations, requests prompt log-probabilities for
        discretized activation tokens, and converts them into expected values
        per token.

        Returns:
            SequenceSimulation object with predicted activations
        """
        prompt = self._make_logprob_prompt(tokens)
        sampling_params: dict[str, Any] = {
            "max_tokens": 1,
            "prompt_logprobs": PROMPT_LOGPROBS_COUNT,
        }

        response = await self.client.generate(prompt, **sampling_params)
        tokenized_prompt = self.client.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True
        )
        # Use [1] to skip BOS token that tokenizer.encode() may prepend
        tab_token = self.client.tokenizer.encode("\t")[1]

        result = parse_simulation_response(
            response, tokenized_prompt, tab_token, tokens
        )
        return result

    async def _simulate_with_json(self, tokens: List[str]):
        """Simulate using a JSON response from a few-shot prompt.

        Builds a prompt that asks for a JSON payload containing the exact input
        tokens and their corresponding activation values. The response is
        validated and parsed; invalid or mismatched responses yield a zero vector
        aligned with the input tokens.

        Returns:
            SequenceSimulation object with predicted activations
        """
        prompt = self._make_json_prompt(tokens)

        response = await self.client.generate(prompt, max_tokens=JSON_MAX_TOKENS)
        activations = self._parse_json_response(response, tokens)

        return SequenceSimulation(
            tokens=list(tokens),
            expected_activations=activations,
            activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
            distribution_values=[],
            distribution_probabilities=[],
            uncalibrated_simulation=None,
        )

    def _make_logprob_prompt(self, tokens: List[str]) -> Any:
        """Create a few-shot prompt for logprob-based activation prediction."""
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            """We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.
Look at summary of what the neuron does, and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to 10, "unknown" indicates an unknown activation. Most activations will be 0.
""",
        )

        few_shot_examples = self.few_shot_example_set.get_examples()
        for i, example in enumerate(few_shot_examples):
            prompt_builder.add_message(
                Role.USER,
                f"\n\nNeuron {i + 1}\nExplanation of neuron {i + 1} behavior: {EXPLANATION_PREFIX}"
                f"{example.explanation}",
            )
            formatted_activation_records = format_activation_records(
                example.activation_records,
                calculate_max_activation(example.activation_records),
                start_indices=example.first_revealed_activation_indices,
            )
            prompt_builder.add_message(
                Role.ASSISTANT, f"\nActivations: {formatted_activation_records}\n"
            )

        prompt_builder.add_message(
            Role.USER,
            f"\n\nNeuron {len(few_shot_examples) + 1}\nExplanation of neuron "
            f"{len(few_shot_examples) + 1} behavior: {EXPLANATION_PREFIX} "
            f"{self.explanation.strip()}",
        )
        prompt_builder.add_message(
            Role.ASSISTANT,
            f"\nActivations: {format_sequences_for_simulation([tokens])}",
        )
        return prompt_builder.build(self.prompt_format)

    def _make_json_prompt(self, tokens: List[str]) -> Any:
        """Create a few-shot prompt that requests a structured JSON response."""
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            """We're studying neurons in a neural network. Each neuron looks for certain things in a short document. Your task is to read the explanation of what the neuron does, and predict the neuron's activations for each token in the document.

For each document, you will see the full text of the document, then the tokens in the document with the activation left blank. You will print, in valid json, the exact same tokens verbatim, but with the activation values filled in according to the explanation. Pay special attention to the explanation's description of the context and order of tokens or words.

Fill out the activation values with integer values from 0 to 10. Don't use negative numbers. Please think carefully.""",
        )

        # Add few-shot examples for JSON mode
        few_shot_examples = self.few_shot_example_set.get_examples()
        for example in few_shot_examples:
            prompt_builder.add_message(
                Role.USER,
                json.dumps(
                    {
                        "to_find": example.explanation,
                        "document": "".join(example.activation_records[0].tokens),
                        "activations": [
                            {"token": t, "activation": None}
                            for t in example.activation_records[0].tokens
                        ],
                    }
                ),
            )
            prompt_builder.add_message(
                Role.ASSISTANT,
                json.dumps(
                    {
                        "to_find": example.explanation,
                        "document": "".join(example.activation_records[0].tokens),
                        "activations": [
                            {"token": t, "activation": a}
                            for t, a in zip(
                                example.activation_records[0].tokens,
                                example.activation_records[0].activations,
                            )
                        ],
                    }
                ),
            )

        prompt_builder.add_message(
            Role.USER,
            json.dumps(
                {
                    "to_find": self.explanation,
                    "document": "".join(tokens),
                    "activations": [{"token": t, "activation": None} for t in tokens],
                }
            ),
        )
        return prompt_builder.build(
            self.prompt_format, allow_extra_system_messages=True
        )

    def _parse_json_response(self, completion: Any, tokens: List[str]) -> List[float]:
        """Parse and validate a JSON response to extract activations.

        The parser ensures the presence of the ``activations`` field, enforces
        alignment with the input tokens, and validates activation values to be
        numeric within the inclusive range [0, 10]. Any invalid structure or
        value results in a zero vector with the same length as ``tokens``.
        """
        zero_prediction = [0.0] * len(tokens)
        try:
            completion_dict = json.loads(completion.text)
            if "activations" not in completion_dict:
                logger.error(
                    f"JSON response missing 'activations' key: {completion.text}"
                )
                return zero_prediction

            activations_data = completion_dict["activations"]
            if len(activations_data) != len(tokens):
                logger.error(
                    f"JSON response activations length mismatch. Expected {len(tokens)}, got {len(activations_data)}"
                )
                return zero_prediction

            predicted_activations = []
            for activation_entry in activations_data:
                if (
                    "token" not in activation_entry
                    or "activation" not in activation_entry
                ):
                    logger.error(f"Malformed activation entry: {activation_entry}")
                    predicted_activations.append(0.0)
                    continue

                try:
                    predicted_activation_float = float(activation_entry["activation"])
                    if not (
                        0 <= predicted_activation_float <= MAX_NORMALIZED_ACTIVATION
                    ):
                        logger.error(
                            f"Activation value out of range [0,10]: {predicted_activation_float}"
                        )
                        predicted_activations.append(0.0)
                    else:
                        predicted_activations.append(predicted_activation_float)
                except (ValueError, TypeError):
                    logger.error(
                        f"Invalid activation value type: {activation_entry['activation']}"
                    )
                    predicted_activations.append(0.0)

            return predicted_activations

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse completion JSON: {completion.text}")
            return zero_prediction


# Backward compatibility alias
ExplanationNeuronSimulator = NeuronSimulator
