"""
OpenAI-style simulator scorer wrapper.

This module exposes a scorer compatible with the library's scoring pipeline. It
converts latent examples into activation records, runs the unified neuron
simulator and scoring utilities, and returns results in the standard scoring
format.
"""

from typing import Sequence

import torch

from delphi.latents.latents import ActivatingExample, NonActivatingExample
from delphi.scorers.scorer import Scorer, ScorerResult

from .data_models import ActivationRecord
from .scoring import simulate_and_score
from .simulator import NeuronSimulator


class RefactoredOpenAISimulator(Scorer):
    """Scorer that simulates activations from an explanation and scores them.

    The scorer builds a ``NeuronSimulator`` in either logprob or JSON mode,
    executes simulations for each provided example, aggregates results by
    quantile, and returns the standard scoring structures expected by the
    pipeline.
    """

    name = "simulator"

    def __init__(self, client, tokenizer, all_at_once=True):
        """Initialize the simulator.

        Args:
            client: LLM client for inference
            tokenizer: Tokenizer for converting tokens
            all_at_once: If True, use logprob mode; if False, use JSON mode
        """
        self.client = client
        self.tokenizer = tokenizer
        self.all_at_once = all_at_once

    async def __call__(self, record) -> ScorerResult:
        """Simulate activations for the record and return scoring results.

        Args:
            record: Latent record providing the explanation and examples to
                evaluate.

        Returns:
            ``ScorerResult`` containing standard scoring structures.
        """
        # Build simulator in requested mode
        simulator = NeuronSimulator(
            self.client, record.explanation, use_logprobs=self.all_at_once
        )

        # Convert examples to activation records
        activation_records = self.to_activation_records(record.test)
        non_activation_records = (
            self.to_activation_records(record.not_active)
            if len(record.not_active) > 0
            else []
        )

        # Run simulation and aggregate scores
        legacy_results = await simulate_and_score(
            simulator, activation_records, non_activation_records
        )

        return ScorerResult(record=record, score=legacy_results)

    def to_activation_records(
        self, examples: Sequence[ActivatingExample | NonActivatingExample]
    ) -> list[ActivationRecord]:
        """Convert latent examples into activation records.

        Non-activating examples are mapped to zero-valued activations aligned
        with their tokens. Activating examples use their normalized activations.

        Args:
            examples: Activating or non-activating latent examples.

        Returns:
            A list of ``ActivationRecord`` instances.
        """
        result = []
        for example in examples:
            if example is None:
                continue

            if isinstance(example, NonActivatingExample):
                # Use zeros for non-activating examples
                activations: list[int] = torch.zeros_like(example.activations).tolist()
            else:
                assert example.normalized_activations is not None
                activations: list[int] = example.normalized_activations.tolist()

            result.append(
                ActivationRecord(
                    self.tokenizer.batch_decode(example.tokens),
                    activations,
                    quantile=(
                        example.quantile
                        if isinstance(example, ActivatingExample)
                        else -1
                    ),
                )
            )

        return result
