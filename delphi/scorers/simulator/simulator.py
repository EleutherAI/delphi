from ..scorer import Scorer, ScorerResult
from .scoring import (
    ExplanationNeuronSimulator,
    LogprobFreeExplanationTokenSimulator,
    simulate_and_score,
)


class OpenAISimulator(Scorer):
    """
    Simple wrapper for the the different simulators.
    LogprobFreeExplanationTokenSimulator heavily inspired by
    https://github.com/hijohnnylin/automated-interpretability,
    which in turn is a fork of https://github.com/openai/automated-interpretability,
    from which the ExplanationNeuronSimulator is also inspired.
    """

    name = "simulator"

    def __init__(
        self,
        client,
        all_at_once=True,
    ):
        self.client = client
        self.all_at_once = all_at_once

    async def __call__(self, record):
        # Simulate and score the explanation.
        cls = (
            ExplanationNeuronSimulator
            if self.all_at_once
            else LogprobFreeExplanationTokenSimulator
        )
        simulator = cls(
            self.client,
            record.explanation,
        )

        valid_activation_records = record.test
        if len(record.not_active) > 0:
            non_activation_records = record.not_active
        else:
            non_activation_records = []

        result = await simulate_and_score(
            simulator, valid_activation_records, non_activation_records
        )

        return ScorerResult(
            record=record,
            score=result,
        )
