from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import dspy
from dspy.utils.exceptions import AdapterParseError

from delphi.logger import logger

from ...latents import LatentRecord
from ..scorer import ScorerResult


@dataclass
class FeatureExample:
    """A text snippet that might activate a feature."""

    text: str
    activating: bool
    correct: Optional[bool] = None


class Classifier:
    name = ""

    def __init__(
        self,
        scorer: dspy.Predict,
    ):
        self.scorer = scorer

    @abstractmethod
    def _prepare_and_batch(self, record: LatentRecord) -> list[list[FeatureExample]]:
        pass

    @abstractmethod
    def _check_correct(self, example: FeatureExample, is_feature: bool) -> bool:
        pass

    async def __call__(self, record: LatentRecord):
        batched = self._prepare_and_batch(record)
        results = []
        for batch in batched:
            try:
                result = await self.scorer.acall(
                    feature_description=record.explanation, feature_examples=batch
                )
            except AdapterParseError or ValueError:
                logger.error("Error in explainer:")
                result = None
            if result is not None:
                for example, is_feature in zip(batch, result.is_feature):
                    example.correct = self._check_correct(example, is_feature)
            results.append(example)
        return ScorerResult(record, results)
