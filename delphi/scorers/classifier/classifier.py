import asyncio
import json
import random
import re
from abc import abstractmethod
from typing import Any

import numpy as np

from delphi import logger

from ...clients.client import Client, Response
from ...latents import LatentRecord
from ..scorer import Scorer, ScorerResult
from .sample import ClassifierOutput, Sample


class Classifier(Scorer):
    def __init__(
        self,
        client: Client,
        verbose: bool,
        n_examples_shown: int,
        seed: int = 42,
        log_prob: bool = False,
        **generation_kwargs,
    ):
        """
        Initialize a Classifier.

        Args:
            client: The client to use for generation
            tokenizer: The tokenizer used to cache the tokens
            verbose: Whether to print verbose output
            n_examples_shown: The number of examples to show in the prompt,
                        a larger number can both leak information and make
                        it harder for models to generate anwers in the correct format
            log_prob: Whether to use log probabilities to allow for AUC calculation
            generation_kwargs: Additional generation kwargs
        """
        if n_examples_shown < 1:
            raise ValueError("n_examples_shown must be positive")
        self.client = client
        self.verbose = verbose
        self.n_examples_shown = n_examples_shown
        self.generation_kwargs = generation_kwargs
        self.log_prob = log_prob
        self.rng = random.Random(seed)

    async def __call__(
        self,
        record: LatentRecord,
    ) -> ScorerResult:
        samples = self._prepare(record)
        self.rng.shuffle(samples)

        batched_samples = self._batch(samples)
        results = await self._query(
            record.explanation,
            batched_samples,
        )

        return ScorerResult(record=record, score=results)

    @abstractmethod
    def _prepare(self, record: LatentRecord) -> list[Sample]:
        pass

    async def _query(
        self,
        explanation: str,
        batches: list[list[Sample]],
    ) -> list[ClassifierOutput]:
        """
        Send and gather batches of samples to the model.
        """
        sem = asyncio.Semaphore(1)

        async def _process(explanation, batch):
            async with sem:
                result = await self._generate(explanation, batch)
                return result

        tasks = [asyncio.create_task(_process(explanation, batch)) for batch in batches]
        results = await asyncio.gather(*tasks)

        return sum(results, [])

    async def _generate(
        self, explanation: str, batch: list[Sample]
    ) -> list[ClassifierOutput]:
        """
        Generate predictions for a batch of samples.
        """

        prompt = self._build_prompt(explanation, batch)
        if self.log_prob:
            self.generation_kwargs["logprobs"] = True
            self.generation_kwargs["top_logprobs"] = 5
        try:
            response = await self.client.generate(prompt, **self.generation_kwargs)
        except Exception as e:
            logger.error(f"Error generating text: {repr(e)}")
            response = None
        if response is None:
            predictions = [None] * len(batch)
            probabilities = [None] * len(batch)
        else:
            assert isinstance(response, Response)
            selections = response.text
            logprobs = response.logprobs if self.log_prob else None
            try:
                predictions, probabilities = self._parse(
                    selections, logprobs, n_predictions=len(batch)
                )
            except Exception as e:
                logger.error(f"Parsing selections failed: {repr(e)}")
                predictions = [None] * len(batch)
                probabilities = [None] * len(batch)

        results = []
        for sample, prediction, probability in zip(batch, predictions, probabilities):
            result = sample.data
            result.prediction = bool(prediction) if prediction is not None else None
            if prediction is not None:
                result.correct = prediction == result.activating
            else:
                result.correct = None
            result.probability = probability
            results.append(result)

            if self.verbose:
                logger.info(
                    f"Example {sample.text}, "
                    f"Prediction: {prediction}, "
                    f"Probability: {probability}"
                )
        return results

    def _parse(
        self,
        string: str,
        logprobs: list[Any] | None = None,
        n_predictions: int | None = None,
    ) -> tuple[list[bool], list[float] | list[None]]:
        """Extract binary predictions and probabilities from a string and
        optionally its token logprobs."""
        # Matches the first instance of text enclosed in square brackets
        pattern = r"\[.*?\]"
        match = re.search(pattern, string)
        if match is None:
            raise ValueError("No match found in string")
        raw_predictions = json.loads(match.group(0))
        if any(
            type(value) not in (bool, int) or value not in (0, 1)
            for value in raw_predictions
        ):
            raise ValueError("Predictions must be binary integers or booleans")
        expected = self.n_examples_shown if n_predictions is None else n_predictions
        predictions = [bool(prediction) for prediction in raw_predictions]
        if len(predictions) != expected:
            raise ValueError(f"Expected {expected} predictions, got {len(predictions)}")
        probabilities = (
            self._parse_logprobs(logprobs, n_predictions=expected)
            if logprobs is not None
            else [None] * expected
        )

        return predictions, probabilities

    def _parse_logprobs(
        self, logprobs: list[Any], n_predictions: int | None = None
    ) -> list[float]:
        """
        Extracts normalized probabilities of '1' vs '0' tokens from the top n
        log probabilities for each token in a response string of form '[x, x, x, ...]'.
        The normalized probability is computed as P(1)/(P(0) + P(1)), where P(0) and
        P(1) are summed over all matching tokens in the top 5 candidates.

        Args:
            logprobs (list): Contains top n log probabilities for each token in the
            response.

        Returns:
            list: Normalized probabilities between 0 and 1 where each value represents
            P(token='1').
        """
        binary_probabilities: list[float] = []

        for i in range(len(logprobs)):
            if logprobs[i].token.strip() in ("0", "1"):
                top_logprobs = logprobs[i].top_logprobs
                prob_0 = 0.0
                prob_1 = 0.0
                for i in range(len(top_logprobs)):
                    token = top_logprobs[i].token
                    logprob = top_logprobs[i].logprob
                    if token.strip() == "0":
                        prob_0 += np.exp(logprob).item()
                    elif token.strip() == "1":
                        prob_1 += np.exp(logprob).item()
                if prob_0 + prob_1 > 0:
                    binary_probabilities.append(prob_1 / (prob_0 + prob_1))
                else:
                    binary_probabilities.append(0.0)

        expected = self.n_examples_shown if n_predictions is None else n_predictions
        if len(binary_probabilities) != expected:
            raise ValueError("Log probabilities do not match the prediction count")
        return binary_probabilities

    def _build_prompt(
        self,
        explanation: str,
        batch: list[Sample],
    ) -> list[dict]:
        """
        Prepare prompt for generation.
        """

        examples = "\n".join(
            f"Example {i}: {sample.text}" for i, sample in enumerate(batch)
        )

        return self.prompt(explanation=explanation, examples=examples)

    @abstractmethod
    def prompt(self, examples: str, explanation: str) -> list[dict]:
        pass

    def _batch(self, samples: list[Sample]) -> list[list[Sample]]:
        return [
            samples[i : i + self.n_examples_shown]
            for i in range(0, len(samples), self.n_examples_shown)
        ]

    def call_sync(self, record: LatentRecord) -> ScorerResult:
        return asyncio.run(self.__call__(record))
