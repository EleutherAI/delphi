import asyncio
import json
import random
import re
import traceback
from abc import abstractmethod
from typing import Literal

import numpy as np

from ...clients.client import Client, Response
from ...latents import LatentRecord
from ...logger import logger
from ..scorer import Scorer, ScorerResult
from .sample import ClassifierOutput, Sample


class Classifier(Scorer):
    def __init__(
        self,
        client: Client,
        verbose: bool,
        n_examples_shown: int,
        log_prob: bool,
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
        self.client = client
        self.verbose = verbose
        self.n_examples_shown = n_examples_shown
        self.generation_kwargs = generation_kwargs
        self.log_prob = log_prob

    async def __call__(  # type: ignore
        self,  # type: ignore
        record: LatentRecord,  # type: ignore
    ) -> ScorerResult:  # type: ignore
        samples = self._prepare(record)
        random.shuffle(samples)

        samples = self._batch(samples)
        results = await self._query(
            record.explanation,
            samples,
        )

        return ScorerResult(record=record, score=results)

    @abstractmethod
    def _prepare(self, record: LatentRecord) -> list[list[Sample]]:
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
            logger.error(f"Error generating text: {e}")
            response = None
        if response is None:
            predictions = [None] * self.n_examples_shown
            probabilities = [None] * self.n_examples_shown
        else:
            assert isinstance(response, Response)
            selections = response.text
            logprobs = response.logprobs if self.log_prob else None
            try:
                predictions, probabilities = self._parse(selections, logprobs)
            except Exception:
                logger.error("Parsing selections failed:\n" + traceback.format_exc())
                predictions = [None] * self.n_examples_shown
                probabilities = [None] * self.n_examples_shown

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

    def _parse(self, string, logprobs=None):
        """Extract binary predictions and probabilities from a string and
        optionally its token logprobs."""
        # Matches the first instance of text enclosed in square brackets
        pattern = r"\[.*?\]"
        match = re.search(pattern, string)
        if match is None:
            raise ValueError("No match found in string")
        predictions: list[bool | Literal[0, 1]] = json.loads(match.group(0))
        assert len(predictions) == self.n_examples_shown
        probabilities = (
            self._parse_logprobs(logprobs)
            if logprobs is not None
            else [None] * self.n_examples_shown
        )

        return predictions, probabilities

    def _parse_logprobs(self, logprobs: list):
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
            if "1" in logprobs[i].token or "0" in logprobs[i].token:
                top_logprobs = logprobs[i].top_logprobs
                prob_0 = 0.0
                prob_1 = 0.0
                for i in range(len(top_logprobs)):
                    token = top_logprobs[i].token
                    logprob = top_logprobs[i].logprob
                    if "0" in token:
                        prob_0 += np.exp(logprob).item()
                    elif "1" in token:
                        prob_1 += np.exp(logprob).item()
                if prob_0 + prob_1 > 0:
                    binary_probabilities.append(prob_1 / (prob_0 + prob_1))
                else:
                    binary_probabilities.append(0.0)

        assert len(binary_probabilities) == self.n_examples_shown
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

    def _batch(self, samples):
        return [
            samples[i : i + self.n_examples_shown]
            for i in range(0, len(samples), self.n_examples_shown)
        ]

    def call_sync(self, record: LatentRecord) -> ScorerResult:
        return asyncio.run(self.__call__(record))
