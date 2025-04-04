import asyncio
import random
from dataclasses import dataclass
from typing import Literal

import torch

from ...latents import LatentRecord, NonActivatingExample
from ..classifier.sample import _prepare_text
from ..scorer import Scorer, ScorerResult


@dataclass
class Batch:
    """
    A set of positive and negative examples to be used for scoring,
    as well as a positive and negative query.
    """

    negative_examples: list[str]
    """Non-activating examples used for negative explanation"""

    positive_examples: list[str]
    """Activating examples used for positive explanation"""

    positive_query: str
    """Activating example used for positive query"""

    negative_query: str
    """Non-activating example used for negative query"""

    quantile_positive_query: int
    """Quantile of the positive query"""

    distance_negative_query: float
    """Distance of the negative query"""


@dataclass
class EmbeddingOutput:
    """
    The output of the embedding scorer.
    """

    batch: Batch
    """The set of examples and queries used for scoring"""

    delta_plus: float = 0
    """The difference in similarity between the positive query
    and the positive examples, and the positive query and the negative examples"""

    delta_minus: float = 0
    """The difference in similarity between the negative query
    and the positive examples, and the negative query and the negative examples"""


class ExampleEmbeddingScorer(Scorer):
    """
    This scorer does not use explanations to score the examples.
    Instead it embeds examples.


    """

    name = "example_embedding"

    def __init__(
        self,
        model,
        verbose: bool = False,
        method: Literal["default", "internal"] = "default",
        number_batches: int = 20,
        seed: int = 42,
        **generation_kwargs,
    ):
        self.model = model
        self.verbose = verbose
        self.generation_kwargs = generation_kwargs
        self.method = method
        self.number_batches = number_batches
        self.random = random.Random(seed)

    async def __call__(
        self,
        record: LatentRecord,
    ) -> ScorerResult:

        # Create tasks with the positive and negative test examples
        batches = self._create_batches(record, number_batches=self.number_batches)

        # Compute the probability of solving the task for each task
        delta_tuples = [self.compute_batch_deltas(batch) for batch in batches]
        score = [
            EmbeddingOutput(batch=batch, delta_plus=delta_plus, delta_minus=delta_minus)
            for batch, (delta_plus, delta_minus) in zip(batches, delta_tuples)
        ]

        return ScorerResult(record=record, score=score)

    def call_sync(self, record: LatentRecord) -> ScorerResult:
        return asyncio.run(self.__call__(record))

    def compute_batch_deltas(self, batch: Batch) -> tuple[float, float]:
        """
        Compute the probability of solving the task.
        """
        with torch.no_grad():
            # Use the embedding model to embed all the examples
            # Concatenate all inputs into a single list
            all_inputs = (
                batch.negative_examples
                + batch.positive_examples
                + [batch.positive_query, batch.negative_query]
            )
            # Encode everything at once
            all_embeddings = self.model.encode(all_inputs)

            # Split the embeddings back into their components
            n_neg = len(batch.negative_examples)
            n_pos = len(batch.positive_examples)
            negative_examples_embeddings = all_embeddings[:n_neg]
            positive_examples_embeddings = all_embeddings[n_neg : n_neg + n_pos]
            positive_query_embedding = all_embeddings[-2].unsqueeze(0)
            negative_query_embedding = all_embeddings[-1].unsqueeze(0)

            # Compute the similarity between the query and the examples
            negative_similarities = self.model.similarity(
                negative_query_embedding,
                torch.cat([negative_examples_embeddings, positive_examples_embeddings]),
            )
            negative_negative_similarity = negative_similarities[
                :, : len(negative_examples_embeddings)
            ]
            negative_positive_similarity = negative_similarities[
                :, len(negative_examples_embeddings) :
            ]

            positive_similarities = self.model.similarity(
                positive_query_embedding,
                torch.cat([negative_examples_embeddings, positive_examples_embeddings]),
            )
            positive_negative_similarity = positive_similarities[
                :, : len(negative_examples_embeddings)
            ]
            positive_positive_similarity = positive_similarities[
                :, len(negative_examples_embeddings) :
            ]

            delta_positive = (
                positive_positive_similarity.mean()
                - positive_negative_similarity.mean()
            )
            delta_negative = (
                negative_positive_similarity.mean()
                - negative_negative_similarity.mean()
            )

        return delta_positive.item(), delta_negative.item()

    def _create_batches(
        self, record: LatentRecord, number_batches: int = 20
    ) -> list[Batch]:

        # Get the positive and negative train examples,
        # which are going to be used as "explanations"
        positive_train_examples = record.train

        # Sample from the not_active examples
        not_active_index = self.random.sample(
            range(len(record.not_active)), len(positive_train_examples)
        )
        negative_train_examples = [record.not_active[i] for i in not_active_index]

        # Get the positive and negative test examples,
        # which are going to be used as "queries"
        positive_test_examples = record.test

        not_active_test_index = [
            i for i in range(len(record.not_active)) if i not in not_active_index
        ]
        negative_test_examples = [record.not_active[i] for i in not_active_test_index]

        batches = []

        for _ in range(number_batches):
            # Prepare the positive query
            positive_query_idx = self.random.sample(
                range(len(positive_test_examples)), 1
            )[0]
            positive_query = positive_test_examples[positive_query_idx]
            n_active_tokens = int((positive_query.activations > 0).sum().item())
            positive_query_str, _ = _prepare_text(
                positive_query, n_incorrect=0, threshold=0.3, highlighted=True
            )
            # Prepare the negative query
            if self.method == "default":
                # In the default method, we just sample a random negative example
                negative_query_idx = self.random.sample(
                    range(len(negative_test_examples)), 1
                )[0]
                negative_query = negative_test_examples[negative_query_idx]
                negative_query_str, _ = _prepare_text(
                    negative_query,
                    n_incorrect=n_active_tokens,
                    threshold=0.3,
                    highlighted=True,
                )
            elif self.method == "internal":
                # In the internal method, we sample a negative example
                # that has a different quantile as the positive query
                positive_query_quantile = positive_query.quantile
                negative_query_quantile = positive_query_quantile
                # TODO: This is kinda ugly, but it probably doesn't matter
                while negative_query_quantile == positive_query_quantile:
                    negative_query_idx = self.random.sample(
                        range(len(positive_test_examples)), 1
                    )[0]
                    negative_query_temp = positive_test_examples[negative_query_idx]
                    negative_query_quantile = negative_query.distance

                negative_query = NonActivatingExample(
                    str_tokens=negative_query_temp.str_tokens,
                    tokens=negative_query_temp.tokens,
                    activations=negative_query_temp.activations,
                    distance=negative_query_temp.quantile,
                )
                # Because it is a converted activating example, it will highlight
                # the activating tokens
                negative_query_str, _ = _prepare_text(
                    negative_query, n_incorrect=0, threshold=0.3, highlighted=True
                )

            # Find all all the positive_train_examples
            # that have the same quantile as the positive_query
            positive_examples = [
                e
                for e in positive_train_examples
                if e.quantile == positive_query.quantile
            ]
            if len(positive_examples) > 10:
                positive_examples = self.random.sample(positive_examples, 10)
            positive_examples_str = [
                _prepare_text(e, n_incorrect=0, threshold=0.3, highlighted=True)[0]
                for e in positive_examples
            ]

            # negative examples
            if self.method == "default":
                # In the default method, we just sample a random negative example
                negative_examples = self.random.sample(negative_train_examples, 10)
                negative_examples_str = [
                    _prepare_text(
                        e, n_incorrect=n_active_tokens, threshold=0.3, highlighted=True
                    )[0]
                    for e in negative_examples
                ]
            elif self.method == "internal":
                # In the internal method, we sample an activating example
                # that has the same quantile as the negative_query
                negative_examples = [
                    e
                    for e in positive_train_examples
                    if e.quantile == negative_query.distance
                ]
                if len(negative_examples) > 10:
                    negative_examples = self.random.sample(negative_examples, 10)
                negative_examples_str = [
                    _prepare_text(e, n_incorrect=0, threshold=0.3, highlighted=True)[0]
                    for e in negative_examples
                ]

            batch = Batch(
                negative_examples=negative_examples_str,
                positive_examples=positive_examples_str,
                positive_query=positive_query_str,
                negative_query=negative_query_str,
                quantile_positive_query=positive_query.quantile,
                distance_negative_query=negative_query.distance,
            )
            batches.append(batch)
        return batches
