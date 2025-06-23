import asyncio
import random
from dataclasses import dataclass
from typing import NamedTuple, Sequence

from delphi.latents.latents import ActivatingExample, NonActivatingExample

from ...latents import Example, LatentRecord
from ..scorer import Scorer, ScorerResult


@dataclass
class EmbeddingOutput:
    text: str
    """The text that was used to evaluate the similarity"""

    distance: float | int
    """Quantile or neighbor distance"""

    similarity: float = 0
    """What is the similarity of the example to the explanation"""


class Sample(NamedTuple):
    text: str
    activations: list[float]
    data: EmbeddingOutput


class EmbeddingScorer(Scorer):
    name = "embedding"

    def __init__(
        self,
        model,
        verbose: bool = False,
        **generation_kwargs,
    ):
        self.model = model
        self.verbose = verbose
        self.generation_kwargs = generation_kwargs

    async def __call__(
        self,
        record: LatentRecord,
    ) -> ScorerResult:
        samples = self._prepare(record)

        random.shuffle(samples)
        results = self._query(
            record.explanation,
            samples,
        )

        return ScorerResult(record=record, score=results)

    def call_sync(self, record: LatentRecord) -> ScorerResult:
        return asyncio.run(self.__call__(record))

    def _prepare(self, record: LatentRecord) -> list[Sample]:
        """
        Prepare and shuffle a list of samples for classification.
        """
        samples = []

        assert (
            record.extra_examples is not None
        ), "Extra (non-activating) examples need to be provided"

        samples.extend(
            examples_to_samples(
                record.extra_examples,
            )
        )

        samples.extend(
            examples_to_samples(
                record.test,
            )
        )

        return samples

    def _query(self, explanation: str, samples: list[Sample]) -> list[EmbeddingOutput]:
        explanation_string = (
            "Instruct: Retrieve sentences that could be related to the explanation."
            "\nQuery:"
        )
        explanation_prompt = explanation_string + explanation
        query_embeding = self.model.encode(explanation_prompt)
        samples_text = [sample.text for sample in samples]

        sample_embedings = self.model.encode(samples_text)
        similarity = self.model.similarity(query_embeding, sample_embedings)[0]

        results = []
        for i in range(len(samples)):
            samples[i].data.similarity = similarity[i].item()
            results.append(samples[i].data)
        return results


def examples_to_samples(
    examples: Sequence[Example],
) -> list[Sample]:
    samples = []
    for example in examples:
        assert isinstance(example, ActivatingExample) or isinstance(
            example, NonActivatingExample
        )
        assert example.str_tokens is not None
        text = "".join(str(token) for token in example.str_tokens)
        activations = example.activations.tolist()
        samples.append(
            Sample(
                text=text,
                activations=activations,
                data=EmbeddingOutput(
                    text=text,
                    distance=(
                        example.quantile
                        if isinstance(example, ActivatingExample)
                        else example.distance
                    ),
                ),
            )
        )

    return samples
