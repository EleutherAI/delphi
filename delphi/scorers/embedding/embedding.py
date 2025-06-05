import asyncio
import random
from dataclasses import dataclass
from typing import NamedTuple

from transformers import PreTrainedTokenizer

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
        tokenizer: PreTrainedTokenizer | None = None,
        verbose: bool = False,
        **generation_kwargs,
    ):
        self.model = model
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs

    async def __call__(  # type: ignore
        self,  # type: ignore
        record: LatentRecord,  # type: ignore
    ) -> ScorerResult:  # type: ignore
        samples = self._prepare(record)

        random.shuffle(samples)
        results = self._query(
            record.explanation,
            samples,
        )

        return ScorerResult(record=record, score=results)

    def call_sync(self, record: LatentRecord) -> list[EmbeddingOutput]:
        return asyncio.run(self.__call__(record))  # type: ignore

    def _prepare(self, record: LatentRecord) -> list[Sample]:
        """
        Prepare and shuffle a list of samples for classification.
        """
        samples = []

        if record.extra_examples is not None:
            samples.extend(
                examples_to_samples(
                    record.extra_examples,
                    tokenizer=self.tokenizer,
                    distance=-1,
                )
            )

        for i, example in enumerate(record.test):
            samples.extend(
                examples_to_samples(
                    [example],
                    tokenizer=self.tokenizer,
                    distance=i + 1,
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

        # # Temporary batching
        # sample_embedings = []
        # for i in range(0, len(samples_text), 10):
        #     sample_embedings.extend(self.model.encode(samples_text[i:i+10]))
        sample_embedings = self.model.encode(samples_text)
        similarity = self.model.similarity(query_embeding, sample_embedings)[0]

        results = []
        for i in range(len(samples)):
            # print(i)
            samples[i].data.similarity = similarity[i].item()
            results.append(samples[i].data)
        return results


def examples_to_samples(
    examples: list[Example],
    tokenizer: PreTrainedTokenizer | None,
    **sample_kwargs,
) -> list[Sample]:
    samples = []
    for example in examples:
        if tokenizer is not None:
            text = "".join(tokenizer.batch_decode(example.tokens))
        else:
            text = "".join(str(token) for token in example.tokens)
        activations = example.activations.tolist()
        samples.append(
            Sample(
                text=text,
                activations=activations,
                data=EmbeddingOutput(text=text, **sample_kwargs),
            )
        )

    return samples
