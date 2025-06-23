import asyncio
import re
from dataclasses import dataclass
from typing import Literal

from beartype.typing import Sequence

from delphi import logger

from ...clients.client import Client
from ...latents import ActivatingExample, Example, LatentRecord, NonActivatingExample
from .classifier import Classifier, ScorerResult
from .prompts.intruder_prompt import prompt as intruder_prompt
from .sample import _prepare_text


@dataclass
class IntruderSentence:
    """
    A sample for an intruder sentence experiment.
    """

    examples: list[str]
    intruder_index: int
    chosen_quantile: int
    activations: list[list[float]]
    tokens: list[list[str]]
    intruder_distance: float


@dataclass
class IntruderResult:
    """
    Result of an intruder experiment.
    """

    interpretation: str = ""
    sample: IntruderSentence | None = None
    prediction: int = 0
    correct_index: int = -1
    correct: bool = False


class IntruderScorer(Classifier):
    name = "intruder"

    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        n_examples_shown: int = 1,
        temperature: float = 0.0,
        cot: bool = False,
        type: Literal["default", "internal"] = "default",
        seed: int = 42,
        **generation_kwargs,
    ):
        """
        Initialize a IntruderScorer.

        Args:
            client: The client to use for generation.
            tokenizer: The tokenizer used to cache the tokens
            verbose: Whether to print verbose output.
            n_examples_shown: The number of examples to show in the prompt,
                        a larger number can both leak information and make
                        it harder for models to generate anwers in the correct format
            temperature: The temperature to use for generation
            type: The type of intruder to use, either "word" or "sentence"
            generation_kwargs: Additional generation kwargs
        """
        super().__init__(
            client=client,
            verbose=verbose,
            n_examples_shown=n_examples_shown,
            temperature=temperature,
            seed=seed,
            **generation_kwargs,
        )
        self.type = type
        if type not in ["default", "internal"]:
            raise ValueError("Type must be either 'default' or 'internal'")
        self.cot = cot

    def prompt(self, examples: str) -> list[dict]:
        return intruder_prompt(examples, cot=self.cot)

    async def __call__(
        self,
        record: LatentRecord,
    ) -> ScorerResult:
        samples = self._prepare_and_batch(record)

        results = await self._query(
            samples,
        )

        return ScorerResult(record=record, score=results)

    def _count_words(self, examples: Sequence[Example]) -> dict[str, int]:
        """
        Count the number of words in the examples and return a dictionary of the counts.
        If activating examples are provided, count activating tokens.
        If non-activating examples are provided, count activating tokens if they exist,
        otherwise count non-activating tokens.
        """
        counts = {}
        for example in examples:
            str_tokens = example.str_tokens
            if example.normalized_activations is not None:
                acts = example.normalized_activations
                # TODO: this is a hack instead of using a threshold
                # select only acts that are larger than 2
                acts[acts < 2] = 0
            else:
                acts = example.activations
            if acts.max() > 0:
                wanted_indices = acts.nonzero()
                for index in wanted_indices:
                    if str_tokens[index] not in counts:
                        counts[str_tokens[index]] = 0
                    counts[str_tokens[index]] += 1
            else:
                for token in str_tokens:
                    if token not in counts:
                        counts[token] = 0
                    counts[token] += 1
        return counts

    def _prepare(self, record: LatentRecord) -> None:
        pass

    def _get_quantiled_examples(
        self, examples: list[ActivatingExample]
    ) -> dict[int, list[ActivatingExample]]:
        """
        Get the quantiled examples.
        """
        quantiles = {}
        for example in examples:
            if example.quantile not in quantiles:
                quantiles[example.quantile] = []
            quantiles[example.quantile].append(example)
        return quantiles

    def _prepare_and_batch(self, record: LatentRecord) -> list[IntruderSentence]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        assert len(record.not_active) > 0, "No non-activating examples found"
        batches = []
        quantiled_intruder_sentences = self._get_quantiled_examples(record.test)

        intruder_sentences = record.not_active
        for i, intruder in enumerate(intruder_sentences):
            # select each quantile equally
            quantile_index = i % len(quantiled_intruder_sentences.keys())

            active_examples = quantiled_intruder_sentences[quantile_index]
            # if there are more examples than the number of examples to show,
            # sample which examples to show
            examples_to_show = min(self.n_examples_shown - 1, len(active_examples))
            example_indices = self.rng.sample(
                range(len(active_examples)), examples_to_show
            )
            active_examples = [active_examples[i] for i in example_indices]

            # convert the examples to strings

            # highlights the active tokens
            majority_examples = []
            active_tokens = 0
            for example in active_examples:
                text, _ = _prepare_text(
                    example, n_incorrect=0, threshold=0.3, highlighted=True
                )
                majority_examples.append(text)
                active_tokens += (example.activations > 0).sum().item()
            active_tokens = int(active_tokens / len(active_examples))
            if self.type == "default":
                # if example is contrastive, use the active tokens
                # otherwise use the non-activating tokens
                if intruder.activations.max() > 0:
                    n_incorrect = 0
                else:
                    n_incorrect = active_tokens
                intruder_sentence, _ = _prepare_text(
                    intruder,
                    n_incorrect=n_incorrect,
                    threshold=0.3,
                    highlighted=True,
                )
            elif self.type == "internal":
                # randomly select a quantile to be the intruder, make sure it's not
                # the same as the source quantile
                intruder_quantile_index = self.rng.randint(
                    0, len(quantiled_intruder_sentences.keys()) - 1
                )
                while intruder_quantile_index == quantile_index:
                    intruder_quantile_index = self.rng.randint(
                        0, len(quantiled_intruder_sentences.keys()) - 1
                    )
                posible_intruder_sentences = quantiled_intruder_sentences[
                    intruder_quantile_index
                ]
                intruder_index_selected = self.rng.randint(
                    0, len(posible_intruder_sentences) - 1
                )
                intruder = posible_intruder_sentences[intruder_index_selected]
                # here the examples are activating, so we have to convert them
                # to non-activating examples
                non_activating_intruder = NonActivatingExample(
                    tokens=intruder.tokens,
                    activations=intruder.activations,
                    str_tokens=intruder.str_tokens,
                    distance=intruder.quantile,
                )
                # we highlight the correct activating tokens though
                intruder_sentence, _ = _prepare_text(
                    non_activating_intruder,
                    n_incorrect=0,
                    threshold=0.3,
                    highlighted=True,
                )
                intruder = non_activating_intruder

            # select a random index to insert the intruder sentence
            intruder_index = self.rng.randint(0, examples_to_show)
            majority_examples.insert(intruder_index, intruder_sentence)

            activations = [example.activations.tolist() for example in active_examples]
            tokens = [example.str_tokens for example in active_examples]
            activations.insert(intruder_index, intruder.activations.tolist())
            tokens.insert(intruder_index, intruder.str_tokens)

            batches.append(
                IntruderSentence(
                    examples=majority_examples,
                    intruder_index=intruder_index,
                    chosen_quantile=quantile_index,
                    activations=activations,
                    tokens=tokens,
                    intruder_distance=intruder.distance,
                )
            )

        return batches

    async def _query(
        self,
        samples: list[IntruderSentence],
    ) -> list[IntruderResult]:
        """
        Send and gather batches of samples to the model.
        """
        sem = asyncio.Semaphore(1)

        async def _process(sample):
            async with sem:
                result = await self._generate(sample)
                return result

        tasks = [asyncio.create_task(_process(sample)) for sample in samples]
        results = await asyncio.gather(*tasks)

        return results

    def _build_prompt(
        self,
        sample: IntruderSentence,
    ) -> list[dict]:
        """
        Prepare prompt for generation.
        """

        examples = "\n".join(
            f"Example {i}: {example}" for i, example in enumerate(sample.examples)
        )

        return self.prompt(examples=examples)

    def _parse(
        self,
        string: str,
    ) -> tuple[str, int]:
        """The answer will be in the format interpretation [RESPONSE]: 1"""
        # Find the first instance of the text with [RESPONSE]:
        pattern = r"\[RESPONSE\]:"
        match = re.search(pattern, string)
        if match is None:
            raise ValueError("No response found in string")
        # get everything before the match
        interpretation = string[: match.start()]
        # get everything after the match
        after = string[match.end() :]
        # the response should be a single number
        try:
            prediction = int(after.strip())
        except ValueError:
            raise ValueError("Response is not a number")
        if prediction < 0 or prediction >= self.n_examples_shown:
            raise ValueError("Response is out of range")
        return interpretation, prediction

    async def _generate(self, sample: IntruderSentence) -> IntruderResult:
        """
        Generate predictions for a batch of samples.
        """

        prompt = self._build_prompt(sample)
        try:
            response = await self.client.generate(prompt, **self.generation_kwargs)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            response = None

        if response is None:
            # default result is a error
            return IntruderResult()
        else:

            try:
                interpretation, prediction = self._parse(response.text)
            except Exception as e:
                logger.error(f"Parsing selections failed: {e}")
                # default result is a error
                return IntruderResult()

        # check that the only prediction is the intruder
        correct = prediction == sample.intruder_index

        result = IntruderResult(
            interpretation=interpretation,
            sample=sample,
            prediction=prediction,
            correct_index=sample.intruder_index,
            correct=correct,
        )

        return result
