from math import ceil
from typing import Literal

import torch

from ...clients.client import Client
from ...latents import LatentRecord
from ...latents.latents import ActivatingExample, NonActivatingExample
from ..scorer import Scorer
from .classifier import Classifier
from .prompts.fuzz_prompt import prompt as fuzz_prompt
from .sample import Sample, examples_to_samples


class FuzzingScorer(Classifier, Scorer):
    name = "fuzz"

    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        n_examples_shown: int = 1,
        threshold: float = 0.3,
        log_prob: bool = False,
        temperature: float = 0.0,
        fuzz_type: Literal["default", "hard", "contrastive"] = "default",
        **generation_kwargs,
    ):
        """
        Initialize a FuzzingScorer.

        Args:
            client: The client to use for generation.
            tokenizer: The tokenizer used to cache the tokens
            verbose: Whether to print verbose output.
            n_examples_shown: The number of examples to show in the prompt,
                        a larger number can both leak information and make
                        it harder for models to generate anwers in the correct format.
            log_prob: Whether to use log probabilities to allow for AUC calculation.
            generation_kwargs: Additional generation kwargs.
        """
        super().__init__(
            client=client,
            verbose=verbose,
            n_examples_shown=n_examples_shown,
            log_prob=log_prob,
            temperature=temperature,
            **generation_kwargs,
        )

        self.threshold = threshold
        self.fuzz_type = fuzz_type
    def prompt(self, examples: str, explanation: str) -> list[dict]:
        return fuzz_prompt(examples, explanation)

    def mean_n_activations_ceil(self, examples: list[ActivatingExample]):
        """
        Calculate the ceiling of the average number of activations in each example.
        """
        avg = sum(
            len(torch.nonzero(example.activations)) for example in examples
        ) / len(examples)

        return ceil(avg)

    def _prepare(self, record: LatentRecord) -> list[Sample]:  # type: ignore
        """
        Prepare and shuffle a list of samples for classification.
        """
        assert len(record.test) > 0, "No test records found"

        n_incorrect = self.mean_n_activations_ceil(record.test) 

        if self.fuzz_type == "default":
            assert len(record.not_active) > 0, "No non-activating examples found"
            # default uses non-activating examples and
            # randomly highlights n_incorrect tokens
            samples = examples_to_samples(
                record.not_active,
                n_incorrect=n_incorrect,
                highlighted=True,
            )
        elif self.fuzz_type == "hard":
            # hard uses activating examples and
            # highlights non active tokens
            extras = []
            for example in record.test:
                # convert from activating to non-activating
                new_example = NonActivatingExample(
                    tokens=example.tokens,
                    activations=example.activations,
                    str_tokens=example.str_tokens,
                    normalized_activations=example.normalized_activations,
                    distance=-1)
                extras.append(new_example)
            samples = examples_to_samples(
                extras,
                n_incorrect=n_incorrect,
                highlighted=True,
            )
        elif self.fuzz_type == "contrastive":
            # contrastive uses non-activating examples and
            # highlights active tokens of neighbours
            assert len(record.not_active) > 0, "No non-activating examples found"
            
            samples = examples_to_samples(
                record.not_active,
                n_incorrect=0,
                highlighted=True,
            )

        samples.extend(
            examples_to_samples(
                record.test,  # type: ignore
                n_incorrect=0,
                highlighted=True,
            )
        )
        return samples
