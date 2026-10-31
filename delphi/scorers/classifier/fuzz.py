from collections import defaultdict
from typing import Literal

from delphi import logger

from ...clients.client import Client
from ...latents import LatentRecord
from ...latents.latents import ActivatingExample, NonActivatingExample
from ..scorer import Scorer
from .classifier import Classifier
from .prompts.fuzz_prompt import prompt as fuzz_prompt
from .sample import Sample, activation_highlight_indices, examples_to_samples


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
        fuzz_type: Literal["default", "active"] = "default",
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
            temperature: Which temperature to use for the scorer model.
            fuzz_type: Which type of fuzzing to use. Default uses non-activating
                examples and transfers positive mark sets one-to-one. Active pairs
                those sets with zero-activation tokens in activating examples.
        """
        super().__init__(
            client=client,
            verbose=verbose,
            n_examples_shown=n_examples_shown,
            log_prob=log_prob,
            temperature=temperature,
            **generation_kwargs,
        )
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if fuzz_type not in ("default", "active"):
            raise ValueError(f"Unknown fuzz_type: {fuzz_type}")
        self.log_prob = log_prob
        self.threshold = threshold
        self.fuzz_type = fuzz_type

    def prompt(self, examples: str, explanation: str) -> list[dict]:
        return fuzz_prompt(examples, explanation)

    def _convert_to_non_activating(
        self, examples: list[ActivatingExample]
    ) -> list[NonActivatingExample]:
        """
        Convert a list of activating examples to a list of non-activating examples.
        """
        return [
            NonActivatingExample(
                tokens=example.tokens,
                activations=example.activations,
                str_tokens=example.str_tokens,
                distance=-1.0,
            )
            for example in examples
        ]

    def _prepare(self, record: LatentRecord) -> list[Sample]:
        """Pair masks one-to-one so mark-only rules have exactly chance accuracy.

        Default negatives are windows known not to activate the target feature.
        Their stored activations may describe a neighbouring feature and are not
        used to choose marks. Active mode pairs masks with other test windows
        only when every marked token has zero target activation.
        """
        if not record.test:
            raise ValueError("No test records found")
        negatives = (
            record.not_active
            if self.fuzz_type == "default"
            else self._convert_to_non_activating(record.test)
        )
        if not negatives:
            raise ValueError("No non-activating examples found")

        masks = [activation_highlight_indices(e, self.threshold) for e in record.test]
        candidates: dict[int, list[int]] = defaultdict(list)
        for i, positive in enumerate(record.test):
            if not masks[i]:
                continue
            for j, negative in enumerate(negatives):
                if len(positive.tokens) != len(negative.tokens):
                    continue
                if self.fuzz_type == "active" and bool(
                    (negative.activations[masks[i]] > 0).any()
                ):
                    continue
                candidates[i].append(j)
            self.rng.shuffle(candidates[i])

        # Maximum bipartite matching preserves every retained positive mask
        # exactly once on each side, including when active-mode masks conflict.
        matched: dict[int, int] = {}

        def assign(i: int, seen: set[int]) -> bool:
            for j in candidates[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in matched or assign(matched[j], seen):
                    matched[j] = i
                    return True
            return False

        order = list(candidates)
        self.rng.shuffle(order)
        for i in order:
            assign(i, set())
        if not matched:
            raise ValueError("No compatible positive/negative fuzzing pairs")
        if len(matched) < len(record.test) or len(matched) < len(negatives):
            logger.warning(
                f"Fuzzing retained {len(matched)} matched pairs from "
                f"{len(record.test)} positives and {len(negatives)} negatives"
            )

        samples = []
        for j, i in matched.items():
            for example in (record.test[i], negatives[j]):
                samples.extend(
                    examples_to_samples(
                        [example],
                        highlighted=True,
                        threshold=self.threshold,
                        highlight_indices=masks[i],
                    )
                )
        return samples
