import random
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ...latents import ActivatingExample, NonActivatingExample
from ...logger import logger

L = "<<"
R = ">>"
DEFAULT_MESSAGE = (
    "<<NNsight>> is the best library for <<interpretability>> on huge models!"
)


@dataclass
class ClassifierOutput:
    str_tokens: list[str]
    """list of strings"""

    activations: list[float]
    """list of floats"""

    distance: float | int
    """Quantile or neighbor distance"""

    activating: bool
    """Whether the example is activating or not"""

    prediction: bool | None = False
    """Whether the model predicted the example activating or not"""

    probability: float | None = 0.0
    """The probability of the example activating"""

    correct: bool | None = False
    """Whether the prediction is correct"""


class Sample(NamedTuple):
    text: str
    data: ClassifierOutput


def examples_to_samples(
    examples: list[ActivatingExample] | list[NonActivatingExample],
    n_incorrect: int = 0,
    threshold: float = 0.3,
    highlighted: bool = False,
    **sample_kwargs,
) -> list[Sample]:
    samples = []

    for example in examples:
        text, str_toks = _prepare_text(example, n_incorrect, threshold, highlighted)
        match example:
            case ActivatingExample():
                activating = True
                distance = example.quantile
            case NonActivatingExample():
                activating = False
                distance = example.distance

        samples.append(
            Sample(
                text=text,
                data=ClassifierOutput(
                    str_tokens=str_toks,
                    activations=example.activations.tolist(),
                    activating=activating,
                    distance=distance,
                    **sample_kwargs,
                ),
            )
        )

    return samples


# NOTE: Should reorganize below, it's a little confusing


def _prepare_text(
    example: ActivatingExample | NonActivatingExample,
    n_incorrect: int,
    threshold: float,
    highlighted: bool,
) -> tuple[str, list[str]]:
    str_toks = example.str_tokens
    clean = "".join(str_toks)
    # Just return text if there's no highlighting
    if not highlighted:
        return clean, str_toks

    threshold = threshold * example.max_activation

    # Highlight tokens with activations above threshold
    # if correct example
    if n_incorrect == 0:

        def threshold_check(i):
            return example.activations[i] >= threshold

        return _highlight(str_toks, threshold_check), str_toks

    # Highlight n_incorrect tokens with activations
    # below threshold if incorrect example
    below_threshold = torch.nonzero(example.activations <= threshold).squeeze()

    # Rare case where there are no tokens below threshold
    if below_threshold.dim() == 0:
        logger.error("Failed to prepare example.")
        return DEFAULT_MESSAGE, str_toks

    random.seed(22)

    n_incorrect = min(n_incorrect, len(below_threshold))

    random_indices = set(random.sample(below_threshold.tolist(), n_incorrect))

    def check(i):
        return i in random_indices

    return _highlight(str_toks, check), str_toks


def _highlight(tokens, check):
    result = []

    i = 0
    while i < len(tokens):
        if check(i):
            result.append(L)

            while i < len(tokens) and check(i):
                result.append(tokens[i])
                i += 1

            result.append(R)
        else:
            result.append(tokens[i])
            i += 1

    return "".join(result)
