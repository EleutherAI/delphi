import random
from collections import deque
from dataclasses import dataclass
from itertools import groupby
from typing import Callable, NamedTuple

import torch

from delphi import logger

from ...latents import ActivatingExample, NonActivatingExample

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
    assert n_incorrect >= 0, (
        "n_incorrect must be 0 if highlighting correct example "
        "or positive if creating false positives. "
        f"Got {n_incorrect}"
    )

    str_toks = example.str_tokens
    assert str_toks is not None, "str_toks were not set"

    # Just return text if there's no highlighting
    if not highlighted:
        clean = "".join(str_toks)

        return clean, str_toks

    abs_threshold = threshold * example.max_activation

    # Highlight tokens with activations above threshold
    # if this is a correct example
    if n_incorrect == 0:

        def is_above_activation_threshold(i: int) -> bool:
            return example.activations[i] >= abs_threshold

        return _highlight(str_toks, is_above_activation_threshold), str_toks

    # Highlight n_incorrect tokens with activations
    # below threshold if this is an incorrect example
    tokens_below_threshold = torch.nonzero(
        example.activations <= abs_threshold
    ).squeeze()

    # Rare case where there are no tokens below threshold
    if tokens_below_threshold.dim() == 0:
        logger.error(
            f"Tried to prepare false-positive example with {n_incorrect} tokens "
            "incorrectly highlighted, but no tokens were below activation threshold."
        )
        return DEFAULT_MESSAGE, str_toks

    random.seed(22)

    num_tokens_to_highlight = min(n_incorrect, tokens_below_threshold.shape[0])

    # The activating token is always ctx_len - ctx_len//4
    # so we always highlight this one, and if num_tokens_to_highlight > 1
    # we highlight num_tokens_to_highlight - 1 random ones
    token_pos = len(str_toks) - len(str_toks) // 4
    if token_pos in tokens_below_threshold:
        random_indices = [token_pos]

        num_remaining_tokens_to_highlight = num_tokens_to_highlight - 1
        if num_remaining_tokens_to_highlight > 0:
            remaining_tokens_below_threshold = tokens_below_threshold.tolist()
            remaining_tokens_below_threshold.remove(token_pos)

            random_indices.extend(
                random.sample(
                    remaining_tokens_below_threshold,
                    num_remaining_tokens_to_highlight,
                )
            )
    else:
        random_indices = random.sample(
            tokens_below_threshold.tolist(), num_tokens_to_highlight
        )

    random_indices = set(random_indices)

    def is_false_positive(i):
        return i in random_indices

    return _highlight(str_toks, is_false_positive), str_toks


def _highlight(tokens: list[str], check: Callable[[int], bool]) -> str:
    result: deque[str] = deque()

    tokens_grouped_by_check_fn = groupby(
        enumerate(tokens), key=lambda item: check(item[0])
    )

    for should_highlight, token_group in tokens_grouped_by_check_fn:
        highlighted_tokens = deque(token for _token_index, token in token_group)

        if should_highlight:
            highlighted_tokens.appendleft(L)
            highlighted_tokens.append(R)

        result.extend(highlighted_tokens)

    return "".join(result)
