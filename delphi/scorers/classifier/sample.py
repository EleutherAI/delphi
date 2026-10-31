import random
from collections import deque
from dataclasses import dataclass
from itertools import groupby
from typing import Callable, NamedTuple

import torch

from ...latents import ActivatingExample, NonActivatingExample

L = "<<"
R = ">>"


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

    highlighted_indices: list[int] | None = None
    """Token positions marked in the actual scorer input."""

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
    highlight_indices: list[int] | None = None,
    rng: random.Random | None = None,
    **sample_kwargs,
) -> list[Sample]:
    samples = []

    for example in examples:
        text, str_toks = _prepare_text(
            example, n_incorrect, threshold, highlighted, highlight_indices, rng
        )
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
                    highlighted_indices=(
                        list(highlight_indices)
                        if highlight_indices is not None
                        else None
                    ),
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
    highlight_indices: list[int] | None = None,
    rng: random.Random | None = None,
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

    if highlight_indices is not None:
        indices = set(highlight_indices)
        if not indices or any(i < 0 or i >= len(str_toks) for i in indices):
            raise ValueError("Highlight indices must be nonempty and within the window")
        return _highlight(str_toks, indices.__contains__), str_toks

    positive_indices = activation_highlight_indices(example, threshold)
    if n_incorrect == 0:
        indices = set(positive_indices)
    else:
        # Zero activations remain eligible even when the relative threshold is zero.
        eligible = [i for i in range(len(str_toks)) if i not in positive_indices]
        if not eligible:
            raise ValueError("No tokens available for incorrect highlighting")
        indices = set((rng or random).sample(eligible, min(n_incorrect, len(eligible))))
    return _highlight(str_toks, indices.__contains__), str_toks


def activation_highlight_indices(
    example: ActivatingExample | NonActivatingExample, threshold: float
) -> list[int]:
    """Use one threshold rule for both rendering and fuzzing mask construction."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    mask = (example.activations > 0) & (
        example.activations >= threshold * example.max_activation
    )
    return torch.nonzero(mask, as_tuple=True)[0].tolist()


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
