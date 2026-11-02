import torch

from delphi.latents import ActivatingExample, NonActivatingExample
from delphi.scorers.classifier.sample import _prepare_text


def _activating_example() -> ActivatingExample:
    return ActivatingExample(
        tokens=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64),
        activations=torch.tensor([0.0, 0.1, 0.2, 0.0, 0.4, 0.0, 0.9, 0.0]),
        str_tokens=["a", "b", "c", "d", "e", "f", "g", "h"],
        quantile=1,
    )


def _non_activating_example() -> NonActivatingExample:
    return NonActivatingExample(
        tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        activations=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        str_tokens=["x", "y", "z", "w"],
        distance=0.0,
    )


def test_prepare_text_highlighted_correct_example_returns_markers():
    text, str_toks = _prepare_text(
        _activating_example(), n_incorrect=0, threshold=0.3, highlighted=True
    )

    assert str_toks == ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert "<<" in text and ">>" in text


def test_prepare_text_false_positive_does_not_force_a_token_position():
    import random

    rng = random.Random(4)
    texts = {
        _prepare_text(
            _non_activating_example(),
            n_incorrect=1,
            threshold=0.3,
            highlighted=True,
            rng=rng,
        )[0]
        for _ in range(40)
    }
    assert len(texts) == 4
