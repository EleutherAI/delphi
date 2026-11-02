from typing import Literal

import pytest
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi.config import SamplerConfig
from delphi.latents import ActivatingExample, Latent, LatentRecord, sampler
from delphi.latents.samplers import split_quantiles


def activating_examples(n: int = 1000) -> list[ActivatingExample]:
    """Examples sorted by decreasing activation, as the constructor produces them."""
    return [
        ActivatingExample(
            tokens=torch.tensor([i % 50, 1]),
            activations=torch.tensor([float(1000 - i), 0.0]),
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("train_type", ["top", "random", "quantiles", "mix"])
def test_test_examples_are_not_shown_to_the_explainer(
    train_type: Literal["top", "random", "quantiles", "mix"],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    examples = activating_examples()
    record = LatentRecord(latent=Latent("test", 0), examples=examples)

    sampler(record, SamplerConfig(train_type=train_type), tokenizer=tokenizer)

    train_ids = {id(example) for example in record.train}
    assert len(record.test) == SamplerConfig().n_examples_test
    assert not any(id(example) in train_ids for example in record.test)
    # Test examples still come from their activation quantile
    position = {id(example): i for i, example in enumerate(examples)}
    for example in record.test:
        assert position[id(example)] // 100 == example.quantile


@pytest.mark.parametrize(
    "n_samples,n_quantiles",
    [(50, 10), (25, 10), (9, 10), (5, 10), (1, 10), (50, 23), (100, 7)],
)
def test_split_quantiles_returns_as_many_as_asked_for(n_samples: int, n_quantiles: int):
    # n_samples was floored to a multiple of n_quantiles, so a request that
    # didn't divide came back short, and one below n_quantiles came back empty.
    samples = split_quantiles(activating_examples(), n_quantiles, n_samples)

    assert len(samples) == n_samples


def test_split_quantiles_samples_stay_in_their_own_quantile():
    # The remainder changes how many come from a quantile, not which quantile
    # an example belongs to.
    examples = activating_examples()
    position = {id(example): i for i, example in enumerate(examples)}

    for example in split_quantiles(examples, 10, 25):
        assert position[id(example)] // 100 == example.quantile


def test_fewer_test_examples_than_quantiles(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    # The symptom this is really about: a latent scored on no examples at all.
    record = LatentRecord(latent=Latent("test", 0), examples=activating_examples())

    sampler(
        record,
        SamplerConfig(n_examples_test=5, n_quantiles=10),
        tokenizer=tokenizer,
    )

    assert len(record.test) == 5
