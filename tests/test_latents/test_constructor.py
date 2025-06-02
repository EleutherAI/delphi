import random
from itertools import chain
from typing import Any, Literal

import pytest
import torch
from jaxtyping import Int
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import (
    ActivatingExample,
    Latent,
    LatentDataset,
    LatentRecord,
    constructor,
    sampler,
)
from delphi.latents.latents import ActivationData


def test_save_load_cache(
    cache_setup: dict[str, Any],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    sampler_cfg = SamplerConfig(
        n_examples_train=3,
        n_examples_test=3,
        n_quantiles=3,
        train_type="quantiles",
        test_type="quantiles",
    )
    dataset = LatentDataset(
        cache_setup["temp_dir"],
        sampler_cfg,
        ConstructorConfig(min_examples=0),
        tokenizer,
    )
    tokens: Int[Tensor, "examples ctx_len"] = dataset.load_tokens()  # type: ignore
    assert (tokens == cache_setup["tokens"][: len(tokens)]).all()
    for record in dataset:
        assert len(record.train) <= sampler_cfg.n_examples_train
        assert len(record.test) <= sampler_cfg.n_examples_test


@pytest.fixture(scope="module")
def seed():
    random.seed(0)
    torch.manual_seed(0)


@pytest.mark.parametrize("n_samples", [5, 10, 100, 1000])
@pytest.mark.parametrize("n_quantiles", [2, 5, 10, 23])
@pytest.mark.parametrize("n_examples", [0, 2, 5, 10, 20])
@pytest.mark.parametrize("train_type", ["top", "random", "quantiles"])
def test_simple_cache(
    n_samples: int,
    n_quantiles: int,
    n_examples: int,
    train_type: Literal["top", "random", "quantiles"],
    ctx_len: int = 32,
    seed: None = None,
    *,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    torch.manual_seed(0)
    tokens = torch.randint(
        0,
        100,
        (
            n_samples,
            ctx_len,
        ),
    )
    all_activation_data = []
    all_activations = []
    for feature_idx in range(2):
        activations = torch.rand(n_samples, ctx_len, 1) * (
            torch.rand(n_samples)[..., None, None] ** 2
        )
        all_activations.append(activations)
        mask = activations > 0.1
        locations = torch.nonzero(mask)
        locations[..., 2] = feature_idx
        all_activation_data.append(ActivationData(locations, activations[mask]))
    activation_data, other_activation_data = all_activation_data
    activations, other_activations = all_activations
    record = LatentRecord(latent=Latent("test", 0), examples=[])
    constructor(
        record,
        activation_data,
        constructor_cfg=ConstructorConfig(
            example_ctx_len=ctx_len,
            min_examples=1,
            n_non_activating=50,
            non_activating_source="neighbours",
        ),
        tokens=tokens,
        tokenizer=tokenizer,
        all_data={0: activation_data, 1: other_activation_data},
    )
    for i, j in zip(record.examples[:-1], record.examples[1:]):
        assert i.max_activation >= j.max_activation
    for i in record.examples:
        index = (tokens == i.tokens).all(dim=-1).float().argmax()
        assert (tokens[index] == i.tokens).all()
        assert activations[index].max() == i.max_activation
    sampler(
        record,
        SamplerConfig(
            n_examples_train=n_examples,
            n_examples_test=n_examples,
            n_quantiles=n_quantiles,
            train_type=train_type,
            test_type="quantiles",
        ),
        tokenizer=tokenizer,
    )
    assert len(record.train) <= n_examples
    assert len(record.test) <= n_examples
    for neighbor in record.neighbours:
        assert neighbor.latent_index == 1
    for example in chain(record.train, record.test):
        assert isinstance(example, ActivatingExample)
        assert example.normalized_activations is not None
        assert example.normalized_activations.shape == example.activations.shape
        assert (example.normalized_activations <= 10).all()
        assert (example.normalized_activations >= 0).all()
    for quantile_list in (record.test,) + (  # type: ignore
        (record.train,) if train_type == "quantiles" else ()
    ):
        quantile_list: list[ActivatingExample] = quantile_list
        for k, i in enumerate(quantile_list):
            for j in quantile_list[k + 1 :]:
                if i.quantile != j.quantile:
                    assert i.max_activation >= j.max_activation
                    assert i.quantile < j.quantile
