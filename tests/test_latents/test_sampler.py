from typing import Literal

import pytest
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi.config import SamplerConfig
from delphi.latents import ActivatingExample, Latent, LatentRecord, sampler


@pytest.mark.parametrize("train_type", ["top", "random", "quantiles", "mix"])
def test_test_examples_are_not_shown_to_the_explainer(
    train_type: Literal["top", "random", "quantiles", "mix"],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    # Examples sorted by decreasing activation, as the constructor produces them
    examples = [
        ActivatingExample(
            tokens=torch.tensor([i % 50, 1]),
            activations=torch.tensor([float(1000 - i), 0.0]),
        )
        for i in range(1000)
    ]
    record = LatentRecord(latent=Latent("test", 0), examples=examples)

    sampler(record, SamplerConfig(train_type=train_type), tokenizer=tokenizer)

    train_ids = {id(example) for example in record.train}
    assert len(record.test) == SamplerConfig().n_examples_test
    assert not any(id(example) in train_ids for example in record.test)
    # Test examples still come from their activation quantile
    position = {id(example): i for i, example in enumerate(examples)}
    for example in record.test:
        assert position[id(example)] // 100 == example.quantile
