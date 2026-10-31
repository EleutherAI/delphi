import pytest
import torch
from transformers import PreTrainedTokenizer

from delphi.config import ConstructorConfig
from delphi.latents import Latent, LatentRecord
from delphi.latents.constructors import constructor, random_non_activating_windows
from delphi.latents.latents import ActivationData


def record(positives, negatives=()):
    return LatentRecord(
        latent=Latent("test", 0),
        test=positives,
        not_active=list(negatives),
        explanation="test",
    )


class Tokenizer(PreTrainedTokenizer):
    def __init__(self):
        pass

    @property
    def vocab(self):
        return {"zero": 0, "one": 1}

    def batch_decode(self, tokens):
        return [str(int(token)) for token in tokens]


def test_one_available_negative_window_retains_batch_dimension():
    examples = random_non_activating_windows(
        torch.tensor([0]), torch.tensor([[1, 2, 3]]), 1, Tokenizer()
    )
    assert len(examples) == 1
    assert examples[0].tokens.shape == (3,)


def test_random_negatives_do_not_duplicate_windows():
    examples = random_non_activating_windows(
        torch.arange(20), torch.arange(40).reshape(20, 2), 20, Tokenizer()
    )
    assert len({tuple(e.tokens.tolist()) for e in examples}) == 20


@pytest.mark.parametrize("center", [False, True])
def test_center_flag_selects_documented_constructor_branch(monkeypatch, center):
    import importlib

    module = importlib.import_module("delphi.latents.constructors")
    calls = []

    def pool(name):
        def run(**kwargs):
            calls.append(name)
            return torch.tensor([[1, 2, 3, 4]]), torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        return run

    monkeypatch.setattr(module, "pool_max_activation_windows", pool("windows"))
    monkeypatch.setattr(module, "pool_centered_activation_windows", pool("centered"))
    constructor(
        record([]),
        ActivationData(torch.tensor([[0, 1, 0]]), torch.tensor([1.0])),
        ConstructorConfig(
            example_ctx_len=4,
            min_examples=0,
            n_non_activating=0,
            center_examples=center,
        ),
        torch.arange(8).reshape(1, 8),
        Tokenizer(),
    )
    assert calls == ["centered" if center else "windows"]


def test_constructor_with_exactly_one_nonactivating_window():
    item = constructor(
        record([]),
        ActivationData(torch.tensor([[0, 1, 0]]), torch.tensor([1.0])),
        ConstructorConfig(example_ctx_len=4, min_examples=0, n_non_activating=1),
        torch.arange(8).reshape(1, 8),
        Tokenizer(),
    )
    assert len(item.not_active) == 1
    assert item.not_active[0].tokens.tolist() == [4, 5, 6, 7]


def test_negative_sampling_does_not_reset_global_torch_rng():
    state = torch.random.get_rng_state().clone()
    random_non_activating_windows(
        torch.arange(3), torch.arange(6).reshape(3, 2), 2, Tokenizer()
    )
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("center, expected_peak", [(False, 2), (True, 6)])
def test_centering_changes_the_actual_peak_offset(center, expected_peak):
    item = constructor(
        record([]),
        ActivationData(torch.tensor([[0, 10, 0]]), torch.tensor([1.0])),
        ConstructorConfig(
            example_ctx_len=8,
            min_examples=1,
            n_non_activating=0,
            center_examples=center,
        ),
        torch.arange(32).reshape(1, 32),
        Tokenizer(),
    )
    assert item.examples[0].activations.argmax().item() == expected_peak
    assert item.examples[0].tokens[expected_peak].item() == 10


def test_centered_constructor_without_interior_windows_returns_none():
    item = constructor(
        record([]),
        ActivationData(torch.tensor([[0, 1, 0]]), torch.tensor([1.0])),
        ConstructorConfig(
            example_ctx_len=4, min_examples=1, n_non_activating=0, center_examples=True
        ),
        torch.arange(4).reshape(1, 4),
        Tokenizer(),
    )
    assert item is None
