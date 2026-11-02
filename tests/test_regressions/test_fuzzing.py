import random
from collections import Counter

import pytest
import torch

from delphi.clients.client import Client, Response
from delphi.latents import ActivatingExample, Latent, LatentRecord, NonActivatingExample
from delphi.scorers import FuzzingScorer
from delphi.scorers.classifier.sample import _prepare_text


class ConstantClient(Client):
    def __init__(self, text="[1]"):
        super().__init__(model="test")
        self.text = text

    async def generate(self, prompt, **kwargs):
        return Response(text=self.text)


def positive(acts):
    return ActivatingExample(
        tokens=torch.arange(len(acts)),
        activations=torch.tensor(acts).float(),
        str_tokens=[str(i) for i in range(len(acts))],
    )


def negative(length, acts=None):
    return NonActivatingExample(
        tokens=torch.arange(length),
        activations=(
            torch.tensor(acts).float() if acts is not None else torch.zeros(length)
        ),
        str_tokens=[str(i) for i in range(length)],
    )


def record(positives, negatives=()):
    return LatentRecord(
        latent=Latent("test", 0),
        test=positives,
        not_active=list(negatives),
        explanation="test",
    )


def assert_matched(samples):
    signatures = [Counter(), Counter()]
    for sample in samples:
        indices = sample.data.highlighted_indices
        assert indices
        signatures[sample.data.activating][
            (len(sample.data.str_tokens), tuple(indices))
        ] += 1
    assert signatures[0] == signatures[1]


@pytest.mark.parametrize("seed", range(5))
def test_default_matches_entire_mark_distribution_with_unequal_counts(
    seed, monkeypatch
):
    # Length, mark count, position and contiguity all used to leak the label.
    positives = [
        positive([0, 1, 0, 0]),
        positive([1, 1, 0, 0]),
        positive([0, 0, 1, 1]),
        positive([0, 1]),
    ]
    negatives = [negative(4) for _ in range(7)] + [negative(2)]
    scorer = FuzzingScorer(ConstantClient(), seed=seed)

    def reseed(*args, **kwargs):
        pytest.fail("Highlighting must not reset the global random seed")

    monkeypatch.setattr(random, "seed", reseed)
    samples = scorer._prepare(record(positives, negatives))
    assert len(samples) == 8
    assert_matched(samples)
    for sample in samples:
        assert "<<" in sample.text


def test_threshold_is_used_and_neighbour_marks_do_not_control_negative_masks():
    scorer = FuzzingScorer(ConstantClient(), threshold=0.8)
    samples = scorer._prepare(
        record([positive([1, 0.5, 0, 0])], [negative(4, [0, 0, 1, 1])])
    )
    assert_matched(samples)
    assert all(s.data.highlighted_indices == [0] for s in samples)


def test_active_pairs_only_zero_activation_tokens_and_preserves_masks():
    scorer = FuzzingScorer(ConstantClient(), fuzz_type="active")
    samples = scorer._prepare(record([positive([1, 0, 0]), positive([0, 1, 0])]))
    assert len(samples) == 4
    assert_matched(samples)
    for sample in samples:
        if not sample.data.activating:
            assert all(
                sample.data.activations[i] == 0 for i in sample.data.highlighted_indices
            )


def test_active_maximum_matching_can_reassign_an_earlier_pair():
    # A flexible mask must leave the only compatible window for a constrained one.
    for seed in range(10):
        samples = FuzzingScorer(
            ConstantClient(), fuzz_type="active", seed=seed
        )._prepare(
            record([positive([1, 0, 0]), positive([0, 1, 0]), positive([0, 1, 1])])
        )
        assert len(samples) == 4
        assert_matched(samples)


@pytest.mark.parametrize("fuzz_type", ["default", "active"])
def test_no_compatible_pairs_fails_explicitly(fuzz_type):
    with pytest.raises(ValueError, match="compatible"):
        FuzzingScorer(ConstantClient(), fuzz_type=fuzz_type)._prepare(
            record([positive([1, 1])], [negative(3)])
        )


def test_zero_activations_are_never_marked_as_correct():
    assert _prepare_text(negative(3), 0, 0.3, True)[0] == "012"


def test_exactly_one_incorrect_candidate_is_used():
    text, _ = _prepare_text(positive([1, 0, 1]), 1, 0.3, True)
    assert text == "0<<1>>2"


def test_no_incorrect_candidate_raises_instead_of_injecting_unrelated_text():
    with pytest.raises(ValueError, match="No tokens"):
        _prepare_text(positive([1, 1]), 1, 0.3, True)


def test_threshold_boundary_cannot_be_both_correct_and_incorrect():
    text, _ = _prepare_text(positive([1, 0.5, 0]), 1, 0.5, True)
    assert text == "01<<2>>"


def test_incorrect_highlighting_has_no_forced_slot_or_global_seed_reset(monkeypatch):
    rng = random.Random(4)

    def reseed(*args, **kwargs):
        pytest.fail("Highlighting must not reset the global random seed")

    monkeypatch.setattr(random, "seed", reseed)
    texts = {_prepare_text(negative(4), 1, 0.3, True, rng=rng)[0] for _ in range(40)}
    assert len(texts) == 4
