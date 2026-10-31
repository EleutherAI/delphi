from math import log
from types import SimpleNamespace

import pytest
import torch

from delphi.clients.client import Client, Response
from delphi.latents import ActivatingExample, Latent, LatentRecord, NonActivatingExample
from delphi.scorers import DetectionScorer


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
        activations=torch.tensor(acts).float()
        if acts is not None
        else torch.zeros(length),
        str_tokens=[str(i) for i in range(length)],
    )


def record(positives, negatives=()):
    return LatentRecord(
        latent=Latent("test", 0),
        test=positives,
        not_active=list(negatives),
        explanation="test",
    )


@pytest.mark.parametrize("answer", ["[2]", "[-1]", '["0"]', "[null]", "[0.0]"])
def test_invalid_predictions_are_rejected(answer):
    with pytest.raises(ValueError, match="binary"):
        DetectionScorer(ConstantClient())._parse(answer)


@pytest.mark.asyncio
async def test_final_partial_batch_is_scored():
    scorer = DetectionScorer(ConstantClient("[1, 0]"), n_examples_shown=3)
    result = await scorer(record([positive([1, 0])], [negative(2)]))
    assert len(result.score) == 2
    assert [s.prediction for s in result.score] == [True, False]


def test_logprob_candidates_are_exact_binary_tokens():
    scorer = DetectionScorer(ConstantClient(), log_prob=True)
    token = SimpleNamespace(
        token="1",
        top_logprobs=[
            SimpleNamespace(token="0", logprob=log(0.1)),
            SimpleNamespace(token="1", logprob=log(0.3)),
            SimpleNamespace(token="10", logprob=log(0.6)),
        ],
    )
    assert scorer._parse("[1]", [token])[1] == pytest.approx([0.75])
