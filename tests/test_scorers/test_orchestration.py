import asyncio
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from delphi.clients.client import Client, Response
from delphi.explainers.orchestration import (
    BestOfKOrchestrator,
    HillClimbingOrchestrator,
    score_value,
)
from delphi.latents import ActivatingExample, Latent, LatentRecord, NonActivatingExample
from delphi.scorers.classifier.sample import ClassifierOutput
from delphi.scorers.scorer import Scorer, ScorerResult


class CandidateClient(Client):
    def __init__(self, candidates):
        super().__init__("fake")
        self.candidates = iter(candidates)
        self.prompts = []

    async def generate(self, prompt, **kwargs):
        name = next(self.candidates)
        self.prompts.append(prompt)
        await asyncio.sleep(0.02 if name == "A" else 0)
        return Response(text=f"[EXPLANATION]: {name}")


class Judge(Scorer):
    def __init__(self, fail=()):
        self.calls = []
        self.fail = fail
        self.rng = random.Random(2)
        self.draws = []

    async def __call__(self, record):
        assert isinstance(record, LatentRecord)
        self.calls.append(record)
        self.draws.append(self.rng.random())
        if record.explanation in self.fail:
            raise RuntimeError("candidate failed")
        samples = [
            ClassifierOutput(
                e.str_tokens,
                e.activations.tolist(),
                0,
                isinstance(e, ActivatingExample),
                prediction=True,
                correct=record.explanation == "B",
            )
            for e in record.test + record.not_active
        ]
        return ScorerResult(record, samples)


def example(name):
    return ActivatingExample(
        tokens=torch.tensor([1]),
        activations=torch.tensor([1.0]),
        normalized_activations=torch.tensor([10.0]),
        str_tokens=[name],
    )


def record():
    return LatentRecord(
        Latent("test", 0),
        train=[example("training")],
        test=[example(f"positive{i}") for i in range(8)],
        not_active=[
            NonActivatingExample(
                torch.tensor([2]), torch.tensor([0.0]), [f"negative{i}"], -1.0
            )
            for i in range(8)
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [(), ("A",)])
async def test_candidate_identity_failure_handling_and_final_holdout(failed):
    judge = Judge(failed)
    saved = []
    search = BestOfKOrchestrator(
        CandidateClient(["A", "B"]),
        [(judge, Path("scores"))],
        num_explanations=2,
        scorer_postprocess=lambda result, **kwargs: saved.append(result),
    )
    initial_rng = judge.rng.getstate()
    original = record()
    result = await search(original)
    assert result.explanation == "B"
    assert [r.explanation for r in judge.calls] == ["A", "B", "B"]
    validation_ids = {id(e) for e in judge.calls[0].test + judge.calls[0].not_active}
    final_ids = {id(e) for e in result.record.test + result.record.not_active}
    assert not validation_ids & final_ids
    assert len(validation_ids | final_ids) == 16
    assert original.explanation == ""
    assert saved == [ScorerResult(result.record, saved[0].score)]
    assert len(set(judge.draws)) == 1 and judge.rng.getstate() == initial_rng


@pytest.mark.asyncio
async def test_all_failed_candidates_raise_instead_of_selecting_unscored_explanation():
    judge = Judge(("A", "B"))
    search = BestOfKOrchestrator(
        CandidateClient(["A", "B"]), [(judge, Path())], num_explanations=2
    )
    with pytest.raises(ValueError, match="No candidate"):
        await search(record())
    assert len(judge.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy, expected", [("best", "B"), ("last", "A")])
async def test_iteration_uses_validation_feedback_but_scores_final_only_once(
    strategy, expected
):
    client = CandidateClient(["B", "A"])
    judge = Judge()
    search = HillClimbingOrchestrator(
        client, [(judge, Path())], num_rounds=2, carryforward_strategy=strategy
    )
    result = await search(record())
    assert result.explanation == expected
    assert len(judge.calls) == 3
    final_text = {
        e.str_tokens[0] for e in result.record.test + result.record.not_active
    }
    assert all(word not in str(client.prompts) for word in final_text)


def test_split_rejects_leakage_and_insufficient_examples():
    search = BestOfKOrchestrator(CandidateClient([]), [(Judge(), Path())])
    data = record()
    data.train = [data.test[0]]
    with pytest.raises(ValueError, match="overlap"):
        search._split(data)
    data = record()
    data.test = data.test[:1]
    with pytest.raises(ValueError, match="at least two"):
        search._split(data)


def test_embedding_judge_uses_record_labels_and_simulator_uses_overall_score():
    data = record()
    scores = [SimpleNamespace(text=e.str_tokens[0], similarity=0.8) for e in data.test]
    scores += [
        SimpleNamespace(text=e.str_tokens[0], similarity=0.2) for e in data.not_active
    ]
    assert score_value(ScorerResult(data, scores)) == pytest.approx(0.6)
    assert (
        score_value(
            ScorerResult(data, [SimpleNamespace(distance=0, ev_correlation_score=0.7)])
        )
        == 0.7
    )


@pytest.mark.asyncio
async def test_one_shot_generation_parses_final_candidate_lines():
    class OneShot(CandidateClient):
        async def generate(self, prompt, **kwargs):
            return Response(
                text=(
                    "Thought about [EXPLANATION]: rejected\n"
                    "[EXPLANATION]: A\n[EXPLANATION]: B"
                )
            )

    judge = Judge()
    search = BestOfKOrchestrator(
        OneShot([]), [(judge, Path())], num_explanations=2, is_multishot=False
    )
    assert (await search(record())).explanation == "B"


@pytest.mark.parametrize("kwargs", [{"num_explanations": 0}, {"judge_scorer_index": 4}])
def test_search_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        BestOfKOrchestrator(CandidateClient([]), [(Judge(), Path())], **kwargs)
