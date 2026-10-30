import torch

from delphi.clients.client import Client, Response
from delphi.latents import ActivatingExample, Latent, LatentRecord, NonActivatingExample
from delphi.scorers.classifier.intruder import IntruderScorer
from delphi.scorers.embedding.example_embedding import ExampleEmbeddingScorer


class DummyClient(Client):
    def __init__(self):
        super().__init__(model="dummy")

    async def generate(self, prompt, **kwargs):
        return Response(text="[RESPONSE]: 0")


def _make_activating_example(token: str, quantile: int) -> ActivatingExample:
    tokens = torch.tensor([1, 2, 3], dtype=torch.int64)
    activations = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    return ActivatingExample(
        tokens=tokens,
        activations=activations,
        str_tokens=[token, token, token],
        quantile=quantile,
    )


def _make_non_activating_example(token: str, distance: float) -> NonActivatingExample:
    tokens = torch.tensor([1, 2, 3], dtype=torch.int64)
    activations = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    return NonActivatingExample(
        tokens=tokens,
        activations=activations,
        str_tokens=[token, token, token],
        distance=distance,
    )


def test_intruder_prepare_and_batch_returns_string_examples(monkeypatch):
    def fake_prepare_text(example, n_incorrect, threshold, highlighted):
        return f"fmt-{example.str_tokens[0]}-{n_incorrect}", example.str_tokens

    monkeypatch.setattr(
        "delphi.scorers.classifier.intruder._prepare_text", fake_prepare_text
    )

    record = LatentRecord(
        latent=Latent(module_name="layers.0", latent_index=0),
        test=[
            _make_activating_example("A", quantile=0),
            _make_activating_example("B", quantile=1),
        ],
        not_active=[
            _make_non_activating_example("N0", distance=0.1),
            _make_non_activating_example("N1", distance=0.2),
        ],
    )

    scorer = IntruderScorer(DummyClient(), n_examples_shown=3, type="default", seed=0)
    batches = scorer._prepare_and_batch(record)

    assert len(batches) == 2
    for batch in batches:
        assert all(isinstance(example, str) for example in batch.examples)
        assert len(batch.activations) == len(batch.examples)
        assert len(batch.tokens) == len(batch.examples)
        assert 0 <= batch.intruder_index < len(batch.examples)


def test_intruder_internal_takes_intruder_from_another_quantile():
    # Every quantile holds the same number of examples, so the quantile lists
    # compare equal in length and their examples would be compared elementwise
    record = LatentRecord(
        latent=Latent(module_name="layers.0", latent_index=0),
        test=[
            _make_activating_example(f"Q{quantile}", quantile=quantile)
            for quantile in range(3)
            for _ in range(2)
        ],
        not_active=[
            _make_non_activating_example(f"N{i}", distance=0.1) for i in range(6)
        ],
    )

    scorer = IntruderScorer(DummyClient(), n_examples_shown=3, type="internal", seed=0)
    batches = scorer._prepare_and_batch(record)

    assert len(batches) == 6
    for batch in batches:
        assert batch.intruder_distance != batch.chosen_quantile


def test_example_embedding_internal_batch_creation(monkeypatch):
    def fake_prepare_text(example, n_incorrect, threshold, highlighted):
        return f"fmt-{example.str_tokens[0]}-{n_incorrect}", example.str_tokens

    monkeypatch.setattr(
        "delphi.scorers.embedding.example_embedding._prepare_text", fake_prepare_text
    )

    record = LatentRecord(
        latent=Latent(module_name="layers.0", latent_index=1),
        train=[
            _make_activating_example("T0", quantile=0),
            _make_activating_example("T1", quantile=1),
        ],
        test=[
            _make_activating_example("Q0", quantile=0),
            _make_activating_example("Q1", quantile=1),
            _make_activating_example("Q2", quantile=2),
        ],
        not_active=[
            _make_non_activating_example("N0", distance=0.1),
        ],
    )

    scorer = ExampleEmbeddingScorer(
        model=object(),
        method="internal",
        number_batches=2,
        seed=0,
    )
    batches = scorer._create_batches(record, number_batches=2)

    assert len(batches) == 2
    for batch in batches:
        assert isinstance(batch.distance_negative_query, (int, float))
        assert batch.distance_negative_query in {0, 1, 2}
        assert len(batch.negative_examples) > 0
        assert isinstance(batch.positive_examples, list)
