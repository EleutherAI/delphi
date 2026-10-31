import orjson
import torch
from transformers import PreTrainedTokenizer

from delphi.latents import ActivatingExample, Latent, LatentRecord


def positive(acts):
    return ActivatingExample(
        tokens=torch.arange(len(acts)),
        activations=torch.tensor(acts).float(),
        str_tokens=[str(i) for i in range(len(acts))],
    )


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


def test_save_does_not_delete_live_record_fields(tmp_path):
    item = record([positive([1, 0])])
    before = item.__dict__.copy()
    item.save(str(tmp_path))
    assert item.__dict__ == before
    item.save(str(tmp_path))
    assert (
        orjson.loads((tmp_path / "test_latent0.json").read_bytes())["explanation"]
        == "test"
    )


def test_display_accepts_test_examples():
    tokenizer = Tokenizer()
    html = record([positive([0, 1])]).display(
        tokenizer, example_source="test", do_display=False
    )
    assert "zero" in html and "one" in html
