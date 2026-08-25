import pytest
import torch
from transformers import AutoTokenizer

from delphi.utils import decode_per_token


@pytest.fixture(scope="module")
def pythia_tokenizer():
    return AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")


def test_decode_per_token_is_per_token(pythia_tokenizer):
    tokens = torch.tensor(pythia_tokenizer("Hello world, this is a test.")["input_ids"])
    str_tokens = decode_per_token(pythia_tokenizer, tokens)

    # One string per token id, unlike batch_decode on transformers >= 5,
    # which joins a 1-D tensor into a single string.
    assert len(str_tokens) == len(tokens)
    for token_id, str_token in zip(tokens, str_tokens):
        assert str_token == pythia_tokenizer.decode(token_id)


def test_decode_per_token_roundtrips_text(pythia_tokenizer):
    text = "Hello world, this is a test."
    tokens = torch.tensor(pythia_tokenizer(text)["input_ids"])
    str_tokens = decode_per_token(pythia_tokenizer, tokens)

    # BPE space markers must come back as real spaces ("Ġworld" -> " world"),
    # so the per-token strings concatenate to the original text.
    assert "".join(str_tokens) == text
