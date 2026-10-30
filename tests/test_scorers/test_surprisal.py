import torch
from torch.nn.functional import cross_entropy
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi.scorers.surprisal.surprisal import Sample, SurprisalOutput, SurprisalScorer


def test_kv_cache_losses_match_full_forward(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=len(tokenizer), n_layer=2, n_head=2, n_embd=64)
    ).eval()
    model.tokenizer = tokenizer
    scorer = SurprisalScorer(model, verbose=False, batch_size=2)

    # 5 samples with batch_size=2: two full batches and a partial one
    texts = [
        "The cat sat on the mat",
        "Dogs bark",
        "A bird sings",
        "Rain falls",
        "It snows",
    ]
    samples = [
        Sample(text=text, activations=[], data=SurprisalOutput(text=text, distance=0))
        for text in texts
    ]
    explanation = "Description: animals\n Sentences:\n"

    losses = scorer.compute_loss_with_kv_cache(explanation, samples, batch_size=2)

    explanation_ids = tokenizer.encode(explanation, add_special_tokens=False)
    for text, text_losses in zip(texts, losses):
        ids = tokenizer.encode(text, add_special_tokens=False)
        with torch.no_grad():
            logits = model(torch.tensor([explanation_ids + ids])).logits[0]
        expected = cross_entropy(
            logits[len(explanation_ids) : -1],
            torch.tensor(ids[1:]),
            reduction="none",
        )
        torch.testing.assert_close(
            torch.tensor(text_losses[: len(ids) - 1]), expected, atol=1e-4, rtol=0
        )
