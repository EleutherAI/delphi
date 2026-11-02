import asyncio
import math
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from delphi.scorers.simulator.simulation.simulator import NeuronSimulator


class _UniformLogprobClient:
    """Returns prompt logprobs with P("7") = P("0") = 0.5 at every position."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    async def generate(self, prompt, **kwargs):
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_dict=False
        )
        logprobs = {
            0: SimpleNamespace(decoded_token="7", logprob=math.log(0.5)),
            1: SimpleNamespace(decoded_token="0", logprob=math.log(0.5)),
        }
        return SimpleNamespace(prompt_logprobs=[logprobs] * (len(prompt_ids) + 1))


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        # prepends a BOS token
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        # does not prepend a BOS token
        "Qwen/Qwen2.5-0.5B-Instruct",
    ],
)
def test_logprob_simulation_reads_every_token(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    simulator = NeuronSimulator(
        _UniformLogprobClient(tokenizer), "cats", use_logprobs=True
    )
    tokens = ["The", " cat", " sat"]

    simulation = asyncio.run(simulator.simulate(tokens))

    # A parsing failure would be swallowed by `simulate` and return all zeros
    assert simulation.expected_activations == pytest.approx([3.5, 3.5, 3.5])
