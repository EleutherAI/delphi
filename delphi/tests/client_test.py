import logging
import os
from typing import Literal

import dotenv
import fire
import torch
from beartype.claw import beartype_package
from jaxtyping import Int
from torch import Tensor
from transformers import AutoTokenizer

beartype_package("delphi")

from delphi.clients import Client, Offline, OpenRouter  # noqa: E402
from delphi.clients.types import Message  # noqa: E402
from delphi.explainers import DefaultExplainer  # noqa: E402
from delphi.latents.latents import (  # noqa: E402
    ActivatingExample,
    Latent,
    LatentRecord,
    NonActivatingExample,
)
from delphi.latents.samplers import SamplerConfig, sampler  # noqa: E402
from delphi.logger import logger  # noqa: E402
from delphi.scorers import DetectionScorer, FuzzingScorer  # noqa: E402

logger.addHandler(logging.StreamHandler())


async def test(
    explainer_provider: Literal["offline", "openrouter"] = "offline",
    # meta-llama/llama-3.3-70b-instruct
    explainer_model: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    model_max_len: int = 5120,
    num_gpus: int = 1,
    scorer_type: Literal["fuzz", "detect"] = "fuzz",
):
    """Test different client and that the explainer
    and scorer are calling the client correctly.

    Args:
        explainer_provider (Literal["offline", "openrouter"], optional):
        Which client type to use. Defaults to "offline".
        explainer_model (str, optional): VLLM model name or OpenRouter ID.
        Defaults to "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4".
        model_max_len (int, optional): Maximum length for VLLM. Defaults to 5120.
        num_gpus (int, optional): Number of GPUs to use for VLLM (TP size).
        Defaults to 1.
        scorer_type (Literal["fuzz", "detect"], optional): Scoring type to use.
        Defaults to "fuzz".
    """

    def make_scorer(client: Client):
        if scorer_type == "fuzz":
            return FuzzingScorer(client, verbose=True)
        elif scorer_type == "detect":
            return DetectionScorer(client, verbose=True)
        # other cases impossible due to beartype

    print("Creating data")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

    texts_activating = [
        "I like dogs. Dogs are great.",
        "Dog dog dog dog dog dog",
    ]
    texts_non_activating = [
        "I like cats. Cats are great.",
        "Cat cat cat cat cat cat",
    ]
    explanation = "Sentences mentioning dogs."
    activating_examples = []
    for text in texts_activating:
        token_ids: Int[Tensor, "ctx_len"] = tokenizer(text, return_tensors="pt")[
            "input_ids"
        ][
            0
        ]  # type: ignore
        dog_tokens = tokenizer("Dog dog Dog dogs Dogs", return_tensors="pt")[
            "input_ids"
        ][
            0
        ]  # type: ignore
        activating_examples.append(
            ActivatingExample(
                tokens=token_ids,
                activations=(token_ids[:, None] == dog_tokens[None, :])
                .any(dim=1)
                .float(),
                str_tokens=tokenizer.batch_decode(token_ids, skip_special_tokens=True),
            )
        )
    non_activating_examples = []

    for text in texts_non_activating:
        token_ids: Int[Tensor, "ctx_len"] = tokenizer(text, return_tensors="pt")[
            "input_ids"
        ][0]
        non_activating_examples.append(
            NonActivatingExample(
                tokens=token_ids,
                activations=torch.rand_like(token_ids, dtype=torch.float32),
                str_tokens=tokenizer.batch_decode(token_ids, skip_special_tokens=True),
            )
        )

    record = LatentRecord(
        latent=Latent("test", 0),
        examples=activating_examples,
        not_active=non_activating_examples,
        explanation=explanation,
    )
    record = sampler(
        record,
        SamplerConfig(
            n_examples_train=len(activating_examples),
            n_examples_test=len(activating_examples),
            n_quantiles=1,
            train_type="quantiles",
            test_type="quantiles",
        ),
    )

    most_recent_generation = None

    class MockClient(Client):
        async def generate(self, prompt, **kwargs) -> str:
            nonlocal most_recent_generation
            most_recent_generation = prompt, kwargs
            raise NotImplementedError("Prompt received")

    client = MockClient("")
    gen_kwargs_dict = {
        "max_length": 100,
        "num_return_sequences": 1,
    }
    explainer = DefaultExplainer(
        client, verbose=True, generation_kwargs=gen_kwargs_dict
    )
    try:
        await explainer(record)
    except NotImplementedError:
        pass
    assert most_recent_generation is not None, "Prompt not received"
    full_prompt: list[Message] = most_recent_generation[0]
    last_element: Message = full_prompt[-1]
    assert "dog" in last_element["content"]
    print(last_element["content"])
    scorer = make_scorer(client)
    try:
        await scorer(record)
    except NotImplementedError:
        pass
    assert most_recent_generation is not None, "Prompt not received"
    full_prompt: list[Message] = most_recent_generation[0]
    last_element: Message = full_prompt[-1]
    assert "dog" in last_element["content"]

    print("Mock tests passed")

    print("Loading model")
    if explainer_provider == "offline":
        client = Offline(
            explainer_model,
            max_memory=0.9,
            # Explainer models context length - must be able to accommodate the longest
            # set of examples
            max_model_len=model_max_len,
            num_gpus=num_gpus,
            statistics=False,
        )
    elif explainer_provider == "openrouter":
        client = OpenRouter(
            explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    explainer = DefaultExplainer(client, verbose=True)
    scorer = make_scorer(client)

    print("Testing explainer")
    explainer_result = await explainer(record)
    assert explainer_result.explanation, "No explanation generated"
    # assert "dog" in explainer_result.explanation.lower(), \
    # f'Explanation does not contain "dog": {explainer_result.explanation}'
    print("Explanation:", explainer_result.explanation)

    print("Testing scorer")
    scorer_result = await scorer(record)
    accuracy = 0
    n_failing = 0
    for output in scorer_result.score:
        if output.correct is None:
            n_failing += 1
        else:
            accuracy += int(output.correct)
    assert n_failing <= 1, f"Scorer failed {n_failing} times"
    accuracy /= len(scorer_result.score)
    assert accuracy > 0.5, f"Accuracy is {accuracy}"

    print("All tests passed!")
    if explainer_provider == "offline":
        assert isinstance(client, Offline)
        await client.close()


if __name__ == "__main__":
    dotenv.load_dotenv()
    fire.Fire(test)
