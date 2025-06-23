import asyncio
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from delphi import logger

from .client import Client, Response


@dataclass
class Top_Logprob:
    token: str
    logprob: float


@dataclass
class Logprobs:
    token: str
    top_logprobs: list[Top_Logprob]


@dataclass
class Statistics:
    num_prompt_tokens: int
    num_new_tokens: int
    num_generated_tokens: int


class Offline(Client):
    provider = "offline"

    def __init__(
        self,
        model: str,
        max_memory: float = 0.85,
        prefix_caching: bool = True,
        batch_size: int = 100,
        max_model_len: int = 4096,
        number_tokens_to_generate: int = 500,
        num_gpus: int = 2,
        enforce_eager: bool = False,
        statistics: bool = False,
    ):
        """Client for offline generation. Models not already present in the on-disk
        HuggingFace cache will be downloaded. Note that temperature must be increased
        for best-of-n sampling.
        """
        super().__init__(model)
        self.model = model
        self.queue = asyncio.Queue()
        self.task = None
        self.client = LLM(
            model=model,
            gpu_memory_utilization=max_memory,
            enable_prefix_caching=prefix_caching,
            tensor_parallel_size=num_gpus,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )
        self.sampling_params = SamplingParams(max_tokens=number_tokens_to_generate)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.batch_size = batch_size
        self.statistics = statistics

        if self.statistics:
            self.statistics_path = Path("statistics")
            self.statistics_path.mkdir(parents=True, exist_ok=True)

    async def process_func(
        self,
        batches: Union[str, list[Union[dict[str, str], list[dict[str, str]]]]],
        kwargs,
    ):
        """
        Process a single request.
        """

        # This is actually stupid
        for kwarg in kwargs:
            if "logprobs" in kwarg:
                self.sampling_params.logprobs = kwarg["top_logprobs"]
            if "prompt_logprobs" in kwarg:
                self.sampling_params.prompt_logprobs = kwarg["prompt_logprobs"]
            if "max_tokens" in kwarg:
                self.sampling_params.max_tokens = kwarg["max_tokens"]
            if "temperature" in kwarg:
                self.sampling_params.temperature = kwarg["temperature"]
        loop = asyncio.get_running_loop()
        prompts = []
        statistics = []

        for batch in batches:
            prompt = self.tokenizer.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True
            )
            prompts.append(prompt)
            if self.statistics:
                non_cached_tokens = len(
                    self.tokenizer.apply_chat_template(
                        batch[-1:], add_generation_prompt=True, tokenize=True  # type: ignore
                    )
                )
                statistics.append(
                    Statistics(
                        num_prompt_tokens=len(prompt),
                        num_new_tokens=non_cached_tokens,
                        num_generated_tokens=0,
                    )
                )
        response = await loop.run_in_executor(
            None,
            partial(
                self.client.generate,  # type: ignore
                prompt_token_ids=prompts,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            ),
        )

        new_response = []
        for i, r in enumerate(response):
            logprobs, prompt_logprobs = self._parse_logprobs(r)
            if self.statistics:
                statistics[i].num_generated_tokens = len(r.outputs[0].token_ids)
                # save the statistics to a file, name is a hash of the prompt
                statistics[i].prompt = batches[i][-1]["content"]  # type: ignore
                statistics[i].response = r.outputs[0].text
                with open(
                    f"statistics/{hash(batches[i][-1]['content'][-100:])}.json", "w"  # type: ignore
                ) as f:
                    json.dump(statistics[i].__dict__, f, indent=4)
            new_response.append(
                Response(
                    text=r.outputs[0].text,
                    logprobs=logprobs,
                    prompt_logprobs=prompt_logprobs,
                )
            )
        return new_response

    async def generate(
        self, prompt: Union[str, list[dict[str, str]]], **kwargs
    ) -> Response:  # type: ignore
        """
        Enqueue a request and wait for the result.
        """
        future = asyncio.Future()
        if self.task is None:
            self.task = asyncio.create_task(self._process_batches())
        await self.queue.put((prompt, future, kwargs))
        return await future

    def _parse_logprobs(self, response):
        response_tokens = response.outputs[0].token_ids
        logprobs = response.outputs[0].logprobs
        prompt_logprobs = response.prompt_logprobs
        if logprobs is None and prompt_logprobs is None:
            return None, None
        logprobs_list = None
        if logprobs is not None:
            logprobs_list = []
            for i in range(len(logprobs)):
                log_prob_dict = logprobs[i]
                top_logprobs = []
                decoded_token = ""
                for token, logprob in log_prob_dict.items():
                    if token == response_tokens[i]:
                        decoded_token = logprob.decoded_token
                        top_logprobs.append(
                            Top_Logprob(token=decoded_token, logprob=logprob.logprob)
                        )
                    else:
                        top_logprobs.append(
                            Top_Logprob(
                                token=logprob.decoded_token, logprob=logprob.logprob
                            )
                        )
                logprobs_list.append(
                    Logprobs(token=decoded_token, top_logprobs=top_logprobs)
                )

        return logprobs_list, prompt_logprobs

    async def _process_batches(self):
        """
        Continuously process batches of requests.
        """
        batch_count = 0
        while True:
            batch = []
            batch_futures = []
            batch_kwargs = []
            # Collect a batch of requests
            start_time = asyncio.get_event_loop().time()
            while len(batch) < self.batch_size:
                try:
                    prompt, future, kwargs = self.queue.get_nowait()
                    batch.append(prompt)
                    batch_futures.append(future)
                    batch_kwargs.append(kwargs)
                except asyncio.QueueEmpty:
                    if batch:  # If we have any items, process them
                        break
                    await asyncio.sleep(0.1)  # Short sleep if queue is empty
                    continue

                if (
                    asyncio.get_event_loop().time() - start_time > 1
                ):  # Time-based batch cutoff
                    break

            if not batch:
                continue
            # Process the batch
            try:
                results = await self.process_func(batch, batch_kwargs)
                batch_count += 1

                for result, future in zip(results, batch_futures):
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {repr(e)}")
                for future in batch_futures:
                    if not future.done():
                        future.set_exception(e)

    async def close(self):
        """
        Clean up resources when the client is no longer needed.
        """
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.client
        self.client = None
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
