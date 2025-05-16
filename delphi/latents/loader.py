import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from asyncer import asyncify
from jaxtyping import Float, Int
from safetensors.numpy import load_file
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi.utils import (
    load_tokenized_data,
    to_int64_tensor,
)

from ..config import ConstructorConfig, SamplerConfig
from .constructors import constructor
from .latents import ActivationData, Latent, LatentData, LatentRecord
from .samplers import sampler


@dataclass
class TensorBuffer:
    """
    Lazy loading buffer for cached splits.
    """

    path: str
    """Path to the tensor file."""

    module_path: str
    """Path of the module."""

    latents: Optional[Int[Tensor, "num_latents"]] = None
    """Tensor of latent indices."""

    _tokens: Optional[Int[Tensor, "batch seq"]] = None
    """Tensor of tokens."""

    def __iter__(self):
        """
        Iterate over the buffer, yielding BufferOutput objects.

        Yields:
            Union[BufferOutput, None]: BufferOutput if enough examples,
                None otherwise.
        """
        latents, split_locations, split_activations = self.load_data_per_latent()

        for i in range(len(latents)):
            latent_locations = split_locations[i]
            latent_activations = split_activations[i]
            yield LatentData(
                Latent(self.module_path, int(latents[i].item())),
                self.module_path,
                ActivationData(latent_locations, latent_activations),
            )

    @property
    def tokens(self) -> Int[Tensor, "batch seq"] | None:
        if self._tokens is None:
            self._tokens = self.load_tokens()
        return self._tokens

    def load_data_per_latent(self):
        locations, activations, _ = self.load()
        indices = torch.argsort(locations[:, 2], stable=True)
        activations = activations[indices]
        locations = locations[indices]
        unique_latents, counts = torch.unique_consecutive(
            locations[:, 2], return_counts=True
        )
        latents = unique_latents
        split_locations = torch.split(locations, counts.tolist())
        split_activations = torch.split(activations, counts.tolist())

        return latents, split_locations, split_activations

    def load(
        self,
    ) -> tuple[
        Int[Tensor, "locations 3"],
        Float[Tensor, "activations"],
        Int[Tensor, "batch seq"] | None,
    ]:
        """Load stored tensor buffer data.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Locations, activations, and tokens.
        """
        split_data = load_file(self.path)
        first_latent = int(self.path.split("/")[-1].split("_")[0])
        activations = torch.tensor(split_data["activations"])
        locations = to_int64_tensor(split_data["locations"])
        if "tokens" in split_data:
            tokens = torch.tensor(split_data["tokens"].astype(np.int64))
        else:
            tokens = None

        locations[:, 2] = locations[:, 2] + first_latent

        if self.latents is not None:
            wanted_locations = torch.isin(locations[:, 2], self.latents)
            locations = locations[wanted_locations]
            activations = activations[wanted_locations]

        return locations, activations, tokens

    def load_tokens(self) -> Int[Tensor, "batch seq"] | None:
        _, _, tokens = self.load()
        return tokens


class LatentDataset:
    """
    Dataset which constructs TensorBuffers for each module and latent.
    """

    def __init__(
        self,
        raw_dir: os.PathLike,
        sampler_cfg: SamplerConfig,
        constructor_cfg: ConstructorConfig,
        tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None,
        modules: Optional[list[str]] = None,
        latents: Optional[dict[str, torch.Tensor]] = None,
        neighbours_path: Optional[os.PathLike] = None,
    ):
        """
        Initialize a LatentDataset.

        Args:
            raw_dir: Directory containing raw latent data.
            latent_cfg: Configuration for latent processing.
            experiment_cfg: Configuration for example creation
            and sampling.
            tokenizer: Tokenizer used to tokenize the data.
            modules: list of module names to include.
            latents: Dictionary of latents per module.
        """
        self.constructor_cfg = constructor_cfg
        self.sampler_cfg = sampler_cfg
        self.buffers: list[TensorBuffer] = []
        self.all_data: dict[str, dict[int, ActivationData] | None] = {}
        self.tokens = None
        self.neighbours_path = neighbours_path
        if modules is None:
            self.modules = os.listdir(raw_dir)
        else:
            self.modules = modules
        if latents is None:
            self._build(raw_dir)
        else:
            self._build_selected(raw_dir, latents)
        # TODO: this assumes that all modules have the same config
        if len(self.modules) == 0:
            raise ValueError("No modules found in the cache folder")
        cache_config_path = f"{raw_dir}/{self.modules[0]}/config.json"
        for module in self.modules:
            if not os.path.exists(f"{raw_dir}/{module}"):
                raise FileNotFoundError(
                    f"Could not find {module} in {raw_dir}. "
                    "Please checked the parameters passed to the dataset."
                )
            cache_config_path = f"{raw_dir}/{module}/config.json"
            if not os.path.exists(cache_config_path):
                raise FileNotFoundError(
                    "Each directory in the cache folder must have a config.json file. "
                    f"Could not find {cache_config_path} in {raw_dir}."
                )
        with open(cache_config_path, "r") as f:
            cache_config = json.load(f)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(cache_config["model_name"])
        else:
            self.tokenizer = tokenizer
        self.cache_config = cache_config

        if self.constructor_cfg.non_activating_source == "neighbours":
            # path is always going to end with /latents
            if self.neighbours_path is None:
                neighbours_path = Path(raw_dir).parent / "neighbours"
            else:
                neighbours_path = Path(self.neighbours_path)
            self.neighbours = self.load_neighbours(
                neighbours_path, self.constructor_cfg.neighbours_type
            )
            # TODO: is it possible to do this without loading all data?
            self.all_data = self._load_all_data(raw_dir, self.modules)
        else:
            self.neighbours = None

        self.load_tokens()

    def load_tokens(self):
        """
        Load tokenized data for the dataset.
        Caches the tokenized data if not already loaded.

        Returns:
            torch.Tensor: The tokenized dataset.
        """
        if not hasattr(self, "tokens"):
            self.tokens = load_tokenized_data(
                self.cache_config["ctx_len"],
                self.tokenizer,  # type: ignore
                self.cache_config["dataset_repo"],
                self.cache_config["dataset_split"],
                self.cache_config["dataset_name"],
                column_name=self.cache_config.get(
                    "dataset_column",
                    self.cache_config.get("dataset_row", "raw_content"),
                ),
            )
        return self.tokens

    def load_neighbours(self, neighbours_path: Path, neighbours_type: str):
        neighbours = {}
        for hookpoint in self.modules:
            with open(
                neighbours_path / f"{hookpoint}-{neighbours_type}.json", "r"
            ) as f:
                neighbours[hookpoint] = json.load(f)
        return neighbours

    def _edges(self, raw_dir: os.PathLike, module: str) -> list[tuple[int, int]]:
        module_dir = Path(raw_dir) / module
        safetensor_files = [f for f in module_dir.glob("*.safetensors")]
        edges = []
        for file in safetensor_files:
            start, end = file.stem.split("_")
            edges.append((int(start), int(end)))
        edges.sort(key=lambda x: x[0])
        return edges

    def _build(self, raw_dir: os.PathLike):
        """
        Build dataset buffers which load all cached latents.

        Args:
            raw_dir (os.PathLike): Directory containing raw latent data.
            modules (Optional[list[str]]): list of module names to include.
        """

        for module in self.modules:
            edges = self._edges(raw_dir, module)
            for start, end in edges:
                path = f"{raw_dir}/{module}/{start}_{end}.safetensors"
                tensor_buffer = TensorBuffer(path, module)
                if self.tokens is None:
                    self.tokens = tensor_buffer.tokens
                self.buffers.append(tensor_buffer)
                self.all_data[module] = None
            self.all_data[module] = None

    def _build_selected(
        self,
        raw_dir: os.PathLike,
        latents: dict[str, torch.Tensor],
    ):
        """
        Build a dataset buffer which loads only selected latents.

        Args:
            raw_dir (os.PathLike): Directory containing raw latent data.
            latents (dict[str, Union[int, torch.Tensor]]): Dictionary of latents
                per module.
        """

        for module in self.modules:
            edges = self._edges(raw_dir, module)
            selected_latents = latents[module]
            if len(selected_latents) == 0:
                continue
            if len(edges) == 0:
                raise FileNotFoundError(
                    f"Could not find any safetensor files in {raw_dir}/{module}, "
                    "but latents were selected."
                )
            boundaries = [edges[0][0]] + [edge[1] + 1 for edge in edges]

            bucketized = torch.bucketize(
                selected_latents, torch.tensor(boundaries), right=True
            )
            unique_buckets = torch.unique(bucketized)

            for bucket in unique_buckets:
                mask = bucketized == bucket
                _selected_latents = selected_latents[mask]

                start, end = boundaries[bucket.item() - 1], boundaries[bucket.item()]
                # Adjust end by one as the path avoids overlap
                path = f"{raw_dir}/{module}/{start}_{end-1}.safetensors"
                tensor_buffer = TensorBuffer(
                    path,
                    module,
                    _selected_latents,
                )
                if self.tokens is None:
                    self.tokens = tensor_buffer.tokens
                self.buffers.append(tensor_buffer)
            self.all_data[module] = None

    def __len__(self):
        """Return the number of buffers in the dataset."""
        return len(self.buffers)

    def _load_all_data(self, raw_dir: os.PathLike, modules: list[str]):
        """For each module, load all locations and activations"""
        all_data = {}
        for buffer in self.buffers:
            module = buffer.module_path
            if module not in all_data:
                all_data[module] = {}
            temp_latents = buffer.latents
            # we remove the filter on latents
            buffer.latents = None
            latents, locations, activations = buffer.load_data_per_latent()
            # we restore the filter on latents
            buffer.latents = temp_latents
            for latent, location, activation in zip(latents, locations, activations):
                all_data[module][latent.item()] = ActivationData(location, activation)
        return all_data

    def __iter__(self):
        """
        Synchronous iterator that wraps the asynchronous iterator.
        Creates a new event loop to drive the async generator.
        """
        # Create a new event loop.
        loop = asyncio.new_event_loop()
        async_gen = self.__aiter__()
        try:
            while True:
                # Retrieve the next item from the asynchronous iterator
                record = loop.run_until_complete(anext(async_gen))
                yield record
        except StopAsyncIteration:
            return
        finally:
            loop.close()

    async def __aiter__(self):
        """Asynchronously iterate over the dataset."""
        tasks = set()
        for buffer in self.buffers:
            for data in buffer:
                if data is None:
                    continue
                task = asyncio.create_task(self._aprocess_latent(data))
                tasks.add(task)

                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    yield task.result()
                tasks = pending
        if tasks:
            done, _ = await asyncio.wait(tasks)
            for task in done:
                yield task.result()

    async def _aprocess_buffer(self, buffer: TensorBuffer):
        """
        Asynchronously process a buffer.

        Args:
            buffer (TensorBuffer): Buffer to process.

        Yields:
            Optional[LatentRecord]: Processed latent record or None.
        """
        tasks = set()
        for data in buffer:
            if data is None:
                continue
            task = asyncio.create_task(self._aprocess_latent(data))
            tasks.add(task)
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                yield task.result()
            tasks = pending
        done, _ = await asyncio.wait(tasks)
        for task in done:
            yield task.result()

    @asyncify
    def _aprocess_latent(self, latent_data: LatentData) -> LatentRecord | None:
        return self._process_latent(latent_data)

    def _process_latent(self, latent_data: LatentData) -> LatentRecord | None:
        """
        Asynchronously process a single latent.

        Args:
            buffer_output (BufferOutput): Latent data to process.

        Returns:
            Optional[LatentRecord]: Processed latent record or None.
        """
        # This should never happen but we need to type check
        if self.tokens is None:
            raise ValueError("Tokens are not loaded")
        record = LatentRecord(latent_data.latent)

        # number of activations in the latent
        n_active = len(latent_data.activation_data.activations)
        # number of tokens in the latent
        n_tokens = self.tokens.shape[1] * self.tokens.shape[0]
        # frequency of the latent
        record.per_token_frequency = n_active / n_tokens

        if self.neighbours is not None:
            record.set_neighbours(
                self.neighbours[latent_data.module][
                    str(latent_data.latent.latent_index)
                ],
            )
        record = constructor(
            record=record,
            activation_data=latent_data.activation_data,
            constructor_cfg=self.constructor_cfg,
            tokens=self.tokens,
            tokenizer=self.tokenizer,
            all_data=self.all_data[latent_data.module],
        )
        if record is None:
            return None
        record = sampler(record, self.sampler_cfg, self.tokenizer)
        return record
