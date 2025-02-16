import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Union

import numpy as np
import torch
from safetensors.numpy import load_file
from torchtyping import TensorType
from transformers import AutoTokenizer

from delphi.utils import (
    load_tokenized_data,
)

from ..config import LatentConfig
from .latents import Latent, LatentRecord


class BufferOutput(NamedTuple):
    """
    Represents the output of a TensorBuffer.
    """

    latent: Latent
    """The latent associated with this output."""

    locations: TensorType["locations", 2]
    """Tensor of latent locations."""

    activations: TensorType["locations"]
    """Tensor of latent activations."""

    tokens: TensorType["tokens"]
    """Tensor of all tokens."""


@dataclass
class TensorBuffer:
    """
    Lazy loading buffer for cached splits.
    """

    path: str
    """Path to the tensor file."""

    module_path: str
    """Path of the module."""

    latents: Optional[TensorType["latents"]] = None
    """Tensor of latent indices."""

    min_examples: int = 120
    """Minimum number of examples required. Defaults to 120."""

    def __iter__(self):
        """
        Iterate over the buffer, yielding BufferOutput objects.

        Yields:
            Union[BufferOutput, None]: BufferOutput if enough examples,
                None otherwise.
        """
        latents, split_locations, split_activations, tokens = self.load()

        for i in range(len(latents)):
            latent_locations = split_locations[i]
            latent_activations = split_activations[i]
            if len(latent_locations) < self.min_examples:
                yield None
            else:
                yield BufferOutput(
                    Latent(self.module_path, int(latents[i].item())),
                    latent_locations,
                    latent_activations,
                    tokens,
                )

    def load(self):
        split_data = load_file(self.path)
        first_latent = int(self.path.split("/")[-1].split("_")[0])
        activations = torch.tensor(split_data["activations"])
        locations = torch.tensor(split_data["locations"].astype(np.int64))
        if "tokens" in split_data:
            tokens = torch.tensor(split_data["tokens"].astype(np.int64))
        else:
            tokens = None

        locations[:, 2] = locations[:, 2] + first_latent

        if self.latents is not None:
            wanted_locations = torch.isin(locations[:, 2], self.latents)
            locations = locations[wanted_locations]
            activations = activations[wanted_locations]

        indices = torch.argsort(locations[:, 2], stable=True)
        activations = activations[indices]
        locations = locations[indices]
        unique_latents, counts = torch.unique_consecutive(
            locations[:, 2], return_counts=True
        )
        latents = unique_latents
        split_locations = torch.split(locations, counts.tolist())
        split_activations = torch.split(activations, counts.tolist())

        return latents, split_locations, split_activations, tokens

    def reset(self):
        """Reset the buffer state."""
        self.start = 0
        self.activations = None
        self.locations = None


class LatentDataset:
    """
    Dataset which constructs TensorBuffers for each module and latent.
    """

    def __init__(
        self,
        raw_dir: str,
        cfg: LatentConfig,
        tokenizer: Optional[Callable] = None,
        modules: Optional[list[str]] = None,
        latents: Optional[dict[str, Union[int, torch.Tensor]]] = None,
        constructor: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize a LatentDataset.

        Args:
            raw_dir: Directory containing raw latent data.
            cfg: Configuration for latent processing.
            modules: list of module names to include.
            latents: Dictionary of latents per module.
        """
        self.cfg = cfg
        self.buffers = []

        if latents is None:
            self._build(raw_dir, modules)
        else:
            self._build_selected(raw_dir, modules, latents)
        # TODO: this assumes that all modules have the same config
        cache_config_dir = f"{raw_dir}/{modules[0]}/config.json"
        with open(cache_config_dir, "r") as f:
            cache_config = json.load(f)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(cache_config["model_name"])
        else:
            self.tokenizer = tokenizer
        self.cache_config = cache_config
        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

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
                self.tokenizer,
                self.cache_config["dataset_repo"],
                self.cache_config["dataset_split"],
                self.cache_config["dataset_name"],
                column_name=self.cache_config.get(
                    "dataset_column",
                    self.cache_config.get("dataset_row", "raw_content"),
                ),
            )
        return self.tokens

    def _edges(self, raw_dir: str, module: str) -> list[tuple[int, int]]:
        module_dir = Path(raw_dir) / module
        safetensor_files = [f for f in module_dir.glob("*.safetensors")]
        edges = []
        for file in safetensor_files:
            start, end = file.stem.split("_")
            edges.append((int(start), int(end)))
        edges.sort(key=lambda x: x[0])
        return edges

    def _build(self, raw_dir: str, modules: Optional[list[str]] = None):
        """
        Build dataset buffers which load all cached latents.

        Args:
            raw_dir (str): Directory containing raw latent data.
            modules (Optional[list[str]]): list of module names to include.
        """
        modules = os.listdir(raw_dir) if modules is None else modules

        for module in modules:
            edges = self._edges(raw_dir, module)
            for start, end in edges:
                path = f"{raw_dir}/{module}/{start}_{end}.safetensors"
                self.buffers.append(
                    TensorBuffer(path, module, min_examples=self.cfg.min_examples)
                )

    def _build_selected(
        self,
        raw_dir: str,
        modules: list[str],
        latents: dict[str, Union[int, torch.Tensor]],
    ):
        """
        Build a dataset buffer which loads only selected latents.

        Args:
            raw_dir (str): Directory containing raw latent data.
            modules (list[str]): list of module names to include.
            latents (dict[str, Union[int, torch.Tensor]]): Dictionary of latents
                per module.
        """

        for module in modules:
            edges = self._edges(raw_dir, module)
            selected_latents = latents[module]
            if isinstance(selected_latents, int):
                selected_latents = torch.tensor([selected_latents])
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

                self.buffers.append(
                    TensorBuffer(
                        path,
                        module,
                        _selected_latents,
                        min_examples=self.cfg.min_examples,
                    )
                )

    def __len__(self):
        """Return the number of buffers in the dataset."""
        return len(self.buffers)

    def reset(self):
        """Reset all buffers in the dataset."""
        for buffer in self.buffers:
            buffer.reset()

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
        for buffer in self.buffers:
            async for record in self._aprocess_buffer(buffer):
                yield record

    async def _aprocess_buffer(self, buffer: TensorBuffer):
        """
        Asynchronously process a buffer.

        Args:
            buffer (TensorBuffer): Buffer to process.

        Yields:
            Optional[LatentRecord]: Processed latent record or None.
        """
        for data in buffer:
            if data is not None:
                record = await self._aprocess_latent(data)
                if record is not None:
                    yield record
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def _aprocess_latent(self, buffer_output: BufferOutput) -> LatentRecord:
        """
        Asynchronously process a single latent.

        Args:
            buffer_output (BufferOutput): Latent data to process.

        Returns:
            Optional[LatentRecord]: Processed latent record or None.
        """
        record = LatentRecord(buffer_output.latent)
        if self.constructor is not None:
            self.constructor(record=record, buffer_output=buffer_output)
        if self.sampler is not None:
            self.sampler(record)
        if self.transform is not None:
            self.transform(record)
        return record


class LatentLoader:
    """
    Loader class for processing latent records from a LatentDataset.
    """

    def __init__(
        self,
        latent_dataset: "LatentDataset",
        constructor: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize a LatentLoader.

        Args:
            latent_dataset (LatentDataset): The dataset to load latents from.
            constructor (Optional[Callable]): Function to construct latent records.
            sampler (Optional[Callable]): Function to sample from latent records.
            transform (Optional[Callable]): Function to transform latent records.
        """
        self.latent_dataset = latent_dataset
        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

    async def __aiter__(self):
        """
        Asynchronous iterator for processing latent records.

        Yields:
            LatentRecord: Processed latent records.
        """
        for buffer in self.latent_dataset.buffers:
            async for record in self._aprocess_buffer(buffer):
                yield record

    async def _aprocess_buffer(self, buffer):
        """
        Asynchronously process a buffer.

        Args:
            buffer (TensorBuffer): Buffer to process.

        Yields:
            Optional[LatentRecord]: Processed latent record or None.
        """
        for data in buffer:
            if data is not None:
                record = await self._aprocess_latent(data)
                if record is not None:
                    yield record
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def _aprocess_latent(self, buffer_output):
        """
        Asynchronously process a single latent.

        Args:
            buffer_output (BufferOutput): Latent data to process.

        Returns:
            Optional[LatentRecord]: Processed latent record or None.
        """
        record = LatentRecord(buffer_output.latent)
        if self.constructor is not None:
            self.constructor(record=record, buffer_output=buffer_output)
        if self.sampler is not None:
            self.sampler(record)
        if self.transform is not None:
            self.transform(record)
        return record

    def __iter__(self):
        """
        Synchronous iterator for processing latent records.

        Yields:
            LatentRecord: Processed latent records.
        """
        for buffer in self.latent_dataset.buffers:
            for record in self._process_buffer(buffer):
                yield record

    def _process_buffer(self, buffer):
        """
        Process a buffer synchronously.

        Args:
            buffer (TensorBuffer): Buffer to process.

        Yields:
            Optional[LatentRecord]: Processed latent record or None.
        """
        for data in buffer:
            if data is not None:
                record = self._process_latent(data)
                if record is not None:
                    yield record

    def _process_latent(self, buffer_output):
        """
        Process a single latent synchronously.

        Args:
            buffer_output (BufferOutput): Latent data to process.

        Returns:
            Optional[LatentRecord]: Processed latent record or None.
        """
        record = LatentRecord(buffer_output.latent)
        if self.constructor is not None:
            self.constructor(record=record, buffer_output=buffer_output)
        if self.sampler is not None:
            self.sampler(record)
        if self.transform is not None:
            self.transform(record)
        return record
