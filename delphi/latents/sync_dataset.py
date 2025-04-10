# sync_latent_dataset.py
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents.constructors import constructor
from delphi.latents.latents import ActivationData, Latent, LatentData, LatentRecord
from delphi.latents.samplers import sampler
from delphi.utils import load_tokenized_data


@dataclass
class TensorBuffer:
    """
    Lazy loading buffer for cached splits.
    """

    path: str
    """Path to the tensor file."""

    module_path: str
    """Path of the module."""

    latents: Optional[torch.Tensor] = None
    """Tensor of latent indices."""

    _tokens: Optional[torch.Tensor] = None
    """Tensor of tokens."""

    def __iter__(self) -> Iterator[LatentData]:
        """
        Iterate over the buffer, yielding LatentData objects.
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
    def tokens(self) -> Optional[torch.Tensor]:
        if self._tokens is None:
            self._tokens = self.load_tokens()
        return self._tokens

    def load_data_per_latent(
        self,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
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

    def load(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Load the tensor buffer's data.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Locations, activations,
                and tokens (if present in the cache).
        """
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

        return locations, activations, tokens

    def load_tokens(self) -> Optional[torch.Tensor]:
        _, _, tokens = self.load()
        return tokens


class SyncLatentDataset:
    """
    Synchronous version of LatentDataset which constructs TensorBuffers
    for each module and latent.
    """

    def __init__(
        self,
        raw_dir: str,
        sampler_cfg: SamplerConfig,
        constructor_cfg: ConstructorConfig,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        modules: Optional[List[str]] = None,
        latents: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize a SyncLatentDataset.

        Args:
            raw_dir: Directory containing raw latent data.
            sampler_cfg: Configuration for sampling examples.
            constructor_cfg: Configuration for constructing examples.
            tokenizer: Tokenizer used to tokenize the data.
            modules: list of module names to include.
            latents: Dictionary of latents per module.
        """
        self.constructor_cfg = constructor_cfg
        self.sampler_cfg = sampler_cfg
        self.buffers: List[TensorBuffer] = []
        self.all_data: Dict[str, Optional[Dict[int, ActivationData]]] = {}
        self.tokens = None

        if modules is None:
            self.modules = os.listdir(raw_dir)
        else:
            self.modules = modules

        # Check that modules exist
        for module in self.modules:
            if not os.path.exists(f"{raw_dir}/{module}"):
                raise FileNotFoundError(
                    f"Could not find {module} in {raw_dir}. "
                    "Please check the parameters passed to the dataset."
                )

        if len(self.modules) == 0:
            raise ValueError("No modules found in the cache folder")

        # Load cache configuration
        cache_config_path = f"{raw_dir}/{self.modules[0]}/config.json"
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

        # Build dataset
        if latents is None:
            self._build(raw_dir)
        else:
            self._build_selected(raw_dir, latents)

        # Handle neighbours if needed
        if self.constructor_cfg.non_activating_source == "neighbours":
            # path is always going to end with /latents
            split_path = raw_dir.split("/")[:-1]
            neighbours_path = "/".join(split_path) + "/neighbours"
            self.neighbours = self.load_neighbours(
                neighbours_path, self.constructor_cfg.neighbours_type
            )
            # Load all data for neighbours
            self.all_data = self._load_all_data(raw_dir, self.modules)
        else:
            self.neighbours = None

        self.load_tokens()

    def load_tokens(self) -> torch.Tensor:
        """
        Load tokenized data for the dataset.

        Returns:
            torch.Tensor: The tokenized dataset.
        """
        if not hasattr(self, "tokens") or self.tokens is None:
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

    def load_neighbours(self, neighbours_path: str, neighbours_type: str) -> Dict:
        """
        Load neighbour data for latents.

        Args:
            neighbours_path: Path to neighbours directory.
            neighbours_type: Type of neighbours to load.

        Returns:
            Dict of neighbours by module and latent index.
        """
        neighbours = {}
        for hookpoint in self.modules:
            with open(
                neighbours_path + f"/{hookpoint}-{neighbours_type}.json", "r"
            ) as f:
                neighbours[hookpoint] = json.load(f)
        return neighbours

    def _edges(self, raw_dir: str, module: str) -> List[Tuple[int, int]]:
        """
        Find edges (start and end indices) for safetensor files.

        Args:
            raw_dir: Directory containing raw latent data.
            module: Module name.

        Returns:
            List of (start, end) tuples for each safetensor file.
        """
        module_dir = Path(raw_dir) / module
        safetensor_files = [f for f in module_dir.glob("*.safetensors")]
        edges = []
        for file in safetensor_files:
            start, end = file.stem.split("_")
            edges.append((int(start), int(end)))
        edges.sort(key=lambda x: x[0])
        return edges

    def _build(self, raw_dir: str):
        """
        Build dataset buffers which load all cached latents.

        Args:
            raw_dir: Directory containing raw latent data.
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
        raw_dir: str,
        latents: Dict[str, torch.Tensor],
    ):
        """
        Build a dataset buffer which loads only selected latents.

        Args:
            raw_dir: Directory containing raw latent data.
            latents: Dictionary of latents per module.
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

    def _load_all_data(
        self, raw_dir: str, modules: List[str]
    ) -> Dict[str, Dict[int, ActivationData]]:
        """
        For each module, load all locations and activations.

        Args:
            raw_dir: Directory containing raw latent data.
            modules: List of module names to include.

        Returns:
            Dict of activation data by module and latent index.
        """
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

    def _process_sync(self) -> Dict[int, LatentRecord]:
        """
        Process all buffers synchronously.

        Returns:
            Dict mapping latent indices to LatentRecord objects.
        """
        results = {}

        for buffer in self.buffers:
            for latent_data in buffer:
                record = self._process_latent(latent_data)
                if record is not None:
                    results[record.latent.latent_index] = record

        return results

    def _process_latent(self, latent_data: LatentData) -> Optional[LatentRecord]:
        """
        Process a single latent synchronously.

        Args:
            latent_data: Latent data to process.

        Returns:
            Processed latent record or None.
        """
        # This should never happen but we need to type check
        if self.tokens is None:
            raise ValueError("Tokens are not loaded")

        record = LatentRecord(latent_data.latent)

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
            all_data=self.all_data[latent_data.module],
            tokenizer=self.tokenizer,
        )

        # Not enough examples to explain the latent
        if record is None:
            return None

        record = sampler(record, self.sampler_cfg)
        return record

    def load_examples(
        self,
    ) -> Tuple[Dict[str, List], Dict[str, float], PreTrainedTokenizer]:
        """
        Load examples synchronously.

        Returns:
            Tuple of (examples, max_activations, tokenizer).
        """
        all_examples = {}
        maximum_activations = {}

        processed_records = self._process_sync()

        for latent_idx, record in processed_records.items():
            latent_str = str(latent_idx)
            all_examples[latent_str] = record.train
            maximum_activations[latent_str] = record.max_activation

        return all_examples, maximum_activations, self.tokenizer
