import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from jaxtyping import Float
from safetensors.numpy import save_file
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel

from delphi.config import CacheConfig
from delphi.latents.collect_activations import collect_activations

location_tensor_shape = Float[Tensor, "batch sequence num_latents"]
token_tensor_shape = Float[Tensor, "batch sequence"]


class Cache:
    """
    The Cache class stores latent locations and activations for modules.
    It provides methods for adding, saving, and retrieving non-zero activations.
    """

    def __init__(
        self,
        filters: dict[str, Float[Tensor, "indices"]] | None = None,
        batch_size: int = 64,
    ):
        """
        Initialize the Cache.

        Args:
            filters: Filters for selecting specific latents.
            batch_size: Size of batches for processing. Defaults to 64.
        """
        self.latent_locations_batches: dict[str, list[location_tensor_shape]] = (
            defaultdict(list)
        )
        self.latent_activations_batches: dict[str, list[location_tensor_shape]] = (
            defaultdict(list)
        )
        self.tokens_batches: dict[str, list[token_tensor_shape]] = defaultdict(list)

        self.latent_locations: dict[str, location_tensor_shape] = {}
        self.latent_activations: dict[str, location_tensor_shape] = {}
        self.tokens: dict[str, token_tensor_shape] = {}

        self.filters = filters
        self.batch_size = batch_size

    def add(
        self,
        latents: location_tensor_shape,
        tokens: token_tensor_shape,
        batch_number: int,
        module_path: str,
    ):
        """
        Add the latents from a module to the cache.

        Args:
            latents: Latent activations.
            tokens: Input tokens.
            batch_number: Current batch number.
            module_path: Path of the module.
        """
        latent_locations, latent_activations = self.get_nonzeros(latents, module_path)
        latent_locations = latent_locations.cpu()
        latent_activations = latent_activations.cpu()
        tokens = tokens.cpu()

        # Adjust batch indices
        latent_locations[:, 0] += batch_number * self.batch_size
        self.latent_locations_batches[module_path].append(latent_locations)
        self.latent_activations_batches[module_path].append(latent_activations)
        self.tokens_batches[module_path].append(tokens)

    def save(self):
        """
        Concatenate the latent locations and activations for all modules.
        """
        for module_path in self.latent_locations_batches.keys():
            self.latent_locations[module_path] = torch.cat(
                self.latent_locations_batches[module_path], dim=0
            )

            self.latent_activations[module_path] = torch.cat(
                self.latent_activations_batches[module_path], dim=0
            )

            self.tokens[module_path] = torch.cat(
                self.tokens_batches[module_path], dim=0
            )

    def get_nonzeros_batch(
        self, latents: location_tensor_shape
    ) -> tuple[
        Float[Tensor, "batch sequence num_latents"], Float[Tensor, "batch sequence "]
    ]:
        """
        Get non-zero activations for large batches that exceed int32 max value.

        Args:
            latents: Input latent activations.

        Returns:
            tuple[Tensor, Tensor]: Non-zero latent locations and activations.
        """
        # Calculate the maximum batch size that fits within sys.maxsize
        max_batch_size = torch.iinfo(torch.int32).max // (
            latents.shape[1] * latents.shape[2]
        )
        nonzero_latent_locations = []
        nonzero_latent_activations = []

        for i in range(0, latents.shape[0], max_batch_size):
            batch = latents[i : i + max_batch_size]

            # Get nonzero locations and activations
            batch_locations = torch.nonzero(batch.abs() > 1e-5)
            batch_activations = batch[batch.abs() > 1e-5]

            # Adjust indices to account for batching
            batch_locations[:, 0] += i
            nonzero_latent_locations.append(batch_locations)
            nonzero_latent_activations.append(batch_activations)

        # Concatenate results
        nonzero_latent_locations = torch.cat(nonzero_latent_locations, dim=0)
        nonzero_latent_activations = torch.cat(nonzero_latent_activations, dim=0)
        return nonzero_latent_locations, nonzero_latent_activations

    def get_nonzeros(self, latents: location_tensor_shape, module_path: str) -> tuple[
        location_tensor_shape,
        location_tensor_shape,
    ]:
        """
        Get the nonzero latent locations and activations.

        Args:
            latents: Input latent activations.
            module_path: Path of the module.

        Returns:
            tuple[Tensor, Tensor]: Non-zero latent locations and activations.
        """
        size = latents.shape[1] * latents.shape[0] * latents.shape[2]
        if size > torch.iinfo(torch.int32).max:
            (
                nonzero_latent_locations,
                nonzero_latent_activations,
            ) = self.get_nonzeros_batch(latents)
        else:
            nonzero_latent_locations = torch.nonzero(latents.abs() > 1e-5)
            nonzero_latent_activations = latents[latents.abs() > 1e-5]

        # Return all nonzero latents if no filter is provided
        if self.filters is None:
            return nonzero_latent_locations, nonzero_latent_activations

        # Return only the selected latents if a filter is provided
        else:
            selected_latents = self.filters[module_path]
            mask = torch.isin(nonzero_latent_locations[:, 2], selected_latents)

            return nonzero_latent_locations[mask], nonzero_latent_activations[mask]


class LatentCache:
    """
    LatentCache manages the caching of latent activations for a model.
    Handles the process of running the model, storing activations, and saving to disk
    """

    def __init__(
        self,
        model: PreTrainedModel,
        hookpoint_to_sparse_encode: dict[str, Callable],
        batch_size: int,
        filters: dict[str, Float[Tensor, "indices"]] | None = None,
    ):
        """
        Initialize the LatentCache.

        Args:
            model: The model to cache latents for.
            hookpoint_to_sparse_encode: Dictionary of sparse encoding functions.
            batch_size: Size of batches for processing.
            filters: Filters for selecting specific latents.
        """
        self.model = model
        self.hookpoint_to_sparse_encode = hookpoint_to_sparse_encode

        self.batch_size = batch_size
        self.width = None
        self.cache = Cache(filters, batch_size=batch_size)
        if filters is not None:
            self.filter_submodules(filters)

    def load_token_batches(
        self, n_tokens: int, tokens: token_tensor_shape
    ) -> list[token_tensor_shape]:
        """
        Load and prepare token batches for processing.

        Args:
            n_tokens: Total number of tokens to process.
            tokens: Input tokens.

        Returns:
            list[Tensor]: list of token batches.
        """
        max_batches = n_tokens // tokens.shape[1]
        tokens = tokens[:max_batches]

        n_mini_batches = len(tokens) // self.batch_size

        token_batches = [
            tokens[self.batch_size * i : self.batch_size * (i + 1), :]
            for i in range(n_mini_batches)
        ]

        return token_batches

    def filter_submodules(self, filters: dict[str, Float[Tensor, "indices"]]):
        """
        Filter submodules based on the provided filters.

        Args:
            filters: Filters for selecting specific latents.
        """
        filtered_submodules = {}
        for hookpoint in self.hookpoint_to_sae.keys():
            if hookpoint in filters:
                filtered_submodules[hookpoint] = self.hookpoint_to_sae[hookpoint]
        self.hookpoint_to_sae = filtered_submodules

    def run(
        self,
        n_tokens: int,
        tokens: token_tensor_shape,
        detuple_activations: bool = True,
    ):
        """
        Run the latent caching process.

        Args:
            n_tokens: Total number of tokens to process.
            tokens: Input tokens.
        """
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()

        with tqdm(total=total_batches, desc="Caching latents") as pbar:
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                with torch.inference_mode():
                    with collect_activations(
                        self.model,
                        list(self.hookpoint_to_sparse_encode.keys()),
                        detuple_activations=detuple_activations,
                    ) as activations:
                        self.model(batch.to(self.model.device))

                        for hookpoint, latents in activations.items():
                            sae_latents = self.hookpoint_to_sparse_encode[hookpoint](
                                latents
                            )
                            self.cache.add(sae_latents, batch, batch_number, hookpoint)
                            if self.width is None:
                                self.width = sae_latents.shape[2]
                            del sae_latents
                        del activations

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        print(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()
        try:
            del sae_latents
        except UnboundLocalError:
            pass

    def save(self, save_dir: Path, save_tokens: bool = True):
        """
        Save the cached latents to disk.

        Args:
            save_dir: Directory to save the latents.
            save_tokens: Whether to save the dataset tokens used to generate the cache.
            Defaults to True.
        """
        for module_path in self.cache.latent_locations.keys():
            output_file = save_dir / f"{module_path}.safetensors"

            data = {
                "locations": self.cache.latent_locations[module_path].numpy(),
                "activations": self.cache.latent_activations[module_path].numpy(),
            }
            if save_tokens:
                data["tokens"] = self.cache.tokens[module_path].numpy()

            save_file(data, output_file)

    def _generate_split_indices(self, n_splits: int) -> list[tuple[Tensor, Tensor]]:
        """
        Generate indices for splitting the latent space.

        Args:
            n_splits: Number of splits to generate.

        Returns:
            list[tuple[int, int]]: list of start and end indices for each split.
        """
        assert self.width is not None, "Width must be set before generating splits"
        boundaries = torch.linspace(0, self.width, steps=n_splits + 1).long()

        # Adjust end by one
        return list(zip(boundaries[:-1], boundaries[1:] - 1))

    def estimate_size(self) -> int:
        """
        Estimate the size of the cache in bytes.

        Returns:
            int: Estimated size of the cache in bytes.
        """
        total_size = 0
        for module_path in self.cache.latent_locations.keys():
            total_size += (
                self.cache.latent_locations[module_path].element_size()
                * self.cache.latent_locations[module_path].numel()
            )
            total_size += (
                self.cache.latent_activations[module_path].element_size()
                * self.cache.latent_activations[module_path].numel()
            )
            total_size += (
                self.cache.tokens[module_path].element_size()
                * self.cache.tokens[module_path].numel()
            )

        return total_size

    def save_splits(self, n_splits: int, save_dir: Path, save_tokens: bool = True):
        """
        Save the cached non-zero latent activations and locations in splits.

        Args:
            n_splits: Number of splits to generate.
            save_dir: Directory to save the splits.
            save_tokens: Whether to save the dataset tokens used to generate the cache.
            Defaults to True.
        """
        split_indices = self._generate_split_indices(n_splits)
        for module_path in self.cache.latent_locations.keys():
            latent_locations = self.cache.latent_locations[module_path]
            latent_activations = self.cache.latent_activations[module_path]
            tokens = self.cache.tokens[module_path].numpy()

            latent_indices = latent_locations[:, 2]

            for start, end in split_indices:
                mask = (latent_indices >= start) & (latent_indices <= end)

                masked_activations = latent_activations[mask].half().numpy()

                masked_locations = latent_locations[mask].numpy()

                # Optimization to reduce the max value to enable a smaller dtype
                masked_locations[:, 2] = masked_locations[:, 2] - start.item()

                if (
                    masked_locations[:, 2].max() < 2**16
                    and masked_locations[:, 0].max() < 2**16
                ):
                    masked_locations = masked_locations.astype(np.uint16)
                else:
                    masked_locations = masked_locations.astype(np.uint32)
                    print(
                        "Warning: Increasing the number of splits might reduce the"
                        "memory usage of the cache."
                    )

                module_dir = save_dir / module_path
                module_dir.mkdir(parents=True, exist_ok=True)

                output_file = module_dir / f"{start}_{end}.safetensors"

                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }
                if save_tokens:
                    split_data["tokens"] = tokens

                save_file(split_data, output_file)

    def save_config(self, save_dir: Path, cfg: CacheConfig, model_name: str):
        """
        Save the configuration for the cached latents.

        Args:
            save_dir: Directory to save the configuration.
            cfg: Configuration object.
            model_name: Name of the model.
        """
        for module_path in self.cache.latent_locations.keys():
            config_file = save_dir / module_path / "config.json"
            with open(config_file, "w") as f:
                config_dict = cfg.to_dict()
                config_dict["model_name"] = model_name
                json.dump(config_dict, f, indent=4)
