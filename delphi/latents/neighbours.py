import json
from typing import Optional

import cupy as cp
import cupyx.scipy.sparse as cusparse
import numpy as np
import torch
from safetensors.numpy import load_file

from delphi.latents.loader import LatentDataset


class NeighbourCalculator:
    """
    Class to compute the neighbours of selected latents using different methods:
    - similarity: uses autoencoder weights
    - correlation: uses pre-activation records and autoencoder
    - co-occurrence: uses latent dataset statistics
    """

    def __init__(
        self,
        latent_dataset: Optional["LatentDataset"] = None,
        autoencoder: Optional["Autoencoder"] = None,
        pre_activation_record: Optional["PreActivationRecord"] = None,
        number_of_neighbours: int = 10,
        neighbour_cache: Optional[dict[str, dict[int, list[int]]]] = None,
    ):
        """
        Initialize a NeighbourCalculator.

        Args:
            latent_dataset (Optional[LatentDataset]): Dataset containing latent
                activations
            autoencoder (Optional[Autoencoder]): The trained autoencoder model
            pre_activation_record (Optional[PreActivationRecord]): Record of
                pre-activation values
        """
        self.latent_dataset = latent_dataset
        self.autoencoder = autoencoder
        self.pre_activation_record = pre_activation_record

        # load the neighbour cache from the path
        if neighbour_cache is not None:
            self.neighbour_cache = neighbour_cache
        else:
            # Dictionary to cache computed neighbour lists
            self.neighbour_cache: dict[str, dict[int, list[int]]] = {}

    def _compute_neighbour_list(self, method: str) -> None:
        """
        Compute complete neighbour lists using specified method.

        Args:
            method (str): One of 'similarity', 'correlation', or 'co-occurrence'
        """
        if method == "similarity":
            if self.autoencoder is None:
                raise ValueError(
                    "Autoencoder is required for similarity-based neighbours"
                )
            self.neighbour_cache[method] = self._compute_similarity_neighbours()

        elif method == "correlation":
            if self.autoencoder is None or self.pre_activation_record is None:
                raise ValueError(
                    "Autoencoder and pre-activation record are required for "
                    "correlation-based neighbours"
                )
            self.neighbour_cache[method] = self._compute_correlation_neighbours()

        elif method == "co-occurrence":
            if self.latent_dataset is None:
                raise ValueError(
                    "Latent dataset is required for co-occurrence-based neighbours"
                )
            self.neighbour_cache[method] = self._compute_cooccurrence_neighbours()

        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_similarity_neighbours(self) -> dict[int, list[int]]:
        """
        Compute neighbour lists based on weight similarity in the autoencoder.
        """

        # We use the encoder vectors to compute the similarity between latents
        encoder = self.autoencoder.encoder

        weight_matrix_normalized = encoder.weight / encoder.weight.norm(
            dim=1, keepdim=True
        )

        # Compute the similarity between latents
        similarity_matrix = weight_matrix_normalized.T @ weight_matrix_normalized

        # Get the indices of the top k neighbours for each latent
        top_k_indices = torch.topk(
            similarity_matrix, self.number_of_neighbours, dim=1
        ).indices

        # Return the neighbour lists
        return {i: top_k_indices[i].tolist() for i in range(len(top_k_indices))}

    def _compute_correlation_neighbours(self) -> dict[int, list[int]]:
        """
        Compute neighbour lists based on activation correlation patterns.
        """

        # the preactivation_matrix has the shape (number_of_samples,hidden_dimension)
        preactivation_matrix = self.pre_activation_record

        # compute the covariance matrix of the preactivation matrix
        covariance_matrix = torch.cov(preactivation_matrix.T)

        # load the encoder
        encoder_matrix = self.autoencoder.encoder.weight

        # covariance between the latents is u^T * covariance_matrix * u
        covariance_between_latents = (
            encoder_matrix.T @ covariance_matrix @ encoder_matrix
        )

        # the correlation is then the covariance devided by the product of stddevs

        product_of_std = torch.diag(covariance_matrix) ** 2

        correlation_matrix = covariance_between_latents / product_of_std

        # get the indices of the top k neighbours for each latent
        top_k_indices = torch.topk(
            correlation_matrix, self.number_of_neighbours, dim=1
        ).indices

        # return the neighbour lists
        return {i: top_k_indices[i].tolist() for i in range(len(top_k_indices))}

    def _compute_cooccurrence_neighbours(self) -> dict[int, list[int]]:
        """
        Compute neighbour lists based on latent co-occurrence in the dataset.
        """
        # To be implemented

        paths = []
        for buffer in self.latent_dataset.buffers:
            paths.append(buffer.path)

        all_locations = []
        all_activations = []
        for path in paths:
            split_data = load_file(path)
            first_latent = int(path.split("/")[-1].split("_")[0])
            activations = torch.tensor(split_data["activations"])
            locations = torch.tensor(split_data["locations"].astype(np.int64))
            locations[:, 2] = locations[:, 2] + first_latent
            all_locations.append(locations)
            all_activations.append(activations)

        # concatenate the locations and activations
        locations = torch.cat(all_locations).cuda()
        activations = torch.cat(all_activations).cuda()
        n_latents = int(torch.max(locations[:, 2])) + 1

        # 1. Get unique values of first 2 dims (i.e. absolute token index) and counts
        # Trick is to use Cantor pairing function to have a bijective mapping between
        # (batch_id, ctx_pos) and a unique 1D index
        # Faster than running `torch.unique_consecutive` on the first 2 dims
        idx_cantor = (locations[:, 0] + locations[:, 1]) * (
            locations[:, 0] + locations[:, 1] + 1
        ) // 2 + locations[:, 1]
        unique_idx, idx_counts = torch.unique_consecutive(
            idx_cantor, return_counts=True
        )
        n_tokens = len(unique_idx)

        # 2. The Cantor indices are not consecutive, so create sorted ones from counts
        locations_flat = torch.repeat_interleave(
            torch.arange(n_tokens, device=locations.device), idx_counts
        )
        del idx_cantor, unique_idx, idx_counts

        rows = cp.asarray(locations[:, 2])
        cols = cp.asarray(locations_flat)
        data = cp.ones(len(rows))
        sparse_matrix = cusparse.coo_matrix(
            (data, (rows, cols)), shape=(n_latents, n_tokens)
        )
        cooc_matrix = sparse_matrix @ sparse_matrix.T

        # Compute Jaccard similarity
        def compute_jaccard(cooc_matrix):
            self_occurrence = cooc_matrix.diagonal()
            jaccard_matrix = cooc_matrix / (
                self_occurrence[:, None] + self_occurrence - cooc_matrix
            )
            return jaccard_matrix

        del rows, cols, data, sparse_matrix
        # Compute Jaccard similarity matrix
        jaccard_matrix = compute_jaccard(cooc_matrix)

        # get the indices of the top k neighbours for each latent
        top_k_indices = torch.topk(
            jaccard_matrix, self.number_of_neighbours, dim=1
        ).indices

        # return the neighbour lists
        return {i: top_k_indices[i].tolist() for i in range(len(top_k_indices))}

    def populate_neighbour_cache(self, methods: list[str]) -> None:
        """
        Populate the neighbour cache with the computed neighbour lists
        """
        for method in methods:
            self._compute_neighbour_list(method)

    def save_neighbour_cache(self) -> None:
        """
        Save the neighbour cache to the path as a json file
        """
        with open(self.path, "w") as f:
            json.dump(self.neighbour_cache, f)

    def load_neighbour_cache(self) -> dict[str, dict[int, list[int]]]:
        """
        Load the neighbour cache from the path as a json file
        """
        with open(self.path, "r") as f:
            return json.load(f)
