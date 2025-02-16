from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable, field


@dataclass
class ExperimentConfig(Serializable):
    n_examples_train: int = 40
    """Number of examples to sample for latent explanation generation."""

    n_examples_test: int = 50
    """Number of examples to sample for latent explanation testing."""

    n_quantiles: int = 10
    """Number of latent activation quantiles to sample."""

    example_ctx_len: int = 32
    """Length of each sampled example sequence. Longer sequences
    reduce detection scoring performance in weak models."""

    n_non_activating: int = 50
    """Number of non-activating examples to sample."""

    train_type: Literal["top", "random", "quantiles"] = "quantiles"
    """Type of sampler to use for latent explanation generation."""

    test_type: Literal["quantiles", "activation"] = "quantiles"
    """Type of sampler to use for latent explanation testing."""


@dataclass
class LatentConfig(Serializable):
    width: int = 131_072
    """Number of latents in each autoencoder"""

    min_examples: int = 200
    """Minimum number of examples to generate for a single latent.
    If the number of activating examples is less than this, the
    latent will not be explained and scored."""

    max_examples: int = 10_000
    """Maximum number of examples to generate for a single latent."""

    n_splits: int = 5
    """Number of splits that latents will be divided into."""


@dataclass
class CacheConfig(Serializable):
    dataset_repo: str = "EleutherAI/fineweb-edu-dedup-10b"
    """Dataset repository to use for generating latent activations."""

    faiss_index_type: Literal["flat", "hnsw"] = "hnsw"

    faiss_hnsw_config: dict[str, int] = field(
        default_factory=lambda: {"M": 32, "efConstruction": 200, "efSearch": 128}
    )

    faiss_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    dataset_split: str = "train[:1%]"
    """Dataset split to use for generating latent activations."""

    dataset_name: str = ""
    """Dataset name to use."""

    dataset_row: str = "text"
    """Dataset row to use."""

    batch_size: int = 32
    """Number of sequences to process in a batch."""

    ctx_len: int = 256
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)."""

    n_tokens: int = 10_000_000
    """Number of tokens to cache."""

    n_splits: int = 5
    """Number of splits to divide .safetensors into."""
