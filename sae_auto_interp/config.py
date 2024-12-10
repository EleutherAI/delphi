from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable


@dataclass
class ExperimentConfig(Serializable):
    
    n_examples_train: int = 40
    """Number of examples to sample for training"""

    n_examples_test: int = 5
    """Number of examples to sample for testing"""

    n_quantiles: int = 20
    """Number of quantiles to sample"""

    example_ctx_len: int = 32
    """Length of each example"""

    n_random: int = 50
    """Number of random examples to sample"""

    train_type: Literal["top", "random", "quantiles"] = "random"
    """Type of sampler to use for training"""

    test_type: Literal["quantiles", "activation"] = "quantiles"
    """Type of sampler to use for testing"""




@dataclass
class FeatureConfig(Serializable):
    width: int = 131072
    """Number of features in the autoencoder"""

    min_examples: int = 200
    """Minimum number of examples for a feature to be included"""

    max_examples: int = 10000
    """Maximum number of examples for a feature to included"""

    n_splits: int = 5
    """Number of splits that features were devided into"""

    tokenizer_or_model_name: str = "data/tinystories/restricted_tokenizer_v2"
    """Name of tokenizer, often the same as the model name"""


@dataclass
class CacheConfig(Serializable):

    dataset_repo: str = "EleutherAI/rpj-v2-sample"
    """Dataset repository to use"""

    dataset_split: str = "train[:1%]"
    """Dataset split to use""" 

    dataset_name: str = ""
    """Dataset name to use"""

    dataset_row: str = "raw_content"
    """Dataset row to use"""

    batch_size: int = 32
    """Number of sequences to process in a batch"""

    ctx_len: int = 256
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)"""

    n_tokens: int = 10_000_000
    """Number of tokens to cache"""

    n_splits: int = 5
    """Number of splits to divide .safetensors into"""
