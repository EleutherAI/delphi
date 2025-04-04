import random
from typing import Literal

from ..config import SamplerConfig
from ..logger import logger
from .latents import ActivatingExample, LatentRecord


def normalize_activations(
    examples: list[ActivatingExample], max_activation: float, eps: float = 1e-6
) -> list[ActivatingExample]:
    max_activation = max(max_activation, eps)
    for example in examples:
        example.normalized_activations = (
            (example.activations * 10 / max_activation).floor().clamp(0, 10)
        )
    return examples


def split_quantiles(
    examples: list[ActivatingExample], n_quantiles: int, n_samples: int, seed: int = 22
) -> list[ActivatingExample]:
    """
    Randomly select (n_samples // n_quantiles) samples from each quantile.
    """
    random.seed(seed)

    quantile_size = len(examples) // n_quantiles
    samples_per_quantile = n_samples // n_quantiles
    samples: list[ActivatingExample] = []
    for i in range(n_quantiles):
        # Take an evenly spaced slice of the examples for the quantile.
        quantile = examples[i * quantile_size : (i + 1) * quantile_size]

        # Take a random sample of the examples.
        if len(quantile) < samples_per_quantile:
            sample = quantile
            logger.info(
                f"Quantile {i} has fewer than {samples_per_quantile} samples, using all"
            )
        else:
            sample = random.sample(quantile, samples_per_quantile)
        # set the quantile index
        for example in sample:
            example.quantile = i
        samples.extend(sample)

    return samples


def train(
    examples: list[ActivatingExample],
    max_activation: float,
    n_train: int,
    train_type: Literal["top", "random", "quantiles", "mix"],
    n_quantiles: int = 10,
    seed: int = 22,
    ratio_top: float = 0.2,
):
    match train_type:
        case "top":
            selected_examples = examples[:n_train]
            selected_examples = normalize_activations(selected_examples, max_activation)
            return selected_examples
        case "random":
            random.seed(seed)
            n_sample = min(n_train, len(examples))
            if n_sample < n_train:
                logger.warning(
                    "n_train is greater than the number of examples, using all examples"
                )

            selected_examples = random.sample(examples, n_train)
            selected_examples = normalize_activations(selected_examples, max_activation)
            return selected_examples
        case "quantiles":
            selected_examples = split_quantiles(examples, n_quantiles, n_train)
            selected_examples = normalize_activations(selected_examples, max_activation)
            return selected_examples
        case "mix":
            # TODO: currently it will always do 1/5 top and 4/5 quantiles
            top_examples = examples[: int(n_train * ratio_top)]
            quantiles_examples = split_quantiles(
                examples[int(n_train * ratio_top) :],
                n_quantiles,
                int(n_train * (1 - ratio_top)),
            )
            selected_examples = top_examples + quantiles_examples
            selected_examples = normalize_activations(selected_examples, max_activation)
            return selected_examples


def test(
    examples: list[ActivatingExample],
    max_activation: float,
    n_test: int,
    n_quantiles: int,
    test_type: Literal["quantiles"],
):
    match test_type:
        case "quantiles":
            selected_examples = split_quantiles(examples, n_quantiles, n_test)
            selected_examples = normalize_activations(selected_examples, max_activation)
            return selected_examples


def sampler(
    record: LatentRecord,
    cfg: SamplerConfig,
):
    examples = record.examples
    max_activation = record.max_activation
    _train = train(
        examples,
        max_activation,
        cfg.n_examples_train,
        cfg.train_type,
        n_quantiles=cfg.n_quantiles,
        ratio_top=cfg.ratio_top,
    )
    record.train = _train
    if cfg.n_examples_test > 0:
        _test = test(
            examples,
            max_activation,
            cfg.n_examples_test,
            cfg.n_quantiles,
            cfg.test_type,
        )
        record.test = _test
    return record
