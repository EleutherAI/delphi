import asyncio
import time
from pathlib import Path

import torch
from beartype.claw import beartype_package

beartype_package("delphi")

from delphi.__main__ import run  # noqa: E402
from delphi.config import (  # noqa: E402
    CacheConfig,
    ConstructorConfig,
    RunConfig,
    SamplerConfig,
)
from delphi.log.result_analysis import (  # noqa: E402
    build_scores_df,
    latent_balanced_score_metrics,
)


async def test():
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:1%]",
        dataset_column="text",
        batch_size=8,
        cache_ctx_len=256,
        n_splits=5,
        n_tokens=200_000,
    )
    sampler_cfg = SamplerConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
        n_quantiles=10,
    )
    constructor_cfg = ConstructorConfig(
        min_examples=200,
        max_examples=10_000,
        example_ctx_len=32,
        n_non_activating=50,
        non_activating_source="random",
    )
    run_cfg = RunConfig(
        name="test",
        overwrite=["cache", "scores"],
        model="EleutherAI/pythia-160m",
        sparse_model="EleutherAI/sae-pythia-160m-32k",
        hookpoints=["layers.3.mlp"],
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        explainer_model_max_len=4208,
        max_latents=100,
        seed=22,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True,
        verbose=True,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        cache_cfg=cache_cfg,
    )

    start_time = time.time()
    await run(run_cfg)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # Performs better than random guessing
    scores_path = Path.cwd() / "results" / run_cfg.name / "scores"
    hookpoint_firing_counts = torch.load(
        Path.cwd() / "results" / run_cfg.name / "log" / "hookpoint_firing_counts.pt",
        weights_only=True,
    )
    df = build_scores_df(scores_path, run_cfg.hookpoints, hookpoint_firing_counts)
    for score_type in df["score_type"].unique():
        score_df = df.query(f"score_type == '{score_type}'")

        weighted_mean_metrics = latent_balanced_score_metrics(
            score_df, score_type, verbose=False
        )

        accuracy = weighted_mean_metrics["accuracy"]
        assert accuracy > 0.55, f"Score type {score_type} has an accuracy of {accuracy}"


if __name__ == "__main__":
    asyncio.run(test())
