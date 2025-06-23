import asyncio
import time
from pathlib import Path

import torch

from delphi.__main__ import run
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.log.result_analysis import get_agg_metrics, load_data


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
        min_examples=90,
        example_ctx_len=32,
        n_non_activating=50,
        non_activating_source="random",
        faiss_embedding_cache_enabled=True,
        faiss_embedding_cache_dir=".embedding_cache",
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
        verbose=False,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        cache_cfg=cache_cfg,
    )

    start_time = time.time()
    await run(run_cfg)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    scores_path = Path.cwd() / "results" / run_cfg.name / "scores"

    latent_df, counts = load_data(scores_path, run_cfg.hookpoints)
    processed_df = get_agg_metrics(latent_df, counts)

    # Performs better than random guessing
    for score_type, df in processed_df.groupby("score_type"):
        accuracy = df["accuracy"].mean()
        assert accuracy > 0.55, f"Score type {score_type} has an accuracy of {accuracy}"


if __name__ == "__main__":
    asyncio.run(test())
