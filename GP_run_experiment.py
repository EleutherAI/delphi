import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import asyncio
from functools import partial
from pathlib import Path
from typing import Callable

import orjson
import torch
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi.delphi.clients import Offline, OpenRouter
from delphi.delphi.config import (
    CacheConfig,
    ConstructorConfig,
    RunConfig,
    SamplerConfig,
)
from delphi.delphi.explainers import DefaultExplainer
from delphi.delphi.latents import LatentCache, LatentDataset  # , LatentRecord
from delphi.delphi.latents.neighbours import NeighbourCalculator
from delphi.delphi.log.result_analysis import log_results
from delphi.delphi.pipeline import (
    Pipe,
    Pipeline,
    fan_out_fan_in_wrapper,
    process_wrapper,
)
from delphi.delphi.scorers import DetectionScorer, FuzzingScorer
from delphi.delphi.sparse_coders import load_hooks_sparse_coders, load_sparse_coders
from delphi.delphi.utils import assert_type, load_tokenized_data


def load_artifacts(run_cfg: RunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        run_cfg.model,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
    )

    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        model,
        run_cfg,
        compile=True,
    )

    return run_cfg.hookpoints, hookpoint_to_sparse_encode, model, transcode


def create_neighbours(
    run_cfg: RunConfig,
    latents_path: Path,
    neighbours_path: Path,
    hookpoints: list[str],
):
    """
    Creates a neighbours file for the given hookpoints.
    """
    neighbours_path.mkdir(parents=True, exist_ok=True)

    constructor_cfg = run_cfg.constructor_cfg
    saes = (
        load_sparse_coders(run_cfg, device="cpu")
        if constructor_cfg.neighbours_type != "co-occurrence"
        else {}
    )

    for hookpoint in hookpoints:
        if constructor_cfg.neighbours_type == "co-occurrence":
            neighbour_calculator = NeighbourCalculator(
                cache_dir=latents_path / hookpoint, number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "decoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].cuda(), number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "encoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].cuda(), number_of_neighbours=250
            )
        else:
            raise ValueError(
                f"Neighbour type {constructor_cfg.neighbours_type} not supported"
            )

        neighbour_calculator.populate_neighbour_cache(constructor_cfg.neighbours_type)
        neighbour_calculator.save_neighbour_cache(f"{neighbours_path}/{hookpoint}")


async def process_cache(
    run_cfg: RunConfig,
    latents_path: Path,
    neighbours_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
    # non_active_to_show: int,
):
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `fuzz_scores_path` directory.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)

    fuzz_scores_path = scores_path / "fuzz"
    detection_scores_path = scores_path / "detection"
    fuzz_scores_path.mkdir(parents=True, exist_ok=True)
    detection_scores_path.mkdir(parents=True, exist_ok=True)

    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {
            hook: latent_range for hook in hookpoints
        }  # The latent range to explain

    dataset = LatentDataset(
        raw_dir=str(latents_path),
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
        neighbours_path=str(neighbours_path),
    )

    if run_cfg.explainer_provider == "offline":
        client = Offline(
            run_cfg.explainer_model,
            max_memory=0.9,
            # Explainer models context length - must be able to accommodate the longest
            # set of examples
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
        )
    elif run_cfg.explainer_provider == "openrouter":
        client = OpenRouter(
            run_cfg.explainer_model,
            api_key="",
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    def explainer_postprocess(result):
        result_dict = {}
        result_dict["explanation"] = result.explanation
        # result_dict["short_name"]=result.short_name
        # result_dict["confidence"]=result.confidence
        with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
            f.write(orjson.dumps(result_dict))
        return result

    def explainer_preprocess(result):
        if result is None:
            return None
        record = result
        # remove the first non_active examples, save the rest in extra_examples
        # record.extra_examples = record.not_active[non_active_to_show:]
        # record.not_active = record.not_active[:non_active_to_show]

        return record

    explainer_pipe = process_wrapper(
        DefaultExplainer(
            client,
            threshold=0.3,
            verbose=run_cfg.verbose,
        ),
        preprocess=explainer_preprocess,
        postprocess=explainer_postprocess,
    )

    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        record = result.record

        record.explanation = result.explanation
        # record.not_active = record.extra_examples

        return record

    # Saves the score to a file
    def scorer_postprocess(result, score_dir):
        safe_latent_name = str(result.record.latent).replace("/", "--")

        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    # change the scorer wrapper to handle fan-out if using best of k.
    def scorer_wrapper(function: Callable):
        if run_cfg.explainer == "best_of_k":
            return process_wrapper(
                fan_out_fan_in_wrapper(function),
                preprocess=scorer_preprocess,
                postprocess=partial(
                    scorer_postprocess, score_dir=detection_scores_path
                ),
            )
        else:
            return process_wrapper(
                function,
                preprocess=scorer_preprocess,
                postprocess=partial(
                    scorer_postprocess, score_dir=detection_scores_path
                ),
            )

    scorer_pipe = Pipe(
        scorer_wrapper(
            DetectionScorer(
                client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=True,
            ),
        ),
        FuzzingScorer(
            client,
            n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
            verbose=run_cfg.verbose,
            log_prob=True,
        ),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=fuzz_scores_path),
    )

    pipeline = Pipeline(
        dataset,
        explainer_pipe,
        scorer_pipe,
    )

    await pipeline.run(10)


def populate_cache(
    run_cfg: RunConfig,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    transcode: bool,
):
    """
    Populates an on-disk cache in `latents_path` with SAE latent activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg

    tokens = load_tokenized_data(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        cache_cfg.dataset_name,
        cache_cfg.dataset_column,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        transcode=transcode,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens)

    cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


def non_redundant_hookpoints(
    hookpoint_to_sparse_encode: dict[str, Callable] | list[str],
    results_path: Path,
    overwrite: bool,
) -> dict[str, Callable] | list[str]:
    """
    Returns a list of hookpoints that are not already in the cache.
    """
    if overwrite:
        print("Overwriting results from", results_path)
        return hookpoint_to_sparse_encode
    in_results_path = [x.name for x in results_path.glob("*")]
    if isinstance(hookpoint_to_sparse_encode, dict):
        non_redundant_hookpoints = {
            k: v
            for k, v in hookpoint_to_sparse_encode.items()
            if k not in in_results_path
        }
    else:
        non_redundant_hookpoints = [
            hookpoint
            for hookpoint in hookpoint_to_sparse_encode
            if hookpoint not in in_results_path
        ]
    if not non_redundant_hookpoints:
        print(f"Files found in {results_path}, skipping...")
    return non_redundant_hookpoints


async def run(
    run_cfg: RunConfig,
    # start_latent: int,
    # non_active_to_show: int,
):
    base_path = Path.cwd() / "results"

    # latents_path =  base_path / "latents"

    base_path = base_path / run_cfg.name

    base_path.mkdir(parents=True, exist_ok=True)

    run_cfg.save_json(base_path / "run_config.json", indent=4)

    # All latents will be in the first part of the name

    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"
    neighbours_path = base_path / "neighbours"
    visualize_path = base_path / "visualize"

    latent_range = torch.arange(run_cfg.max_latents) if run_cfg.max_latents else None

    hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(run_cfg)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model, token=run_cfg.hf_token)

    nrh = assert_type(
        dict,
        non_redundant_hookpoints(
            hookpoint_to_sparse_encode, latents_path, "cache" in run_cfg.overwrite
        ),
    )

    if nrh:
        populate_cache(
            run_cfg,
            model,
            hookpoint_to_sparse_encode,
            latents_path,
            tokenizer,
            transcode,
        )

    del model, hookpoint_to_sparse_encode
    if run_cfg.constructor_cfg.non_activating_source == "neighbours":
        nrh = assert_type(
            list,
            non_redundant_hookpoints(
                hookpoints, neighbours_path, "neighbours" in run_cfg.overwrite
            ),
        )
        if nrh:
            create_neighbours(
                run_cfg,
                latents_path,
                neighbours_path,
                nrh,
            )
    else:
        print("Skipping neighbour creation")

    nrh = assert_type(
        list,
        non_redundant_hookpoints(
            hookpoints, scores_path, "scores" in run_cfg.overwrite
        ),
    )
    if nrh:
        await process_cache(
            run_cfg,
            latents_path,
            neighbours_path,
            explanations_path,
            scores_path,
            nrh,
            tokenizer,
            latent_range,
            # non_active_to_show,
        )

    if run_cfg.verbose:
        log_results(scores_path, visualize_path, run_cfg.hookpoints, run_cfg.scorers)


if __name__ == "__main__":
    print("Creating cache config")
    # Create the individual config objects
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/SmolLM2-135M-10B",
        # dataset_split="train[:1%]",
        # dataset_name="",
        # dataset_column="text",  # default
        # batch_size=32,
        cache_ctx_len=32,
        n_tokens=10_000_00,
        # n_splits=5,
    )

    print("Creating constructor config")
    constructor_cfg = ConstructorConfig(
        # faiss_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        # faiss_embedding_cache_dir=".embedding_cache",
        # faiss_embedding_cache_enabled=True,
        example_ctx_len=32,
        min_examples=20,  # Increased to allow for test examples
        n_non_activating=10,  # Reduced for smoke test
        # center_examples=True,
        # non_activating_source="random",
        # neighbours_type="co-occurrence",
    )

    print("Creating sampler config")
    sampler_cfg = SamplerConfig(
        n_examples_train=20,  # Reduced for smoke test
        n_examples_test=10,  # Reduced for smoke test
        n_quantiles=5,  # Reduced to work with fewer examples
        # train_type="quantiles",
        # test_type="quantiles",
        # ratio_top=0.2,
    )

    print("Creating run config")
    # Create RunConfig object with the same parameters as the shell script
    run_cfg = RunConfig(
        cache_cfg=cache_cfg,
        # skip_generate_cache_if_exists=True,
        constructor_cfg=constructor_cfg,
        sampler_cfg=sampler_cfg,
        model="EleutherAI/pythia-70m",
        sparse_model="EleutherAI/sae-pythia-70m-32k",
        hookpoints=["layers.5"],
        # explainer_model="EleutherAI/pythia-70m",
        explainer_model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        explainer_model_max_len=5120,
        explainer_provider="offline",
        # explainer="best_of_k",
        explainer="default",
        # num_explanations=3,
        scorers=["fuzz", "detection"],
        name="pythia-70m-smoketest",
        max_latents=10,
        filter_bos=True,
        # log_probs=False,
        # load_in_8bit=False,
        # hf_token=None,
        # pipeline_num_proc=4,
        num_gpus=2,
        # seed=22,
        # verbose=True,
        # num_examples_per_scorer_prompt=5,
        # overwrite=[],
    )

    # NUM_LATENTS_PRINT = 10

    """parser = ArgumentParser()
    parser.add_arguments(RunConfig, dest="run_cfg")
    args = parser.parse_args()

    asyncio.run(run(args.run_cfg))"""
    asyncio.run(run(run_cfg))
