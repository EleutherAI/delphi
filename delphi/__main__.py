import asyncio
import logging
import os
from functools import partial
from pathlib import Path
from typing import Callable

import orjson
import torch
from simple_parsing import ArgumentParser
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi import logger
from delphi.clients import Offline, OpenRouter
from delphi.config import RunConfig
from delphi.explainers import ContrastiveExplainer, DefaultExplainer, NoOpExplainer
from delphi.explainers.explainer import ExplainerResult
from delphi.latents import LatentCache, LatentDataset
from delphi.latents.neighbours import NeighbourCalculator
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer, OpenAISimulator
from delphi.sparse_coders import load_hooks_sparse_coders, load_sparse_coders
from delphi.utils import assert_type, load_tokenized_data


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

    return (
        list(hookpoint_to_sparse_encode.keys()),
        hookpoint_to_sparse_encode,
        model,
        transcode,
    )


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
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "encoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
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
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
):
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `scores_path` directory.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)

    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {
            hook: latent_range for hook in hookpoints
        }  # The latent range to explain

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    if run_cfg.explainer_provider == "offline":
        llm_client = Offline(
            run_cfg.explainer_model,
            max_memory=run_cfg.max_memory,
            # Explainer models context length - must be able to accommodate the longest
            # set of examples
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
        )
    elif run_cfg.explainer_provider == "openrouter":
        if (
            "OPENROUTER_API_KEY" not in os.environ
            or not os.environ["OPENROUTER_API_KEY"]
        ):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set "
                "`--explainer-provider offline` to use a local explainer model."
            )

        llm_client = OpenRouter(
            run_cfg.explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    if not run_cfg.explainer == "none":

        def explainer_postprocess(result):
            with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
                f.write(orjson.dumps(result.explanation))

            return result

        if run_cfg.constructor_cfg.non_activating_source == "FAISS":
            explainer = ContrastiveExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
            )
        else:
            explainer = DefaultExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
            )

        explainer_pipe = Pipe(
            process_wrapper(explainer, postprocess=explainer_postprocess)
        )
    else:

        def none_postprocessor(result):
            # Load the explanation from disk
            explanation_path = explanations_path / f"{result.record.latent}.txt"
            if not explanation_path.exists():
                raise FileNotFoundError(
                    f"Explanation file {explanation_path} does not exist. "
                    "Make sure to run an explainer pipeline first."
                )

            with open(explanation_path, "rb") as f:
                return ExplainerResult(
                    record=result.record,
                    explanation=orjson.loads(f.read()),
                )

        explainer_pipe = Pipe(
            process_wrapper(
                NoOpExplainer(),
                postprocess=none_postprocessor,
            )
        )

    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        if isinstance(result, list):
            result = result[0]

        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active
        return record

    # Saves the score to a file
    def scorer_postprocess(result, score_dir):
        safe_latent_name = str(result.record.latent).replace("/", "--")

        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorers = []
    for scorer_name in run_cfg.scorers:
        scorer_path = scores_path / scorer_name
        scorer_path.mkdir(parents=True, exist_ok=True)

        if scorer_name == "simulation":
            if isinstance(llm_client, Offline):
                scorer = OpenAISimulator(
                    llm_client, tokenizer=tokenizer, all_at_once=True
                )
            else:
                scorer = OpenAISimulator(
                    llm_client, tokenizer=tokenizer, all_at_once=False
                )
        elif scorer_name == "fuzz":
            scorer = FuzzingScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            )
        elif scorer_name == "detection":
            scorer = DetectionScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            )
        else:
            raise ValueError(f"Scorer {scorer_name} not supported")

        wrapped_scorer = process_wrapper(
            scorer,
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=scorer_path),
        )
        scorers.append(wrapped_scorer)

    pipeline = Pipeline(
        dataset,
        explainer_pipe,
        Pipe(*scorers),
    )

    if run_cfg.pipeline_num_proc > 1 and run_cfg.explainer_provider == "openrouter":
        print(
            "OpenRouter does not support multiprocessing,"
            " setting pipeline_num_proc to 1"
        )
        run_cfg.pipeline_num_proc = 1

    await pipeline.run(run_cfg.pipeline_num_proc)


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

    if run_cfg.verbose:
        cache.generate_statistics_cache()

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
):
    base_path = Path.cwd() / "results"
    if run_cfg.name:
        base_path = base_path / run_cfg.name

    base_path.mkdir(parents=True, exist_ok=True)

    run_cfg.save_json(base_path / "run_config.json", indent=4)

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
            nrh,
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
            explanations_path,
            scores_path,
            nrh,
            tokenizer,
            latent_range,
        )

    if run_cfg.verbose:
        log_results(scores_path, visualize_path, run_cfg.hookpoints, run_cfg.scorers)


if __name__ == "__main__":
    # Configure logging for CLI usage
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("delphi.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    parser = ArgumentParser()
    parser.add_arguments(RunConfig, dest="run_cfg")
    args = parser.parse_args()

    asyncio.run(run(args.run_cfg))
