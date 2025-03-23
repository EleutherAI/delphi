import asyncio
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import replace
from enum import Enum
from functools import partial
from math import ceil
from pathlib import Path

import numpy as np
import orjson
import torch
from sentence_transformers import SentenceTransformer
from simple_parsing import ArgumentParser

from delphi.clients import OpenRouter
from delphi.config import ExperimentConfig, LatentConfig
from delphi.explainers import DefaultExplainer
from delphi.explainers.explainer import ExplainerResult
from delphi.latents import LatentDataset
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer

"""
uv run python -m examples.example_script --model monet_cache_converted/850m --module .model.layers.20.router --latents 6144 --random_subset
uv run python -m sglang_router.launch_server --model-path "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --port 8000 --host 0.0.0.0 --tensor-parallel-size=2 --mem-fraction-static=0.8 --dp-size 2
uv run python -m examples.example_script --model itda_cache/pythia-l9_mlp-transcoder-mean-skip-k32 --module gpt_neox.layers.9.mlp --latents 500 --width 50142 --random_subset
uv run python -m examples.example_script --model sae_pkm/baseline --module .model.layers.9 --latents 500 --random_subset
"""

# run with python examples/example_script.py --model gemma/16k --module
# .model.layers.10 --latents 100 --experiment_name test


class Substitution(Enum):
    NONE = "none"
    SELF = "self"
    OTHER = "other"
    SELF_RANDOM = "self_random"
    OTHER_RANDOM = "other_random"


async def main(args):
    module = args.module
    latent_cfg = args.latent_options
    experiment_cfg = args.experiment_options
    shown_examples = args.shown_examples
    n_latents = args.latents
    experiment_name = args.experiment_name
    start_latent = 0
    sae_model = args.model
    if args.neighbors:
        experiment_cfg.non_activating_source = "neighbors"
        experiment_name += "_neighbors"
        if args.decoder_similarity:
            experiment_cfg.neighbours_type = "decoder_similarity"
            experiment_name += "_ds"
    print("Experiment name:", experiment_name)
    print("Module:", module)
    print("Directory:", sae_model)
    experiment_name_scores = experiment_name
    substitute = args.substitute
    if substitute != Substitution.NONE:
        experiment_name_scores += "_substitute_" + substitute.value
    if substitute in (Substitution.SELF, Substitution.OTHER):
        embedding_model = SentenceTransformer(
            "NovaSearch/stella_en_400M_v5", trust_remote_code=True
        )

    raw_dir = f"results/{args.model}"
    latents = torch.arange(start_latent, start_latent + n_latents)
    cache_config_dir = f"{raw_dir}/{module}/config.json"
    with open(cache_config_dir, "r") as f:
        cache_config = json.load(f)
    if "width" in cache_config:
        latent_cfg.width = cache_config["width"]

    max_feat = 0
    for st_file in (Path(raw_dir) / module).glob("*.safetensors"):
        _, end = map(int, st_file.stem.split("_"))
        max_feat = max(max_feat, end)
    if max_feat != 0:
        latent_cfg.width = max_feat + 1

    if args.random_subset:
        torch.manual_seed(0)
        latents = torch.randperm(latent_cfg.width)[:n_latents]
    latent_dict = {f"{module}": latents}

    # example_constructor = partial(
    #     constructor,
    #     token_loader=lambda: dataset.load_tokens(),
    #     n_non_activating=experiment_cfg.n_non_activating,
    #     ctx_len=experiment_cfg.example_ctx_len,
    #     max_examples=latent_cfg.max_examples,
    # )
    # example_sampler = partial(sampler, cfg=experiment_cfg)
    dataset = LatentDataset(
        raw_dir=raw_dir,
        latent_cfg=latent_cfg,
        experiment_cfg=experiment_cfg,
        modules=[module],
        latents=latent_dict,
    )

    ### Load client ###

    # client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.8,max_model_len=5120)
    client = OpenRouter(
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        api_key="hey",
        base_url="http://localhost:8000/v1/chat/completions",
    )

    ### Build Explainer pipe ###
    def explainer_preprocess(record):
        explanation_path = (
            f"results/explanations/{sae_model}/{experiment_name}/{record.latent}.txt"
        )
        if os.path.exists(explanation_path):
            return ExplainerResult(
                record=record,
                explanation=orjson.loads(open(explanation_path, "rb").read()),
            )
        return record

    def explainer_postprocess(result):
        save_path = f"results/explanations/{sae_model}/{experiment_name}/{result.record.latent}.txt"
        with open(
            save_path,
            "wb",
        ) as f:
            f.write(orjson.dumps(result.explanation))

        return result

    # try making the directory if it doesn't exist
    os.makedirs(f"results/explanations/{sae_model}/{experiment_name}", exist_ok=True)

    explainer_pipe = process_wrapper(
        DefaultExplainer(
            client,
            threshold=0.3,
        ),
        preprocess=explainer_preprocess,
        postprocess=explainer_postprocess,
    )

    # save the experiment config
    with open(
        f"results/explanations/{sae_model}/{experiment_name}/experiment_config.json",
        "w",
    ) as f:
        print(experiment_cfg.to_dict())
        f.write(json.dumps(experiment_cfg.to_dict()))

    pipeline = Pipeline(
        dataset,
        explainer_pipe,
    )
    explanations = await pipeline.run(50)
    og_explanations = explanations.copy()
    to_delete = []
    if substitute != Substitution.NONE:
        group_size = ceil(latent_cfg.width**0.5)
        explanation_dict = {
            r.record.latent.latent_index: r.explanation for r in explanations
        }
        explained_features = list(explanation_dict.keys())
        explanation_list = list(explanation_dict.values())
        try:
            assert isinstance(embedding_model, SentenceTransformer)
            embeds = embedding_model.encode(explanation_list)
            similarity_matrix = embedding_model.similarity(embeds, embeds)
        except NameError:
            similarity_matrix = np.zeros((len(explanation_list), len(explanation_list)))
        for i, k in enumerate(explained_features):
            similarities_for_this_feature = defaultdict(list)
            for j, sim in enumerate(similarity_matrix[i]):
                if i == j:
                    continue
                sim = float(sim)
                v = explained_features[j]
                v_group = v // group_size
                similarities_for_this_feature[v_group].append((sim, j))
            most_similar = {k: max(v) for k, v in similarities_for_this_feature.items()}
            k_group = k // group_size
            if substitute == Substitution.SELF:
                try:
                    explanation = sorted(similarities_for_this_feature[k_group])[-2][1]
                except IndexError:
                    to_delete.append(i)
                    explanations[i] = replace(explanations[i], explanation="")
                    continue
            elif substitute == Substitution.OTHER:
                explanation = random.choice(list(most_similar.values()))[1]
            elif substitute == Substitution.SELF_RANDOM:
                try:
                    explanation = random.choice(
                        list(similarities_for_this_feature[k_group])
                    )[1]
                except IndexError:
                    to_delete.append(i)
                    explanations[i] = replace(explanations[i], explanation="")
                    continue
            elif substitute == Substitution.OTHER_RANDOM:
                explanation = random.choice(
                    [
                        feature_id
                        for group, features in similarities_for_this_feature.items()
                        for sim, feature_id in features
                        if group != k_group
                    ]
                )
            explanation = og_explanations[explanation].explanation
            print("Substituting", explanations[i].explanation, "with", explanation)
            explanations[i] = replace(explanations[i], explanation=explanation)

    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active

        return record

    def scorer_postprocess(result, score_dir):
        record = result.record
        with open(
            f"results/scores/{sae_model}/{experiment_name_scores}/{score_dir}/{record.latent}.txt",
            "wb",
        ) as f:
            f.write(orjson.dumps(result.score))

    os.makedirs(
        f"results/scores/{sae_model}/{experiment_name_scores}/detection", exist_ok=True
    )
    os.makedirs(
        f"results/scores/{sae_model}/{experiment_name_scores}/fuzz", exist_ok=True
    )

    # save the experiment config
    with open(
        f"results/scores/{sae_model}/{experiment_name_scores}/detection/experiment_config.json",
        "w",
    ) as f:
        f.write(json.dumps(experiment_cfg.to_dict()))

    with open(
        f"results/scores/{sae_model}/{experiment_name_scores}/fuzz/experiment_config.json",
        "w",
    ) as f:
        f.write(json.dumps(experiment_cfg.to_dict()))

    log_prob = False
    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(
                client,
                tokenizer=dataset.tokenizer,
                batch_size=shown_examples,
                verbose=False,
                log_prob=log_prob,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="detection"),
        ),
        process_wrapper(
            FuzzingScorer(
                client,
                tokenizer=dataset.tokenizer,
                batch_size=shown_examples,
                verbose=False,
                log_prob=log_prob,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="fuzz"),
        ),
    )

    ### Build the pipeline ###

    def iterate_explanations():
        for record in explanations:
            yield record

    pipeline = Pipeline(
        iterate_explanations,
        scorer_pipe,
    )
    start_time = time.time()
    await pipeline.run(50)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--shown_examples", type=int, default=5)
    parser.add_argument("--model", type=str, default="gemma/16k")
    parser.add_argument("--module", type=str, default=".model.layers.10")
    parser.add_argument("--latents", type=int, default=100)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--random_subset", action="store_true")
    parser.add_argument("--neighbors", action="store_true")
    parser.add_argument("--decoder_similarity", action="store_true")
    parser.add_argument("--substitute", type=Substitution, default=Substitution.NONE)
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(LatentConfig, dest="latent_options")
    args = parser.parse_args()
    asyncio.run(main(args))
