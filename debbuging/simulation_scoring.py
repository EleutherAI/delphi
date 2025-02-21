import asyncio
import os
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer

from delphi.clients import Offline
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.explainers import explanation_loader
from delphi.latents import LatentDataset
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import OpenAISimulator

parser = ArgumentParser()
parser.add_argument("--no_all_at_once", action="store_false", dest="all_at_once")

args = parser.parse_args()
all_at_once = args.all_at_once

feature_dict = {"layers.15.mlp": torch.arange(0, 10)}
dataset = LatentDataset(
    raw_dir="/mnt/ssd-1/gpaulo/SAE-Zoology/extras/end2end/results/smollm-sae-64x/latents",
    sampler_cfg=SamplerConfig(),
    constructor_cfg=ConstructorConfig(),
    modules=["layers.15.mlp"],
)


client = Offline(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_memory=0.5,
    max_model_len=5012,
    num_gpus=1,
    batch_size=5,
    prefix_caching=True,
)
SCORER_OUT_DIR = (
    "results/tests/all_at_once" if all_at_once else "results/tests/one_at_a_time"
)
### Set directories ###

EXPLAINER_OUT_DIR = (
    "/mnt/ssd-1/gpaulo/SAE-Zoology/extras/end2end/results/smollm-sae-64x/explanations"
)

### Build Explainer pipe ###
explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)

### Build Scorer pipe ###


def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    return record


def scorer_postprocess(result):
    with open(f"{SCORER_OUT_DIR}/{result.record.latent}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))


os.makedirs(f"{SCORER_OUT_DIR}", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
scorer_pipe = Pipe(
    process_wrapper(
        OpenAISimulator(client, all_at_once=all_at_once),
        preprocess=scorer_preprocess,
        postprocess=scorer_postprocess,
    )
)

### Build the pipeline ###

pipeline = Pipeline(
    dataset,
    explainer_pipe,
    scorer_pipe,
)

asyncio.run(pipeline.run(1))


experiment_name = args.experiment_name
