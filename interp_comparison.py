#%%
%env CUDA_VISIBLE_DEVICES=0
%load_ext autoreload
%autoreload 2
import json
from pathlib import Path
from functools import lru_cache
import numba as nb
import safetensors.numpy
from tqdm.auto import tqdm
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import numpy as np
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import random
from glob import glob
import os
import safetensors.numpy
from natsort import natsorted
from delphi.latents import LatentDataset
from delphi.config import LatentConfig, ExperimentConfig
sns.set_theme()
#%%
feature_type = "default_neighbors"
score_type = "detection"
layer = 15
all_descs = {}
all_accs = {}
for source in ("baseline", "pkm-h32"):
    feature_source = f"sae-pkm/{source}"
    prefix = f"results/explanations/{feature_source}/{feature_type}"
    feature_descs = {}
    feature_accs = {}
    for feature in glob(f"{prefix}/.model.layers.{layer}.mlp_latent*.txt"):
        score_path = Path("results/scores") / Path(feature).relative_to("results/explanations")
        score_path = Path(f"results/scores/{feature_source}/default") / score_path.relative_to(f"results/scores/{feature_source}/{feature_type}")
        score_path = score_path.parent / score_type / score_path.name
        score_json = json.loads(open(score_path).read())  
        corrects = []
        for text in score_json:
            if text["correct"] is None:
                continue
            corrects.append(int(text["correct"] == True))
        else:
            if corrects:
                score = sum(corrects) / len(corrects)
            else:
                score = 0
        feature_idx = int(feature.split("latent")[1][:-4])
        feature_accs[feature_idx] = score
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"PKM for SmolLM2 135M Layer {layer} {feature_source[8:]}"
    cache_dir = f"results/{feature_source}/.model.layers.{layer}.mlp"
    n_features = int(natsorted(glob(f"{cache_dir}/*.safetensors"))[-1][len(cache_dir) + 1:].rpartition("_")[2].partition(".")[0]) + 1
    
    all_descs[source] = feature_descs
    all_accs[source] = feature_accs
# %%
for source, feature_descs in all_descs.items():
    feature_accs = all_accs[source]
    bins = np.linspace(0, 1, 50)
    plt.hist(
        list(feature_accs.values()),
        bins=bins,
        alpha=0.5,
        label=source,
        density=True,
    )
plt.legend()
# %%
n_bins = 10
bins = np.linspace(0.6, 1, n_bins + 1)
all_bin_accs = {}
for bin_start, bin_end in zip(bins[:-1], bins[1:]):
    bin_accs = {}
    for source, feature_descs in all_descs.items():
        feature_accs = all_accs[source]
        matching_accs = [
            k for k, v in feature_accs.items() if bin_start <= v < bin_end
        ]
        bin_accs[source] = random.sample(matching_accs, 5)
        
    all_bin_accs[bin_start] = bin_accs
all_bin_accs
#%%
async def visualize_features(
    source: str,
    feature_descs: dict[int, str]
):
    feature_source = f"sae-pkm/{source}"
    raw_dir = f"results/{feature_source}"
    features = list(feature_descs.keys())
    ds = LatentDataset(
        raw_dir,
        LatentConfig(), ExperimentConfig(),
        modules=[f".model.layers.{layer}.mlp"],
        latents={f".model.layers.{layer}.mlp":
            # torch.tensor(list(feature_descs))
            torch.tensor(features)
        },
    )
    htmls = {}
    current_group = -1
    async for i in ds:
        html = []
        html.append(f"<h3>{feature_descs[i.latent.latent_index]}</h3>")
        html.append(i.display(ds.tokenizer, do_display=False, example_source="train"))
        htmls[i.latent.latent_index] = "\n".join(html)
        # i.display(ds.tokenizer,)
    result = []
    for k, v in sorted(htmls.items(), key=lambda x: x[0]):
        result.append(v)
    return "\n".join(result)

total = []
for bin_start, source_features in all_bin_accs.items():
    total.append(f"<h1>Bin {bin_start:.2f}</h1>")
    for source, features in source_features.items():
        total.append(f"<h2>{source}</h2>")
        descs = all_descs[source]
        descs = {k: descs[k] for k in features}
        html = await visualize_features(source, descs)
        total.append(html)
with open("results/comparison_visualization.html", "w") as f:
    f.write("\n".join(total))
#%%