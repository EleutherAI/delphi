#%%
%env SPARSIFY_DISABLE_TRITON=1
%env CUDA_VISIBLE_DEVICES=0
%load_ext autoreload
%autoreload 2
from sparsify import SparseCoder
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
torch.set_grad_enabled(False)
sns.set_theme()
#%%
# %env CUDA_VISIBLE_DEVICES=1
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
st = SentenceTransformer("NovaSearch/stella_en_400M_v5", trust_remote_code=True).cuda()
# st = SentenceTransformer("intfloat/e5-large-v2", trust_remote_code=True).cuda()
#%%
for feature_source, layer in [
    ("monet", "8"),
    ("monet", "12"),
    ("monet", "16"),
    ("monet", "20"),
    ("pythia", "8"),
    ("smol", "9"),
]:
    pass
#%%
# feature_source, layer = "sae-pkm/baseline", 10
# feature_source, layer = "sae-pkm/pkm-h16", 10
# feature_source, layer = "sae-pkm/pkm-baseline", 15
# feature_source, layer = "sae-pkm/kron-baseline", 10
# feature_source, layer = "sae-pkm/pkm-h8", 10
feature_source, layer = "monet", "16"

# feature_source, layer = "pythia", "8"
# feature_source, layer = "smol", "9"

if feature_source == "monet":
    # model_size = "1.4b"
    model_size = "850m"
    prefix = f"results/explanations/monet_cache_converted/{model_size}/default_neighbors"
    n_features = 512**2
    feature_descs = {}
    for feature_name in ["feature", "latent"]:
        feature_pattern = f"{prefix}/.model.layers.{layer}.router_{feature_name}*.txt"
        print(feature_pattern)
        for feature in glob(feature_pattern):
            feature_idx = int(feature.split(feature_name)[1][:-4])
            feature_descs[feature_idx] = open(feature).read()[1:-1]
        desc = f"Monet {model_size.upper()} Layer {layer}"
        if feature_descs:
            break
    cache_dir = f"results/monet_cache_converted/{model_size}/.model.layers.{layer}.router"
elif feature_source == "pythia":
    prefix = f"results/explanations/sae_pkm/with_pkm_transcoder/default"
    n_features = 50_000
    feature_descs = {}
    for feature in glob(f"{prefix}/gpt_neox.layers.8_feature*.txt"):
        feature_idx = int(feature.split("feature")[1][:-4])
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"PKM for Pythia 160M Layer {layer}"
elif feature_source.startswith("sae-pkm"):
    prefix = f"results/explanations/{feature_source}/default_neighbors"
    feature_descs = {}
    for feature in glob(f"{prefix}/.model.layers.{layer}.mlp_latent*.txt"):
        feature_idx = int(feature.split("latent")[1][:-4])
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"SmolLM2 135M Layer {layer} {feature_source[8:]}"
    cache_dir = f"results/{feature_source}/.model.layers.{layer}.mlp"
    n_features = int(natsorted(glob(f"{cache_dir}/*.safetensors"))[-1][len(cache_dir) + 1:].rpartition("_")[2].partition(".")[0]) + 1
else:
    prefix = f"results/explanations/sae_pkm/ef64-k64/default"
    n_features = 576 * 64
    feature_descs = {}
    for feature in glob(f"{prefix}/model.layers.{layer}_latent*.txt"):
        feature_idx = int(feature.split("latent")[1][:-4])
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"PKM for SmolLM2 135M Layer {layer}"
print("Found", len(feature_descs), "features")
combined_locations = []
combined_activations = []
for st_file in tqdm(natsorted(glob(f"{cache_dir}/*.safetensors"))):
    start = int(os.path.basename(st_file).partition("_")[0])
    safet = safetensors.numpy.load_file(st_file)
    tokens = safet["tokens"]
    n_examples, seq_len = safet["tokens"].shape
    safet["locations"][:, 2] += start
    combined_locations.append(safet["locations"])
    combined_activations.append(safet["activations"])
combined_locations = np.concatenate(combined_locations)
combined_activations = np.concatenate(combined_activations)
print("Sorting...")
sort_indices = np.argsort(combined_locations[:, 2])
locations = combined_locations[sort_indices]
activations = combined_activations[sort_indices]
#%%
kron_separate = False
if "kron" in feature_source:
    root = n_features // 4
    if "-og8" in feature_source:
        root = n_features // 8
    if kron_separate:
        feature_descs = {
            (
                k // root + (k % root) * (n_features // root)
            ): v for k, v in feature_descs.items()
        }
        combined_locations[:, 2] = \
            combined_locations[:, 2] // root + (combined_locations[:, 2] % root) * (n_features // root)
        root = n_features // root
else:
    root = int(ceil(n_features ** 0.5))
n_features_padded = n_features // root * root
print("Done")
#%%
if "sae-pkm" in feature_source and False:
    raw_dir = f"results/{feature_source}"
    ds = LatentDataset(
        raw_dir,
        LatentConfig(), ExperimentConfig(),
        modules=[f".model.layers.{layer}.mlp"],
        latents={f".model.layers.{layer}.mlp": torch.tensor(list(feature_descs))},
    )
    htmls = []
    current_group = -1
    async for i in ds:
        if i.latent.latent_index // root != current_group:
            current_group = i.latent.latent_index // root
            htmls.append(f"<h1>Group {current_group}</h1>")

        # print(i.latent)
        # print("Explanation:", feature_descs[i.latent.latent_index])
        htmls.append(f"<h2>{feature_descs[i.latent.latent_index]}</h2>")
        htmls.append(i.display(ds.tokenizer, do_display=False, example_source="train"))
        # i.display(ds.tokenizer,)
#%%
if "sae-pkm" in feature_source and False:
    os.makedirs(f"results/group_htmls/{feature_source}", exist_ok=True)
    with open(f"results/group_htmls/{feature_source}_{layer}.html", "w") as f:
        f.write("\n".join(htmls))
#%%
if "sae-pkm" in feature_source:
    weight_dir = f"../e2e/{feature_source}/layers.{layer}.mlp/sae.safetensors"
    weights = safetensors.numpy.load_file(weight_dir)
    W_dec = weights["W_dec"]
    decs = W_dec[np.array(list(feature_descs.keys()))]
    decs = decs / np.linalg.norm(decs, axis=1, keepdims=True)
    dec_sims = np.dot(decs, decs.T)
    same_flat = []
    not_same_flat = []
    for i, f1 in enumerate(feature_descs):
        for j, f2 in enumerate(feature_descs):
            if i >= j:
                continue
            same = f1 // root == f2 // root
            if same:
                same_flat.append(dec_sims[i, j])
            else:
                not_same_flat.append(dec_sims[i, j])
    bins = np.logspace(-2, 0, 25)
    plt.hist(same_flat, bins=bins, alpha=0.5, label="Same group", density=True)
    plt.hist(not_same_flat, bins=bins, alpha=0.5, label="Not same group", density=True)
    plt.xscale("log")
    plt.ylabel("Density")
    plt.xlabel("Cosine similarity")
    plt.legend()
    plt.title(f"{desc} decoder similarity, all pairs")
    plt.savefig(f"results/pkm_cossim/{feature_source}_{layer}-decoder.svg")
    plt.show()
#%%
keys = list(feature_descs.values())
# pref = "passage: "
pref = ""
embeds = {k: v for k, v in
          zip(keys,
              st.encode([pref + k for k in keys]))}
idces = list(feature_descs.keys())
embed_array = [embeds[feature_descs[i]] for i in idces]
embed_array = np.array(embed_array)
sims = st.similarity(embed_array, embed_array).numpy()
same_flat = []
not_same_flat = []
for i, f1 in enumerate(idces):
    for j, f2 in enumerate(idces):
        if i >= j:
            continue
        same = f1 // root == f2 // root
        if same:
            same_flat.append(sims[i, j])
        else:
            not_same_flat.append(sims[i, j])
bins = np.linspace(0, 1, 25)
plt.hist(same_flat, bins=bins, alpha=0.5, label="Same group", density=True)
plt.hist(not_same_flat, bins=bins, alpha=0.5, label="Not same group", density=True)
plt.ylabel("Density")
plt.xlabel("Cosine similarity")
plt.legend()
plt.title(f"{desc} feature description similarity, all pairs")
# !mkdir -p results/pkm_cossim
os.makedirs("results/pkm_cossim", exist_ok=True)
os.makedirs("results/pkm_cossim/sae-pkm", exist_ok=True)
plt.savefig(f"results/pkm_cossim/{feature_source}_{layer}-all.svg")
plt.show()
#%%
best_same_group = []
best_random_group = []
for i, f1 in enumerate(idces):
    best_sims = {}
    for j, f2 in enumerate(idces):
        if i == j:
            continue
        key = f2 // root
        best_sims[key] = max(best_sims.get(key, 0), sims[i, j])
    key = f1 // root
    if key in best_sims:
        best_same_group.append(best_sims[key])
    best_random_group.append(random.choice(
        [v for k, v in best_sims.items() if k != key]))
sns.set_theme()
bins = np.linspace(0, 1, 25)
plt.hist(best_same_group, bins=bins, alpha=0.5, label="Same group", density=True)
plt.hist(best_random_group, bins=bins, alpha=0.5, label="Not same group", density=True)
plt.ylabel("Density")
plt.xlabel("Cosine similarity")
plt.legend()
plt.title(f"{desc} feature description similarity, maximum for group")
plt.savefig(f"results/pkm_cossim/{feature_source}_{layer}.svg")
plt.show()
#%%
if 1:
    # random.seed(9900)
    random.seed(9789)
    random_feature = random.choice(list(feature_descs.keys()))
    similarities_for_random = []
    similarities_self = []
    similarities_other = []
    explanations_self, explanations_other = [], []
    explanations_for_similar = []
    for i, f in enumerate(idces):
        if f == random_feature:
            continue
        sim = sims[i, idces.index(random_feature)]
        expl = feature_descs[f]
        similarities_for_random.append(sim)
        if f // root == random_feature // root:
            similarities_self.append(sim)
            explanations_self.append(expl)
        else:
            similarities_other.append(sim)
            explanations_other.append(expl)
        explanations_for_similar.append(expl)
    bins = np.linspace(0, 1, 25)
    # plt.hist(similarities_for_random, bins=bins, alpha=0.5, density=True)
    plt.hist(similarities_self, bins=bins, alpha=0.5, label="Same group", density=True)
    plt.hist(similarities_other, bins=bins, alpha=0.5, label="Different group", density=True)
    plt.legend()
    plt.show()
    print("Random feature:", feature_descs[random_feature])
    print("Same group:")
    for sim, expl in sorted(zip(similarities_self, explanations_self), reverse=True)[:5]:
        print(sim, expl)
    print("Other groups:")
    for sim, expl in sorted(zip(similarities_other, explanations_other), reverse=True)[:5]:
        print(sim, expl)
#%%
@lru_cache(maxsize=100_000)
@nb.njit
def locations_for_feature(feature_idx):
    start = np.searchsorted(locations[:, 2], feature_idx, side="left")
    end = np.searchsorted(locations[:, 2], feature_idx, side="right")
    return start, end + 1
@nb.njit
def flatten_ctx(ctx):
    return ctx[:, 0] * seq_len + ctx[:, 1]
@lru_cache(maxsize=100_000)
def find_intersections(f1, f2):
    s1, e1 = locations_for_feature(f1)
    s2, e2 = locations_for_feature(f2)
    l1 = flatten_ctx(locations[s1:e1, :2])
    l2 = flatten_ctx(locations[s2:e2, :2])
    if l1.size == 0 or l2.size == 0:
        return -1
    common, idx_a, idx_b = np.intersect1d(l1, l2, return_indices=True)
    return common.size / (l1.size + l2.size - common.size)
#%%
jaccards_same = []
jaccards_diff = []
for f1 in tqdm(feature_descs):
    for f2 in feature_descs:
        if f2 >= f1:
            continue
        iou = find_intersections(f1, f2)
        if f1 // root == f2 // root:
            jaccards_same.append(iou)
        else:
            jaccards_diff.append(iou)
#%%
bins = np.logspace(-7, 1, 50)
plt.xscale("log")
plt.yscale("log")
plt.hist(jaccards_same, bins=bins, alpha=0.5, label="Same group", density=True)
plt.hist(jaccards_diff, bins=bins, alpha=0.5, label="Different groups", density=True)
plt.ylabel("Density")
plt.xlabel("Jaccard similarity")
plt.legend()
plt.title(f"{desc} co-occurrence, all pairs")
plt.savefig(f"results/pkm_cossim/{feature_source}_{layer}-iou.svg")
plt.show()
# %%
if "sae-pkm" in feature_source:
    # scores_name = Path("results/scores") / feature_source / "default/detection"
    # print(scores_name)
    # score_jsons = scores_name.glob(f".model.layers.{layer}.mlp_latent*.txt")
    # is_dead = []
    # for score_json in score_jsons:
    #     data = json.load(open(score_json))
    #     this_dead = True
    #     for d in data:
    #         if d["correct"] is not None:
    #             this_dead = False
    #             break
    #     is_dead.append(this_dead)
    # print("Dead ratio:", sum(is_dead) / len(is_dead))

    # raw_dir = f"results/{feature_source}"
    # ds = LatentDataset(
    #     raw_dir,
    #     LatentConfig(), ExperimentConfig(),
    #     modules=[f".model.layers.{layer}.mlp"],
    #     latents={f".model.layers.{layer}.mlp": torch.tensor(list(feature_descs))},
    # )
    # async for i in ds:
    #     print(len(i.examples))
    dead_mask = []
    for feat in tqdm(feature_descs):
        dead_mask.append(int((combined_locations[:, 2] == feat).sum() != 0))
#%%
# dead[10000:10100]
#%%
if "sae-pkm" in feature_source:
    dead_mask = np.array(dead_mask)
    feature_indices = np.array(list(feature_descs.keys()))
    dead = np.zeros((n_features_padded,))
    dead[feature_indices] = dead_mask
    # group_idces1 = feature_indices // root
    # group_idces2 = feature_indices % root
    # arr1 = np.zeros((n_features // root,))
    # arr2 =
    
    # plt.plot(dead.reshape(-1, root).sum(axis=0))
    # plt.show()
    # plt.plot(dead.reshape(-1, root).sum(axis=1))
    # plt.show()
    plt.plot(dead)
    plt.show()
# %%
b = weights["encoder.bias"]
b = b.reshape(-1, root).mean(axis=0)
plt.plot(b)
#%%
coder = SparseCoder.load_from_disk(os.path.dirname(weight_dir))
#%%
encoder = coder.encoder
decoded = encoder.pre.bias @ encoder.weight.T
grouped = decoded
plt.plot(grouped.detach().cpu().numpy())
plt.ylabel("Projection from pre-encoder bias to feature")
plt.xlabel("Feature")
#%%
@nb.njit
def get_mean(combined_locations, combined_activations, w_dec):
    result = np.zeros((w_dec.shape[1],))
    # max_batch, max_seq = np.max(combined_locations, axis=0)[:2]
    max_batch = np.max(combined_locations[:, 0])
    max_seq = np.max(combined_locations[:, 1])
    n_tokens = max(0, max_batch-1) * max_seq + combined_locations[-1, 1]
    for i in range(combined_locations.shape[0]):
        result += w_dec[combined_locations[i, 2]] * combined_activations[i] / n_tokens
    return result
data_mean = get_mean(
    combined_locations[:10000],
    combined_activations.astype(np.float32),
    weights["W_dec"]
) + weights["b_dec"]
#%%
encoder = coder.encoder
decoded = encoder(torch.tensor(data_mean).to(encoder.pre.weight))
grouped = decoded.reshape(-1, root)
# bins = np.linspace(-1, 1, 50) * 1e3
bins = np.linspace(-1, 1, 50) * 1e5
for i, g in enumerate(grouped):
    # plt.hist(g.detach().cpu().numpy(), bins=bins, alpha=0.4, label=i)
    sns.kdeplot(
        g.detach().cpu().numpy(),
        # bins=bins,
        alpha=0.4,
        label=i,
        bw_adjust=0.5,
    )
plt.legend()
# plt.plot(grouped.detach().cpu().numpy())
# plt.ylabel("Projection from pre-decoder and pre-encoder biases to feature")
# plt.xlabel("Feature")
#%%
@nb.njit
def decode(combined_locations, combined_activations, w_dec):
    max_batch = np.max(combined_locations[:, 0])
    max_seq = np.max(combined_locations[:, 1])
    result = np.zeros((max_batch, w_dec.shape[1],))
    
    n_tokens = max(0, max_batch-1) * max_seq + combined_locations[-1, 1]
    for i in range(combined_locations.shape[0]):
        batch_idx = combined_locations[i, 0]
        if batch_idx >= max_batch:
            continue
        result[batch_idx] += w_dec[combined_locations[i, 2]] * combined_activations[i] / n_tokens
    return result

decoded = decode(
    combined_locations[:1000000],
    combined_activations.astype(np.float32),
    weights["W_dec"]
) + weights["b_dec"]
#%%
_, indices = coder.encoder.topk(torch.tensor(decoded).to(encoder.pre.weight), k=coder.cfg.k)
uniques = torch.unique(indices.flatten())
uniques = uniques.cpu().numpy()
not_dead = np.zeros((n_features_padded,))
not_dead[uniques] = 1
# plt.plot(not_dead.reshape(-1, root).sum(axis=0))
plt.plot(not_dead)
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
config = json.loads((Path(os.path.dirname(os.path.dirname(weight_dir))) / "config.json").read_text())
model = AutoModelForCausalLM.from_pretrained(config["model"], device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(config["model"])
#%%
from collections import OrderedDict
for m in model.modules():
    m._forward_hooks = OrderedDict()
def save_activations(module, input, output):
    global activations
    if isinstance(input, tuple):
        input = input[0]
    activations = input
model.model.layers[layer].mlp.register_forward_hook(save_activations)
model(torch.from_numpy(tokens[:16]).cuda());
#%%
og_state_dict = {k: v.clone() for k, v in coder.state_dict().items()}
#%%
def vis_dead():
    global encoded
    vals = activations.flatten(0, 1).float().cpu()
    encoded = coder.encoder(vals)
    out = coder(vals)
    indices = out.latent_indices
    uniques = torch.unique(indices.flatten())
    uniques = uniques.cpu().numpy()
    not_dead = np.zeros((n_features_padded,))
    not_dead[uniques] = 1
    # plt.plot(not_dead.reshape(-1, root).sum(axis=0))
    # plt.plot(not_dead, label=f"Not dead: {not_dead.mean():.2f}")
    # sns.kdeplot(uniques, label=f"Not dead: {not_dead.mean():.2f}", bw_adjust=0.1)
    sns.displot(
        uniques,
        label=f"Not dead: {not_dead.mean():.2f}",
        kind="kde",
        norm_hist=False,
    )
    # sns.histplot(
    #     uniques,
    #     # bins=100,
    #     label=f"Not dead: {not_dead.mean():.2f}",
    #     # stat="density",
    #     alpha=0.5,
    #     kde=True,
    #     bins=n_features_padded,
    # )
    plt.legend()

coder.load_state_dict(og_state_dict, strict=False)
for u in range(coder.encoder.inner.shape[1]):
    coder.encoder.inner.data = coder.encoder.inner.data.clone()
    coder.encoder.inner.data[:, u] = 0
    vis_dead()
    plt.title(f"{u} group")
    plt.show()
    coder.load_state_dict(og_state_dict, strict=False)
    
# %%
vis_dead()
#%%
activations_ = activations
activations = activations.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)
vis_dead()
activations = activations_
plt.show()
#%%
plt.plot(encoded[0])
# %%
tokenizer.batch_decode(tokens[:16])
# %%