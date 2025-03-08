#%%
%env CUDA_VISIBLE_DEVICES=3
from glob import glob
feature_source, layer = "monet", "20"
# feature_source, layer = "pythia", "8"
# feature_source, layer = "smol", "9"

if feature_source == "monet":
    # model_size = "1.4b"
    model_size = "850m"
    prefix = f"../results/explanations/monet_cache_converted/{model_size}/default"
    n_features = 512 ** 2
    feature_descs = {}
    feature_name = "latent"
    feature_pattern = f"{prefix}/.model.layers.{layer}.router_{feature_name}*.txt"
    print(feature_pattern)
    for feature in glob(feature_pattern):
        feature_idx = int(feature.split(feature_name)[1][:-4])
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"Monet {model_size.upper()} Layer {layer}"
elif feature_source == "pythia":
    prefix = f"../results/explanations/sae_pkm/with_pkm_transcoder/default"
    n_features = 50_000
    feature_descs = {}
    for feature in glob(f"{prefix}/gpt_neox.layers.8_feature*.txt"):
        feature_idx = int(feature.split("feature")[1][:-4])
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"PKM for Pythia 160M Layer {layer}"
else:
    prefix = f"../results/explanations/sae_pkm/ef64-k64/default"
    n_features = 576 * 64
    feature_descs = {}
    for feature in glob(f"{prefix}/model.layers.{layer}_latent*.txt"):
        feature_idx = int(feature.split("latent")[1][:-4])
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"PKM for SmolLM2 135M Layer {layer}"
print("Found", len(feature_descs), "features")
# %%
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("NovaSearch/stella_en_400M_v5", trust_remote_code=True).cuda()
# st = SentenceTransformer("intfloat/e5-large-v2", trust_remote_code=True).cuda()
#%%
keys = list(feature_descs.values())
# pref = "passage: "
pref = ""
embeds = {k: v for k, v in
          zip(keys,
              st.encode([pref + k for k in keys]))}
#%%
from collections import defaultdict
import numpy as np
from math import ceil
idces = list(feature_descs.keys())
embed_array = [embeds[feature_descs[i]] for i in idces]
embed_array = np.array(embed_array)
sims = st.similarity(embed_array, embed_array).numpy()
root = int(ceil(n_features ** 0.5))
n_features_padded = root * root
print(root, n_features, n_features_padded)
idx_roots = np.floor(np.array(idces, dtype=np.float64) / float(root))
same_group = idx_roots[:, None] == idx_roots[None, :]
same_latent = np.eye(sims.shape[0], dtype=np.bool_)
def pad_group(i):
    a = np.zeros((n_features_padded, n_features_padded), dtype=i.dtype)
    idx = np.array(idces)
    idx0 = idx[:, None] + idx[None, :] * 0
    idx1 = idx[:, None] * 0 + idx[None, :]
    a[idx0, idx1] = i
    a = a.reshape(n_features_padded, root, root)
    return a
sims, same_group, same_latent = map(pad_group, (sims, same_group, same_latent))
#%%
not_same = sims * (~same_latent & ~same_group)
not_same = not_same[not_same != 0].ravel()
same = sims * (~same_latent & same_group)
same = same[same != 0].ravel()
bins = np.linspace(0, 1, 25)
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
plt.hist(same, bins=bins, alpha=0.5, label="Same group", density=True)
plt.hist(not_same, bins=bins, alpha=0.5, label="Not same group", density=True)
plt.ylabel("Density")
plt.xlabel("Cosine similarity")
plt.legend()
plt.title(f"{desc} feature description similarity, all pairs")
!mkdir -p ../results/pkm_cossim
plt.savefig(f"../results/pkm_cossim/{feature_source}_{layer}-all.svg")
plt.show()
#%%
k = 0
def take_k(arr, k):
    return -np.partition(-arr, k, axis=-1)[..., k]
not_same = sims * (~same_latent & ~same_group)
not_same = take_k(not_same, k)
n_samples = 16
# n_samples = 0
if n_samples == 0:
    not_same = not_same.sum(-1) / (not_same != 0).sum(-1)
else:
    index = np.random.randint(0, root, size=(n_features_padded, n_samples))
    not_same = np.take_along_axis(not_same, index, axis=-1)
not_same = not_same.flatten()
not_same = not_same[not_same != 0]
same = sims * (~same_latent & same_group)
same = take_k(same, k).flatten()
same = same[same != 0]
# not_same = (sims * (~same_latent & ~same_group)).topk(n_groups, dim=-1)[0][:, -1]
# same = (sims * (~same_latent & same_group)).max(-1)[0]
# same = same[same != 0]
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
bins = np.linspace(0, 1, 25)
plt.hist(same, bins=bins, alpha=0.5, label="Same group", density=True)
plt.hist(not_same, bins=bins, alpha=0.5, label="Not same group", density=True)
plt.ylabel("Density")
plt.xlabel("Cosine similarity")
plt.legend()
plt.title(f"{desc} feature description similarity, maximum for group")
plt.savefig(f"../results/pkm_cossim/{feature_source}_{layer}.svg")
plt.show()
# %%
