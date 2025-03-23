# %%
import re
from datetime import datetime

import wandb
import wandb.wandb_run

api = wandb.Api()
# %%

date_str = "March 7th, 2025 5:04:40 PM"
iso_date = datetime.strptime(date_str, "%B %dth, %Y %I:%M:%S %p").isoformat()
# %%
runs = api.runs(
    "eleutherai/optimized-encoder",
    filters={
        "createdAt": {"$gt": iso_date},
        "state": "finished",
        "displayName": {
            "$regex": r"sae-pkm",
        },
    },
)
runs = sorted(runs, key=lambda x: x.name)
runs = [run for run in runs if not run.name.endswith("baseline-x64")]
print("Got", len(runs), "runs")
# %%
from matplotlib import pyplot as plt

types = "baseline", "pkm", "kron"
type_colors = "red", "green", "blue"
run_info = {}
for layer in (10, 15, 20):
    for run in runs:
        name = run.name[len("sae-pkm/") :]
        fvu = run.summary[f"fvu/layers.{layer}.mlp"]
        dead_pct = run.summary[f"dead_pct/layers.{layer}.mlp"]
        k = run.config["sae"]["k"]
        expansion_factor = run.config["sae"]["expansion_factor"]
        for f in run.files():
            if f.name == "output.log":
                break
        else:
            raise FileNotFoundError
        output_log = f.download(root=f"out_logs/{name}", exist_ok=True).read()
        sae_params = re.match(
            r"Number of SAE parameters\: ([0-9_]+)\n", output_log
        ).group(1)
        sae_params = int(sae_params.replace("_", ""))
        color = type_colors[types.index(name.partition("-")[0])]
        x, y = sae_params, fvu
        run_info[name] = dict(
            sae_params=sae_params,
            fvu=fvu,
            dead_pct=dead_pct,
            k=k,
            expansion_factor=expansion_factor,
        )
        size = 100
        marker = "x" if "-k64" in name else "o"
        if "-x64" in name:
            size *= 2
        plt.scatter(x, y, s=size * 0.5, label=name, c=color, marker=marker)
        if "-h" in name:
            plt.scatter(x, y, edgecolors=color, s=size * 2, facecolors="none")
    plt.title(f"Layer {layer} MLP FVU")
    plt.xlabel("Parameter count")
    plt.ylabel("FVU")
    plt.legend()
    plt.show()
# %%
from functools import lru_cache


@lru_cache(maxsize=None)
def autointerp_scores(latent_path, layer):
    scores = []
    for latent in latent_path.glob(f".model.layers.{layer}.mlp_latent*.txt"):
        n_correct, n_total = 0, 0
        for prediction in json.load(open(latent)):
            n_correct += prediction["correct"] == True
            n_total += 1
        scores.append(n_correct / n_total)
    return scores


# %%
from pathlib import Path
import json

base_score_dir = Path("results/scores/sae-pkm")
base_config = "default_neighbors/detection"
layer = 10
configuration_scores = {}
x_axis = "fvu"
# x_axis = "dead"
for configuration in sorted(base_score_dir.glob("*")):
    config = configuration.name
    try:
        info = run_info[config]
    except KeyError:
        continue
    latent_path = configuration / base_config
    scores = autointerp_scores(latent_path, layer)
    if not scores:
        print("No latents found for", config)
        continue
    avg_score = sum(scores) / len(scores)
    if x_axis == "dead":
        x = info["dead_pct"]
    elif x_axis == "fvu":
        x = info["fvu"]
    else:
        raise ValueError
    y = avg_score
    color = type_colors[types.index(config.partition("-")[0])]
    size = 50
    marker = "x" if "-k64" in config else "o"
    if "-x64" in config:
        size *= 2
    if color == "blue" and "-long" in config:
        color = "cyan"
    plt.scatter(x, y, label=config, c=color, marker=marker, s=size)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), prop={"size": 12})
plt.xlabel("FVU" if x_axis == "fvu" else "% of dead neurons")
plt.ylabel("Average detection score")
# %%
set(run_info.keys()) - set(x.name for x in base_score_dir.glob("*"))
# %%
