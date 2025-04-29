# %%
import re
from datetime import datetime
from pathlib import Path

import wandb
import wandb.wandb_run

api = wandb.Api()
# %%

date_str = "March 7th, 2025 5:04:40 PM"
iso_date = datetime.strptime(date_str, "%B %dth, %Y %I:%M:%S %p").isoformat()
# %%
from natsort import natsorted

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
runs = natsorted(runs, key=lambda x: (x.name, x.created_at))
runs = [run for run in runs if not run.name.endswith("baseline-x64")]
print("Got", len(runs), "runs")
# %%
plot_dir = Path("results/smollm_plots")
plot_dir.mkdir(exist_ok=True)
fvu_dir = plot_dir / "fvu"
fvu_dir.mkdir(exist_ok=True)
interp_dir = plot_dir / "interp"
interp_dir.mkdir(exist_ok=True)
# %%
from functools import lru_cache


@lru_cache(maxsize=None)
def get_history(run):
    return run.history()

# %%
from matplotlib import pyplot as plt
import random
import math

types = "baseline", "pkm", "kron"
# type_colors = "red", "green", "blue"
type_colors = ((1, 0.1, 0.3), (0.1, 0.6, 0.2), (0.1, 0.1, 1))
run_info = {}
run_names = {run.name[len("sae-pkm/") :] for run in runs}
for is_k64 in (False, True):
    runs_selected = [
        r for r in runs
        if ("-k64" in r.name) == is_k64
    ]
    tints = {}
    for layer in (10, 15, 20):
        plt.figure(figsize=(8, 6))
        coordinates = {}
        nudges = {}
        for run in runs_selected:
            name = run.name[len("sae-pkm/") :]
            history = get_history(run)
            fvu = history[f"fvu/layers.{layer}.mlp"]
            lowest_fvu = fvu.argmin()
            fvu = fvu[lowest_fvu]
            dead_pct = history[f"dead_pct/layers.{layer}.mlp"][lowest_fvu]
            k = run.config["sae"]["k"]
            expansion_factor = run.config["sae"]["expansion_factor"]
            for f in run.files():
                if f.name == "output.log":
                    break
            else:
                raise FileNotFoundError
            output_log = f.download(root=f"out_logs/{run.id}", exist_ok=True).read()
            sae_params = re.match(
                r"Number of SAE parameters\: ([0-9_]+)\n", output_log
            ).group(1)
            sae_params = int(sae_params.replace("_", ""))
            nudge = 1
            nudgable = ("-long" in name) or ((name + "-long") in run_names)
            if ("-long" in name):
                name = name[:-5]
            elif ((name + "-long") in run_names):
                continue
            # if ("-long" in name) or ((name + "-long") in run_names):
            if 0:
                if "-long" in name:
                    nudge_key = name
                else:
                    nudge_key = name + "-long"
                nudgable = True
                # if nudge_key in nudges:
                #     nudge = nudges[nudge_key]
                # else:
                #     nudge = random.uniform(0.98, 1.02)
                #     nudges[nudge_key] = nudge
            sae_params *= nudge
            decoder_params = run.config["sae"]["expansion_factor"] * 576
            encoder_params = sae_params - decoder_params
            # print(name, run, sae_params)
            type_name = name.partition("-")[0]
            color = type_colors[types.index(type_name)]
            x = encoder_params
            y = fvu
            # size = 100 * (1 + math.log(sae_params / 50e6))
            size = 80 * ((sae_params / 50e6) ** 2)
            # marker = "x" if "-k64" in name else "o"
            marker = "x" if "-x64" in name else "o"
            run_info[name] = dict(
                sae_params=sae_params,
                encoder_params=encoder_params,
                decoder_params=decoder_params,
                fvu=fvu,
                dead_pct=dead_pct,
                k=k,
                expansion_factor=expansion_factor,
                size=size,
                marker=marker,
            )
            # if "-x64" in name:
            #     size *= 2
            if type_name == "kron" and "-long" in name:
                color = (0.1, 0.7, 0.6)
            tint = 1.0
            # color = list(color)
            # if "-ig4" in name:
            #     color[0] = 1.0
            # if "-u1" in name:
            #     tint = 0.3
            # if "-u8" in name:
            #     tint = 0.1
            color = tuple(max(0, min(1, c * tint)) for c in color)
            if "-ig4" in name:
                x *= 1.015
            coordinates[name] = (x, y, size, color, marker)
        coordinates = dict(sorted(coordinates.items(), key=lambda x: x[1][0]))
        # for a, b in [
        #     ("kron-baseline", "kron-baseline-long"),
        #     ("kron-u2", "kron-u2-long"),
        #     ("kron-ig4", "kron-ig4-long"),
        #     ("kron-x64-k64", "kron-x64-k64-long"),
        #     ("kron-x64", "kron-x64-long"),
        # ]:
        #     a_x, a_y, *_ = coordinates[a]
        #     b_x, b_y, *_ = coordinates[b]
        #     plt.plot([a_x, b_x], [a_y, b_y], c="black", linestyle="--")
        for name, (x, y, size, color, marker) in coordinates.items():
            plt.scatter(x, y, s=size * 0.5, label=name, c=color, marker=marker)
            # if "-h" in name:
            #     plt.scatter(x, y, edgecolors=color, s=size * 2, facecolors="none")
        plt.title(f"Layer {layer} MLP FVU{' (K=64)' if is_k64 else ''}")
        plt.xlabel("Encoder parameter count")
        plt.xscale("log")
        plt.ylabel("FVU")
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), prop={"size": 11})
        plt.savefig(fvu_dir / f"layer_{layer}.png")
        plt.show()
        # break
#%%
coordinates.keys()
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
base_config = "default/detection"
for layer in (10, 15, 20):
    for x_axis in "dead", "fvu":
        plt.figure(figsize=(8, 6))
        configuration_scores = {}
        # x_axis = "dead"
        to_plot = []
        for configuration in natsorted(base_score_dir.glob("*")):
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
            size = info["size"]
            if color == "blue" and "-long" in config:
                color = "cyan"
            # marker = info["marker"]
            marker = "x" if "-k64" in config else "o"
            
            # plt.scatter(x, y, label=config, c=color, marker=info["marker"], s=size)
            to_plot.append((x, y, size, color, marker, config))
        for x, y, size, color, marker, config in sorted(to_plot):
            plt.scatter(x, y, label=config, c=color, marker=marker, s=size)
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), prop={"size": 15})
        plt.xlabel("FVU" if x_axis == "fvu" else "% of dead neurons")
        plt.ylabel("Average detection score")
        plt.savefig(interp_dir / f"layer_{layer}-x_{x_axis}.png")
        plt.title(f"Autointerp scores for layer {layer}")
        plt.show()
# %%btw
set(run_info.keys()) - set(x.name for x in base_score_dir.glob("*"))
# %%
