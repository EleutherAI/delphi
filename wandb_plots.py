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
        "displayName": {"$regex": r"sae-pkm"},
    },
)
runs = sorted(runs, key=lambda x: x.name)
print("Got", len(runs), "runs")
# %%
from matplotlib import pyplot as plt

types = "baseline", "pkm", "kron"
type_colors = "red", "green", "blue"
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
run: wandb.wandb_run.Run
