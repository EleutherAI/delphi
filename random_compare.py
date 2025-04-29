#%%
from pathlib import Path
import json

base_score_dir = Path("results/scores/sae-pkm")
base_config = "default_neighbors/detection"
# base_config = "default/detection"
layer = 20
for configuration in sorted(base_score_dir.glob("*")):
    config = configuration.name
    if config != "baseline":
        continue
    latent_path = configuration / base_config
    if not latent_path.exists():
        continue
    avgs = []
    for distance in 0, 1, 2, 3, 4:
        scores = []
        for latent in latent_path.glob(f".model.layers.{layer}.mlp_latent*.txt"):
            n_correct, n_total = 0, 0
            for prediction in json.load(open(latent)):
                if prediction["distance"] != distance:
                    continue
                n_correct += prediction["correct"] == True
                n_total += 1
            scores.append(n_correct / n_total)
        avg = sum(scores) / len(scores)
        avgs.append(avg)
    print(base_config, avgs)
    print(config, avg)
    break
#%%
from pathlib import Path
import shutil
import json

base_score_dir = Path("results/scores/sae-pkm")
for method in ("detection", "fuzz"):
    base_config = f"default_neighbors/{method}"
    for configuration in sorted(base_score_dir.glob("*")):
        config = configuration.name
        latent_path = configuration / base_config
        if not latent_path.exists():
            continue
        (configuration / "default").mkdir(exist_ok=True)
        shutil.move(latent_path,
                    configuration / f"default/{method}")