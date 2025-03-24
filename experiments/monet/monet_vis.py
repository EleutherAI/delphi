# uv run python -m experiments.monet.cache
# uv run python -m examples.example_script --model monet --module .model.layers.12.router --features 20  --width 6143
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob
import json
import os


sns.set_theme()
for layer in (0, 4, 8, 12, 16, 20):
    for size in ("850m", "1.4b", "4.1b"):
        size_name = size

        def feature_accs(method):
            score_dir = (
                f"results/scores/monet_cache_converted/{size_name}/old/default/{method}"
            )
            feature_accs = []
            for s in os.listdir(score_dir):
                if not s.endswith(".txt"):
                    continue
                if f"layers.{layer}" not in s:
                    continue
                try:
                    data = json.load(open(os.path.join(score_dir, s)))
                except ValueError:
                    print("Error parsing", os.path.join(score_dir, s))
                    continue
                corrects = []
                for text in data:
                    if text["correct"] is None:
                        corrects.append(0)
                    else:
                        corrects.append(int(text["correct"]))
                feature_accs.append(sum(corrects) / len(corrects))
            if not feature_accs:
                raise FileNotFoundError
            return feature_accs

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        try:
            sns.histplot(feature_accs("fuzz"), bins=20, alpha=0.5, ax=axs[0])
            axs[0].set(
                xlabel="Accuracy",
                xlim=(0, 1),
                ylabel="Number of features",
                title="Fuzz",
            )
            sns.histplot(feature_accs("detection"), bins=20, alpha=0.5, ax=axs[1])
            axs[1].set(
                xlabel="Accuracy",
                xlim=(0, 1),
                ylabel="Number of features",
                title="Detect",
            )
        except FileNotFoundError:
            print(f"Skipping layer {layer} for size {size}")
            continue
        fig.suptitle(f"Monet {size.upper()} Layer {layer}, both routers")
        os.makedirs(f"results/monet{size_name}/autointerp", exist_ok=True)
        fig.savefig(f"results/monet{size_name}/autointerp/{layer}.png")
