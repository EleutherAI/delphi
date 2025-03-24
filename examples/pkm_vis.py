import json
import os
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

"""
python -m examples.cache_saes --pkm=True
python -m examples.example_script --model sae_pkm/with_pkm_sae --module gpt_neox.layers.8 --latent 500
"""


def hist(x, *, ax, label):
    # sns.histplot(x, bins=20, alpha=0.5, ax=ax, label=label)
    sns.kdeplot(x, ax=ax, label=label, bw_adjust=0.75)


sns.set_theme()
pkm_score_dir = Path("results/scores/sae_pkm")
for group_name, config_group in {
    # ("with_pkm_sae", "without_pkm_sae"),
    # ("with_pkm_transcoder", "without_pkm_transcoder"),
    # ("baseline", "ef64-k64", "pkm-x32")
    "pkm-comparison": (
        ("baseline", "default"),
        ("ef64-k64", "default"),
        ("kron-baseline", "default"),
        ("pkm-x32", "default"),
    ),
    "pkm-comparison-neighbors": (
        ("baseline", "default_neighbors"),
        ("ef64-k64", "default_neighbors"),
        ("kron-baseline", "default_neighbors"),
        ("pkm-x32", "default_neighbors"),
    ),
    "pkm-comparison-neighbors-decoder": (
        ("baseline", "default_neighbors_ds"),
        ("ef64-k64", "default_neighbors_ds"),
        ("kron-baseline", "default_neighbors_ds"),
        ("pkm-x32", "default_neighbors_ds"),
    ),
    # "pkm-32-neighbors-substitution": (
    #     ("pkm-x32", "default_neighbors_substitute_self"),
    #     ("pkm-x32", "default_neighbors_substitute_other")
    # ),
    "monet-850m-substitution": (
        ("../monet_cache_converted/850m", "default_substitute_self"),
        ("../monet_cache_converted/850m", "default_substitute_other"),
    ),
    "monet-850m-substitution-random": (
        ("../monet_cache_converted/850m", "default_substitute_self_random"),
        ("../monet_cache_converted/850m", "default_substitute_other_random"),
    ),
    "pkm-32-substitution": (
        ("pkm-x32", "default_substitute_self"),
        ("pkm-x32", "default_substitute_other"),
    ),
    "pkm-32-substitution-random": (
        ("pkm-x32", "default_substitute_self_random"),
        ("pkm-x32", "default_substitute_other_random"),
    ),
    "ef64-k64-substitution": (
        ("ef64-k64", "default_substitute_self"),
        ("ef64-k64", "default_substitute_other"),
    ),
    "ef64-k64-substitution-random": (
        ("ef64-k64", "default_substitute_self_random"),
        ("ef64-k64", "default_substitute_other_random"),
    ),
}.items():
    for layer in range(24):
        fuzz_accs = {}
        detect_accs = {}
        for config_name, experiment_name in config_group:
            config_dir = pkm_score_dir / config_name
            config_name = f"{config_name}_{experiment_name}"

            def feature_accs(method):
                skipped = 0
                score_dir = config_dir / experiment_name / method
                feature_accs = []
                for s in os.listdir(score_dir):
                    if not s.endswith(".txt"):
                        continue
                    if f"layers.{layer}." not in s:
                        continue
                    try:
                        data = json.load(open(os.path.join(score_dir, s)))
                    except ValueError:
                        print("Error parsing", os.path.join(score_dir, s))
                        continue
                    corrects = []
                    for text in data:
                        if text["correct"] is None:
                            skipped += 1
                            continue
                        corrects.append(int(text["correct"] == True))
                    if not corrects:
                        continue
                    feature_accs.append(sum(corrects) / len(corrects))
                if not feature_accs:
                    raise FileNotFoundError
                print(f"Skipped {skipped} for {config_name}")
                return feature_accs

            try:
                fuzz_accs[config_name] = feature_accs("fuzz")
                detect_accs[config_name] = feature_accs("detection")
            except FileNotFoundError as e:
                print(f"Skipping layer {layer} for config {group_name}:", e)
                break
        else:
            save_dir = f"results/pkm_autointerp/{group_name}"
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].set(
                xlabel="Accuracy",
                xlim=(0, 1),
                ylabel="Number of features",
                title="Fuzz",
            )
            axs[1].set(
                xlabel="Accuracy",
                xlim=(0, 1),
                ylabel="Number of features",
                title="Detect",
            )
            print("Visalizing", len(fuzz_accs[config_name]), "features into", save_dir)
            for config_name, experiment_name in config_group:
                config_name = f"{config_name}_{experiment_name}"
                hist(fuzz_accs[config_name], ax=axs[0], label=config_name)
                hist(detect_accs[config_name], ax=axs[1], label=config_name)

            axs[0].legend()
            fig.suptitle(f"{group_name} Layer {layer}, both routers")
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f"{save_dir}/{layer}.svg")
