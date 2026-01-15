# Create a file named run_analysis.py with these contents
from pathlib import Path

from delphi.log.result_analysis import log_results

# Adjust the path to your results folder
scores_path = Path("results/pythia_100_test/scores")
viz_path = Path("results/pythia_100_test/visualize")
modules = ["layers.6.mlp"]
scorer_names = ["fuzz", "detection", "surprisal_intervention"]

log_results(
    scores_path, viz_path, modules, scorer_names, model_name="EleutherAI/pythia-160m"
)
