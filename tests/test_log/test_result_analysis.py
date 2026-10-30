import json

import numpy as np
import pandas as pd
import torch

from delphi.log.result_analysis import compute_confusion, get_agg_metrics, load_data


def test_failed_count_counts_dropped_predictions():
    """
    compute_confusion must report how many examples the scorer failed to
    produce a (parseable) prediction for. Failed examples are stored with a
    None/NaN ``prediction`` by the classifier and are dropped by the
    ``prediction.notna()`` filter, so ``failed_count`` should equal the number
    of rows removed by that filter, i.e. ``len(df) - len(valid rows)``.
    """
    df = pd.DataFrame(
        {
            "activating": [True, False, True, False],
            # Two examples failed to be parsed -> NaN prediction.
            "prediction": [1.0, 0.0, np.nan, np.nan],
            "probability": [0.9, 0.1, np.nan, np.nan],
        }
    )

    conf = compute_confusion(df)

    assert conf["total_examples"] == 2
    assert conf["failed_count"] == 2

    # Mirror the "fraction of failed examples" computation in log_results().
    fraction_failed = conf["failed_count"] / (
        conf["total_examples"] + conf["failed_count"]
    )
    assert fraction_failed == 0.5


def test_failed_count_zero_when_all_predictions_valid():
    df = pd.DataFrame(
        {
            "activating": [True, False],
            "prediction": [1.0, 0.0],
            "probability": [0.9, 0.1],
        }
    )

    conf = compute_confusion(df)

    assert conf["total_examples"] == 2
    assert conf["failed_count"] == 0


def _example(activating: bool, prediction: bool) -> dict:
    return {
        "str_tokens": ["a"],
        "distance": 0,
        "activating": activating,
        "prediction": prediction,
        "probability": None,
        "correct": activating == prediction,
        "activations": [1.0],
    }


def _write_scores(tmp_path):
    """Detection scores for latent 0 of "layers.1" (all correct) and of
    "layers.10" (all wrong), whose name starts with "layers.1"."""
    scores = tmp_path / "scores" / "detection"
    scores.mkdir(parents=True)
    (tmp_path / "log").mkdir()
    (scores / "layers.1_latent0.txt").write_text(
        json.dumps([_example(True, True), _example(False, False)])
    )
    (scores / "layers.10_latent0.txt").write_text(
        json.dumps([_example(True, False), _example(False, True)])
    )
    counts = {"layers.1": torch.tensor([5, 3]), "layers.10": torch.tensor([7, 2])}
    torch.save(counts, tmp_path / "log" / "hookpoint_firing_counts.pt")
    return tmp_path / "scores"


def test_load_data_keeps_modules_with_shared_prefix_apart(tmp_path):
    latent_df, _ = load_data(_write_scores(tmp_path), ["layers.1", "layers.10"])

    assert latent_df.groupby("module").size().to_dict() == {
        "layers.1": 2,
        "layers.10": 2,
    }
    layers_1 = latent_df[latent_df["module"] == "layers.1"]
    assert (layers_1["activating"] == layers_1["prediction"]).all()
    # firing counts are looked up by latent index
    assert layers_1["firing_count"].tolist() == [5, 5]


def test_frequency_weighted_f1_with_several_modules(tmp_path):
    latent_df, counts = load_data(_write_scores(tmp_path), ["layers.1", "layers.10"])

    metrics = get_agg_metrics(latent_df, counts)

    # per-module F1 is 1.0 for "layers.1" and 0.0 for "layers.10"
    assert metrics["weighted_f1"].tolist() == [0.5]
