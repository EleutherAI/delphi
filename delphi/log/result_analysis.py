from pathlib import Path

import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def import_plotly():
    """Import plotly with mitigation for MathJax bug."""
    try:
        import plotly.express as px  # type: ignore
        import plotly.io as pio  # type: ignore
    except ImportError:
        raise ImportError(
            "Plotly is not installed.\n"
            "Please install it using `pip install plotly`, "
            "or install the `[visualize]` extra."
        )
    pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
    return px


def compute_auc(df: pd.DataFrame) -> float | None:
    if not df.probability.nunique():
        return None

    valid_df = df[df.probability.notna()]

    return roc_auc_score(valid_df.activating, valid_df.probability)  # type: ignore


def plot_accuracy_hist(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    for label in df["score_type"].unique():
        fig = px.histogram(
            df[df["score_type"] == label],
            x="accuracy",
            nbins=100,
            title=f"Accuracy distribution: {label}",
        )
        fig.write_image(out_dir / f"{label}_accuracy.pdf")


def plot_roc_curve(df: pd.DataFrame, out_dir: Path):
    if not df.probability.nunique():
        return

    # filter out NANs
    valid_df = df[df.probability.notna()]

    fpr, tpr, _ = roc_curve(valid_df.activating, valid_df.probability)
    auc = roc_auc_score(valid_df.activating, valid_df.probability)
    fig = go.Figure(
        data=[
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"),
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")),
        ]
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="FPR",
        yaxis_title="TPR",
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.write_image(out_dir / "roc_curve.pdf")


def compute_confusion(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    df_valid = df[df["prediction"].notna()]
    total = len(df_valid)
    pos = df_valid.activating.sum()
    neg = total - pos

    tp = ((df_valid.prediction >= threshold) & df_valid.activating).sum()
    tn = ((df_valid.prediction < threshold) & ~df_valid.activating).sum()
    fp = ((df_valid.prediction >= threshold) & ~df_valid.activating).sum()
    fn = ((df_valid.prediction < threshold) & df_valid.activating).sum()
    return dict(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_examples=total,
        total_positives=pos,
        total_negatives=neg,
        failed_count=len(df_valid) - total,
    )


def compute_classification_metrics(conf: dict) -> dict:
    tp = conf["true_positives"]
    tn = conf["true_negatives"]
    fp = conf["false_positives"]
    fn = conf["false_negatives"]
    total = conf["total_examples"]
    pos = conf["total_positives"]
    neg = conf["total_negatives"]

    balanced_accuracy = (
        (tp / pos if pos > 0 else 0) + (tn / neg if neg > 0 else 0)
    ) / 2

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / pos if pos > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )
    # accuracy = (tp + tn) / total if total > 0 else 0
    return dict(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=balanced_accuracy,
        true_positive_rate=tp / pos if pos > 0 else 0,
        true_negative_rate=tn / neg if neg > 0 else 0,
        false_positive_rate=fp / neg if neg > 0 else 0,
        false_negative_rate=fn / pos if pos > 0 else 0,
        total_examples=total,
        total_positives=pos,
        total_negatives=neg,
        positive_class_ratio=pos / total if total > 0 else 0,
        negative_class_ratio=neg / total if total > 0 else 0,
    )


def load_data(scores_path: Path, modules: list[str]):
    """Load all on-disk data into a single DataFrame."""

    def parse_score_file(path: Path) -> pd.DataFrame:
        """
        Load a score file and return a raw DataFrame
        """
        data = orjson.loads(path.read_bytes())
        latent_idx = int(path.stem.split("latent")[-1])

        return pd.DataFrame(
            [
                {
                    "text": "".join(ex["str_tokens"]),
                    "distance": ex["distance"],
                    "activating": ex["activating"],
                    "prediction": ex["prediction"],
                    "probability": ex["probability"],
                    "correct": ex["correct"],
                    "activations": ex["activations"],
                    "latent_idx": latent_idx,
                }
                for ex in data
            ]
        )

    counts_file = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
    counts = torch.load(counts_file) if counts_file.exists() else {}

    # Collect per-latent data
    latent_dfs = []
    for score_type_dir in scores_path.iterdir():
        if not score_type_dir.is_dir():
            continue
        for module in modules:
            for file in score_type_dir.glob(f"*{module}*"):
                latent_idx = int(file.stem.split("latent")[-1])

                latent_df = parse_score_file(file)
                latent_df["score_type"] = score_type_dir.name
                latent_df["module"] = module
                latent_df["latent_idx"] = latent_idx
                latent_dfs.append(latent_df)

    return pd.concat(latent_dfs, ignore_index=True), counts


def get_metrics(latent_df: pd.DataFrame) -> pd.DataFrame:
    processed_rows = []
    for score_type, group_df in latent_df.groupby("score_type"):
        conf = compute_confusion(group_df)
        class_m = compute_classification_metrics(conf)
        auc = compute_auc(group_df)

        row = {"score_type": score_type, **conf, **class_m, "auc": auc}
        processed_rows.append(row)

    return pd.DataFrame(processed_rows)


def add_latent_f1(latent_df: pd.DataFrame) -> pd.DataFrame:
    f1s = latent_df.groupby(["module", "latent_idx"]).apply(
        lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
    )
    f1s.name = "f1_score"
    f1s = f1s.reset_index()
    return latent_df.merge(f1s, on=["module", "latent_idx"])


def log_results(
    scores_path: Path, viz_path: Path, modules: list[str], scorer_names: list[str]
):
    import_plotly()

    latent_df, counts = load_data(scores_path, modules)

    if latent_df.empty:
        print("No data found")
        return

    if counts:
        dead = sum((counts[m] == 0).sum().item() for m in modules)
        print(f"Number of dead features: {dead}")
        print(f"Number of interpreted live features: {len(latent_df)}")

        # Load constructor config for run
        with open(scores_path.parent / "run_config.json", "r") as f:
            run_cfg = orjson.loads(f.read())
        constructor_cfg = run_cfg.get("constructor_cfg", {})
        min_examples = constructor_cfg.get("min_examples", None)
        print("min examples", min_examples)

        if min_examples is not None:
            uninterpretable_features = sum(
                [(counts[m] < min_examples).sum() for m in modules]
            )
            print(
                f"Number of features below the interpretation firing"
                f" count threshold: {uninterpretable_features}"
            )

    plot_roc_curve(latent_df, viz_path)

    processed_df = get_metrics(latent_df)

    plot_accuracy_hist(processed_df, viz_path)

    for score_type in processed_df.score_type.unique():
        score_type_summary = processed_df[processed_df.score_type == score_type].iloc[0]
        print(f"\n--- {score_type.title()} Metrics ---")
        print(f"Balanced Accuracy: {score_type_summary['accuracy']:.3f}")
        print(f"F1 Score: {score_type_summary['f1_score']:.3f}")
        print(f"Precision: {score_type_summary['precision']:.3f}")
        print(f"Recall: {score_type_summary['recall']:.3f}")
        # Only print AUC if unbalanced AUC is not -1.
        if score_type_summary["auc"] is not None:
            print(f"AUC: {score_type_summary['auc']:.3f}")
        else:
            print("Logits not available.")

        fractions_failed = [
            score_type_summary["failed_count"]
            / (
                (
                    score_type_summary["total_examples"]
                    + score_type_summary["failed_count"]
                )
            )
        ]
        print(
            f"""Average fraction of failed examples: \
{sum(fractions_failed) / len(fractions_failed)}"""
        )

        print("\nConfusion Matrix:")
        print(
            f"True Positive Rate:  {score_type_summary['true_positive_rate']:.3f} "
            f"({score_type_summary['true_positives'].sum()})"
        )
        print(
            f"True Negative Rate:  {score_type_summary['true_negative_rate']:.3f} "
            f"({score_type_summary['true_negatives'].sum()})"
        )
        print(
            f"False Positive Rate: {score_type_summary['false_positive_rate']:.3f} "
            f"({score_type_summary['false_positives'].sum()})"
        )
        print(
            f"False Negative Rate: {score_type_summary['false_negative_rate']:.3f} "
            f"({score_type_summary['false_negatives'].sum()})"
        )

        print("\nClass Distribution:")
        print(f"""Positives: {score_type_summary['total_positives'].sum():.0f}""")
        print(f"""Negatives: {score_type_summary['total_negatives'].sum():.0f}""")
        print(f"Total: {score_type_summary['total_examples'].sum():.0f}")
