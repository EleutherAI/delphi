from pathlib import Path
from typing import Optional

import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def plot_firing_vs_f1(
    latent_df: pd.DataFrame, num_tokens: int, out_dir: Path, run_label: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for module, module_df in latent_df.groupby("module"):

        if 'firing_count' not in module_df.columns:
            print(f"""WARNING:'firing_count' column not found for module {module}. 
                    Skipping plot.""")
            continue

        module_df = module_df.copy()
        # Filter out rows where f1_score is NaN to avoid errors in plotting
        module_df = module_df[module_df['f1_score'].notna()]
        if module_df.empty:
            continue

        module_df["firing_rate"] = module_df["firing_count"] / num_tokens
        fig = px.scatter(module_df, x="firing_rate", y="f1_score", log_x=True)
        fig.update_layout(
            xaxis_title="Firing rate", yaxis_title="F1 score", xaxis_range=[-5.4, 0]
        )
        fig.write_image(out_dir / f"{run_label}_{module}_firing_rates.pdf")


def import_plotly():
    """Import plotly with mitigiation for MathJax bug."""
    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError:
        raise ImportError(
            "Plotly is not installed.\n"
            "Please install it using `pip install plotly`, "
            "or install the `[visualize]` extra."
        )
    pio.kaleido.scope.mathjax = None
    return px


def compute_auc(df: pd.DataFrame) -> float | None:

    valid_df = df[df.probability.notna()]
    if valid_df.probability.nunique() <= 1:
        return None
    return roc_auc_score(valid_df.activating, valid_df.probability)


def plot_accuracy_hist(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    for label in df["score_type"].unique():
        # Filter out surprisal_intervention as 'accuracy' is not relevant for it
        if label == 'surprisal_intervention':
            continue
        fig = px.histogram(
            df[df["score_type"] == label],
            x="accuracy",
            nbins=100,
            title=f"Accuracy distribution: {label}",
        )
        fig.write_image(out_dir / f"{label}_accuracy.pdf")


def plot_roc_curve(df: pd.DataFrame, out_dir: Path):

    valid_df = df[df.probability.notna()]
    if valid_df.empty or valid_df.activating.nunique() <= 1 or valid_df.probability.nunique() <= 1:
        return

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
    if df_valid.empty:
        return dict(true_positives=0, true_negatives=0, false_positives=0, false_negatives=0,
                    total_examples=0, total_positives=0, total_negatives=0, failed_count=len(df))

    act = df_valid["activating"].astype(bool)
    total = len(df_valid)
    pos = act.sum()
    neg = total - pos
    tp = ((df_valid.prediction >= threshold) & act).sum()
    tn = ((df_valid.prediction < threshold) & ~act).sum()
    fp = ((df_valid.prediction >= threshold) & ~act).sum()
    fn = ((df_valid.prediction < threshold) & act).sum()

    return dict(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn,
        total_examples=total, total_positives=pos, total_negatives=neg,
        failed_count=len(df) - len(df_valid),
    )


def compute_classification_metrics(conf: dict) -> dict:
    tp, tn, fp, fn = conf["true_positives"], conf["true_negatives"], conf["false_positives"], conf["false_negatives"]
    pos, neg = conf["total_positives"], conf["total_negatives"]
    
    balanced_accuracy = ((tp / pos if pos > 0 else 0) + (tn / neg if neg > 0 else 0)) / 2
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / pos if pos > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return dict(
        precision=precision, recall=recall, f1_score=f1, accuracy=balanced_accuracy,
        true_positive_rate=tp / pos if pos > 0 else 0,
        true_negative_rate=tn / neg if neg > 0 else 0,
        false_positive_rate=fp / neg if neg > 0 else 0,
        false_negative_rate=fn / pos if pos > 0 else 0,
    )


def load_data(scores_path: Path, modules: list[str]):
    """Load all on-disk data into a single DataFrame."""

    def parse_score_file(path: Path) -> pd.DataFrame:
        try:
            data = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            print(f"Error decoding JSON from {path}. Skipping file.")
            return pd.DataFrame()
        
        if not isinstance(data, list):
            print(f"""Warning: Expected a list of results in {path}, but found {type(data)}. 
                    Skipping file.""")
            return pd.DataFrame()

        latent_idx = int(path.stem.split("latent")[-1])

        # Updated to extract all possible keys safely using .get()
        return pd.DataFrame(
            [
                {
                    "text": "".join(ex.get("str_tokens", [])),
                    "distance": ex.get("distance"),
                    "activating": ex.get("activating"),
                    "prediction": ex.get("prediction"),
                    "probability": ex.get("probability"),
                    "correct": ex.get("correct"),
                    "activations": ex.get("activations"),
                    "final_score": ex.get("final_score"),
                    "avg_kl_divergence": ex.get("avg_kl_divergence"),
                    "latent_idx": latent_idx,
                }
                for ex in data
            ]
        )

    counts_file = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
    counts = torch.load(counts_file, weights_only=True) if counts_file.exists() else {}
    if not all(module in counts for module in modules):
        print("Missing firing counts for some modules, setting counts to None.")
        print(f"Missing modules: {[m for m in modules if m not in counts]}")
        counts = None

    latent_dfs = []
    for score_type_dir in scores_path.iterdir():
        if not score_type_dir.is_dir():
            continue
        for module in modules:
            for file in score_type_dir.glob(f"*{module}*"):
                latent_df = parse_score_file(file)
                if latent_df.empty:
                    continue
                latent_df["score_type"] = score_type_dir.name
                latent_df["module"] = module
                if counts:
                    latent_idx = latent_df["latent_idx"].iloc[0]
                    latent_df["firing_count"] = (
                        counts[module][latent_idx].item()
                        if module in counts and latent_idx in counts[module]
                        else None
                    )
                latent_dfs.append(latent_df)

    if not latent_dfs:
        return pd.DataFrame(), counts
        
    return pd.concat(latent_dfs, ignore_index=True), counts


def get_agg_metrics(
    latent_df: pd.DataFrame, counts: Optional[dict[str, torch.Tensor]]
) -> pd.DataFrame:
    processed_rows = []
    for score_type, group_df in latent_df.groupby("score_type"):
        # For surprisal_intervention, we don't compute classification metrics
        if score_type == 'surprisal_intervention':
            continue
            
        conf = compute_confusion(group_df)
        class_m = compute_classification_metrics(conf)
        auc = compute_auc(group_df)
        f1_w = frequency_weighted_f1(group_df, counts) if counts else None
        
        row = {
            "score_type": score_type,
            **conf, **class_m, "auc": auc, "weighted_f1": f1_w
        }
        processed_rows.append(row)

    return pd.DataFrame(processed_rows)


def add_latent_f1(latent_df: pd.DataFrame) -> pd.DataFrame:
    f1s = (
        latent_df.groupby(["module", "latent_idx"])
        .apply(
            lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
        )
        .reset_index(name="f1_score")  # <- naive (un-weighted) F1
    )
    return latent_df.merge(f1s, on=["module", "latent_idx"])


def log_results(
    scores_path: Path, viz_path: Path, modules: list[str], scorer_names: list[str]
):
    import_plotly()

    latent_df, counts = load_data(scores_path, modules)
    if latent_df.empty:
        print("No data to analyze.")
        return
        
    latent_df = latent_df[latent_df["score_type"].isin(scorer_names)]
    
    # Separate the dataframes for different processing
    classification_df = latent_df[latent_df['score_type'] != 'surprisal_intervention']
    surprisal_df = latent_df[latent_df['score_type'] == 'surprisal_intervention']

    if not classification_df.empty:
        classification_df = add_latent_f1(classification_df)
        if counts:
            plot_firing_vs_f1(classification_df, num_tokens=10_000_000, out_dir=viz_path, run_label=scores_path.name)
        plot_roc_curve(classification_df, viz_path)
        processed_df = get_agg_metrics(classification_df, counts)
        plot_accuracy_hist(processed_df, viz_path)

    if counts:
        dead = sum((counts[m] == 0).sum().item() for m in modules)
        print(f"Number of dead features: {dead}")
    

    for score_type in latent_df["score_type"].unique():
        
        if score_type == 'surprisal_intervention':
            # Drop duplicates since score is per-latent, not per-example
            unique_latents = surprisal_df.drop_duplicates(subset=['module', 'latent_idx'])
            avg_score = unique_latents['final_score'].mean()
            avg_kl = unique_latents['avg_kl_divergence'].mean()
            
            print(f"\n--- {score_type.title()} Metrics ---")
            print(f"Average Normalized Score: {avg_score:.3f}")
            print(f"Average KL Divergence: {avg_kl:.3f}")

        else:
            if not classification_df.empty:
                score_type_summary = processed_df[processed_df.score_type == score_type].iloc[0]
                print(f"\n--- {score_type.title()} Metrics ---")
                print(f"Class-Balanced Accuracy: {score_type_summary['accuracy']:.3f}")
                print(f"F1 Score: {score_type_summary['f1_score']:.3f}")

                if counts and score_type_summary['weighted_f1'] is not None:
                    print(f"""Frequency-Weighted F1 Score: 
                            {score_type_summary['weighted_f1']:.3f}""")
                
                print(f"Precision: {score_type_summary['precision']:.3f}")
                print(f"Recall: {score_type_summary['recall']:.3f}")
                
                if score_type_summary["auc"] is not None:
                    print(f"AUC: {score_type_summary['auc']:.3f}")
                else:
                    print("AUC not available.")