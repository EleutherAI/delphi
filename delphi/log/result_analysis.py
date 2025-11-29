from pathlib import Path

import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def plot_fuzz_vs_intervention(latent_df: pd.DataFrame, out_dir: Path, run_label: str):
    """
    Replicates the Scatter Plot from the paper (Figure 3/Appendix G).
    Plots Fuzz Score vs. Intervention Score for the same latents.
    """

    # Extract Fuzz Scores (using F1 or Accuracy as the metric)
    fuzz_df = latent_df[latent_df["score_type"] == "fuzz"].copy()
    if fuzz_df.empty:
        return

    # Calculate per-latent F1 for fuzzing
    fuzz_metrics = (
        fuzz_df.groupby(["module", "latent_idx"])
        .apply(
            lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
        )
        .reset_index(name="fuzz_score")
    )

    # Extract Intervention Scores
    int_df = latent_df[latent_df["score_type"] == "surprisal_intervention"].copy()
    if int_df.empty:
        return

    int_metrics = int_df.drop_duplicates(subset=["module", "latent_idx"])[
        ["module", "latent_idx", "avg_kl_divergence", "final_score"]
    ]

    merged = pd.merge(fuzz_metrics, int_metrics, on=["module", "latent_idx"])

    if merged.empty:
        print("Could not merge Fuzz and Intervention scores (no matching latents).")
        return

    # Plot 1: KL vs Fuzz (Causal Impact vs Correlational Quality)
    fig_kl = px.scatter(
        merged,
        x="fuzz_score",
        y="avg_kl_divergence",
        hover_data=["latent_idx"],
        title=f"Correlation vs. Causation (KL) - {run_label}",
        labels={
            "fuzz_score": "Fuzzing Score (Correlation)",
            "avg_kl_divergence": "Intervention KL (Causation)",
        },
        trendline="ols",  # Adds a regression line to show the negative/zero correlation
    )
    fig_kl.write_image(out_dir / "scatter_fuzz_vs_kl.pdf")

    # Plot 2: Score vs Fuzz (Original Paper Metric)
    fig_score = px.scatter(
        merged,
        x="fuzz_score",
        y="final_score",
        hover_data=["latent_idx"],
        title=f"Correlation vs. Causation (Score) - {run_label}",
        labels={
            "fuzz_score": "Fuzzing Score (Correlation)",
            "final_score": "Intervention Score (Surprisal)",
        },
        trendline="ols",
    )
    fig_score.write_image(out_dir / "scatter_fuzz_vs_score.pdf")
    print("Generated Fuzz vs. Intervention scatter plots.")


def plot_intervention_stats(df: pd.DataFrame, out_dir: Path, model_name: str):
    """
    Improved histograms. Plots two versions:
    1. All Features (Log Scale) - to show the dead features.
    2. Live Features Only - to show the distribution of the ones that work.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    display_name = model_name.split("/")[-1] if "/" in model_name else model_name

    # 1. Live/Dead Split Bar Chart
    threshold = 0.01
    df["status"] = df["avg_kl_divergence"].apply(
        lambda x: "Decoder-Live" if x > threshold else "Decoder-Dead"
    )
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]

    total = counts["Count"].sum()
    live = (
        counts[counts["Status"] == "Decoder-Live"]["Count"].sum()
        if "Decoder-Live" in counts["Status"].values
        else 0
    )
    pct = (live / total * 100) if total > 0 else 0

    fig_bar = px.bar(
        counts,
        x="Status",
        y="Count",
        color="Status",
        text="Count",
        title=f"Causal Relevance: {pct:.1f}% Live ({display_name})",
        color_discrete_map={"Decoder-Live": "green", "Decoder-Dead": "red"},
    )
    fig_bar.write_image(out_dir / "intervention_live_dead_split.pdf")

    # 2. "Live Features Only" Histogram
    live_df = df[df["avg_kl_divergence"] > threshold]
    if not live_df.empty:
        fig_live = px.histogram(
            live_df,
            x="avg_kl_divergence",
            nbins=20,
            title=f"Distribution of LIVE Features Only ({display_name})",
            labels={"avg_kl_divergence": "KL Divergence (Causal Effect)"},
        )
        fig_live.update_layout(showlegend=False)
        fig_live.write_image(out_dir / "intervention_kl_dist_LIVE_ONLY.pdf")

    # 3. All Features Histogram (Log Scale)
    fig_all = px.histogram(
        df,
        x="avg_kl_divergence",
        nbins=50,
        title=f"Distribution of All Features ({display_name})",
        labels={"avg_kl_divergence": "KL Divergence"},
        log_y=True,  # Log scale to handle the massive spike at 0
    )
    fig_all.write_image(out_dir / "intervention_kl_dist_log_scale.pdf")


def plot_firing_vs_f1(latent_df, num_tokens, out_dir, run_label):
    out_dir.mkdir(parents=True, exist_ok=True)
    for module, module_df in latent_df.groupby("module"):
        if "firing_count" not in module_df.columns:
            continue
        module_df = module_df[module_df["f1_score"].notna()]
        if module_df.empty:
            continue

        module_df["firing_rate"] = module_df["firing_count"] / num_tokens
        fig = px.scatter(module_df, x="firing_rate", y="f1_score", log_x=True)
        fig.update_layout(
            xaxis_title="Firing rate", yaxis_title="F1 score", xaxis_range=[-5.4, 0]
        )
        fig.write_image(out_dir / f"{run_label}_{module}_firing_rates.pdf")


def import_plotly():
    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError:
        raise ImportError("Install plotly: pip install plotly")
    pio.kaleido.scope.mathjax = None
    return px


def plot_accuracy_hist(df, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)
    for label in df["score_type"].unique():
        if label == "surprisal_intervention":
            continue
        fig = px.histogram(
            df[df["score_type"] == label],
            x="accuracy",
            nbins=100,
            title=f"Accuracy: {label}",
        )
        fig.write_image(out_dir / f"{label}_accuracy.pdf")


def plot_roc_curve(df, out_dir):
    valid_df = df[df.probability.notna()]
    if valid_df.empty or valid_df.activating.nunique() <= 1:
        return
    fpr, tpr, _ = roc_curve(valid_df.activating, valid_df.probability)
    auc = roc_auc_score(valid_df.activating, valid_df.probability)
    fig = go.Figure(
        data=[
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"),
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")),
        ]
    )
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.write_image(out_dir / "roc_curve.pdf")


def compute_confusion(df, threshold=0.5):
    df_valid = df[df["prediction"].notna()]
    if df_valid.empty:
        return dict(
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            total_positives=0,
            total_negatives=0,
        )
    act = df_valid["activating"].astype(bool)
    pred = df_valid["prediction"] >= threshold
    tp, tn = (pred & act).sum(), (~pred & ~act).sum()
    fp, fn = (pred & ~act).sum(), (~pred & act).sum()
    return dict(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_positives=act.sum(),
        total_negatives=(~act).sum(),
    )


def compute_classification_metrics(conf):
    tp, tn, fp, _ = (
        conf["true_positives"],
        conf["true_negatives"],
        conf["false_positives"],
        conf["false_negatives"],
    )
    pos, neg = conf["total_positives"], conf["total_negatives"]
    acc = ((tp / pos if pos else 0) + (tn / neg if neg else 0)) / 2
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / pos if pos else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0
    return dict(accuracy=acc, precision=prec, recall=rec, f1_score=f1)


def compute_auc(df):
    valid = df[df.probability.notna()]
    if valid.probability.nunique() <= 1:
        return None
    return roc_auc_score(valid.activating, valid.probability)


def get_agg_metrics(df):
    rows = []
    for scorer, group in df.groupby("score_type"):
        if scorer == "surprisal_intervention":
            continue
        conf = compute_confusion(group)
        rows.append(
            {
                "score_type": scorer,
                **conf,
                **compute_classification_metrics(conf),
                "auc": compute_auc(group),
            }
        )
    return pd.DataFrame(rows)


def add_latent_f1(df):
    f1s = (
        df.groupby(["module", "latent_idx"])
        .apply(
            lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
        )
        .reset_index(name="f1_score")
    )
    return df.merge(f1s, on=["module", "latent_idx"])


def load_data(scores_path, modules):
    def parse_file(path):
        try:
            data = orjson.loads(path.read_bytes())
            if not isinstance(data, list):
                return pd.DataFrame()
            latent_idx = int(path.stem.split("latent")[-1])
            return pd.DataFrame(
                [
                    {
                        "text": "".join(ex.get("str_tokens", [])),
                        "activating": ex.get("activating"),
                        "prediction": ex.get("prediction"),
                        "probability": ex.get("probability"),
                        "final_score": ex.get("final_score"),
                        "avg_kl_divergence": ex.get("avg_kl_divergence"),
                        "latent_idx": latent_idx,
                    }
                    for ex in data
                ]
            )
        except Exception:
            return pd.DataFrame()

    counts_file = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
    counts = torch.load(counts_file, weights_only=True) if counts_file.exists() else {}

    dfs = []
    for scorer_dir in scores_path.iterdir():
        if not scorer_dir.is_dir():
            continue
        for module in modules:
            for f in scorer_dir.glob(f"*{module}*"):
                df = parse_file(f)
                if df.empty:
                    continue
                df["score_type"] = scorer_dir.name
                df["module"] = module
                if module in counts:
                    idx = df["latent_idx"].iloc[0]
                    if idx < len(counts[module]):
                        df["firing_count"] = counts[module][idx].item()
                dfs.append(df)

    return (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()), counts


def log_results(
    scores_path: Path,
    viz_path: Path,
    modules: list[str],
    scorer_names: list[str],
    model_name: str = "Unknown",
):
    import_plotly()

    latent_df, counts = load_data(scores_path, modules)
    if latent_df.empty:
        print("No data found.")
        return

    print(f"Generating report for: {latent_df['score_type'].unique()}")

    # Split Data
    class_mask = latent_df["score_type"] != "surprisal_intervention"
    class_df = latent_df[class_mask]
    int_df = latent_df[~class_mask]

    # 1. Handle Classification (Fuzz/Detection)
    if not class_df.empty:
        class_df = add_latent_f1(class_df)
        if counts:
            plot_firing_vs_f1(class_df, 10_000_000, viz_path, scores_path.name)
        plot_roc_curve(class_df, viz_path)

        agg_df = get_agg_metrics(class_df)
        plot_accuracy_hist(agg_df, viz_path)

        for _, row in agg_df.iterrows():
            print(f"\n[ {row['score_type'].title()} ]")
            print(f"Accuracy:  {row['accuracy']:.3f}")
            print(f"F1 Score:  {row['f1_score']:.3f}")

    # 2. Handle Intervention
    if not int_df.empty:
        unique_latents = int_df.drop_duplicates(subset=["module", "latent_idx"]).copy()

        avg_score = unique_latents["final_score"].mean()
        avg_kl = unique_latents["avg_kl_divergence"].mean()

        threshold = 0.01
        n_total = len(unique_latents)
        n_live = len(unique_latents[unique_latents["avg_kl_divergence"] > threshold])
        pct = (n_live / n_total * 100) if n_total > 0 else 0

        print("\n--- Surprisal Intervention Analysis ---")
        print(f"Avg Normalized Score: {avg_score:.3f}")
        print(f"Avg KL Divergence:    {avg_kl:.3f}")
        print(f"Decoder-Live %:       {pct:.2f}%")

        plot_intervention_stats(unique_latents, viz_path, model_name)

    # 3. Generate Scatter Plot (Fuzz vs. Intervention)
    if not class_df.empty and not int_df.empty:
        plot_fuzz_vs_intervention(latent_df, viz_path, scores_path.name)
