# from pathlib import Path
# from typing import Optional

# import orjson
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import torch
# from sklearn.metrics import roc_auc_score, roc_curve


# def plot_firing_vs_f1(
#     latent_df: pd.DataFrame, num_tokens: int, out_dir: Path, run_label: str
# ) -> None:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     for module, module_df in latent_df.groupby("module"):

#         if "firing_count" not in module_df.columns:
#             print(
#                 f"""WARNING: 'firing_count' column not found for module {module}.
#                       Skipping plot."""
#             )
#             continue

#         module_df = module_df.copy()
#         # Filter out rows where f1_score is NaN to avoid errors in plotting
#         module_df = module_df[module_df["f1_score"].notna()]
#         if module_df.empty:
#             continue

#         module_df["firing_rate"] = module_df["firing_count"] / num_tokens
#         fig = px.scatter(module_df, x="firing_rate", y="f1_score", log_x=True)
#         fig.update_layout(
#             xaxis_title="Firing rate", yaxis_title="F1 score", xaxis_range=[-5.4, 0]
#         )
#         fig.write_image(out_dir / f"{run_label}_{module}_firing_rates.pdf")


# def import_plotly():
#     """Import plotly with mitigiation for MathJax bug."""
#     try:
#         import plotly.express as px
#         import plotly.io as pio
#     except ImportError:
#         raise ImportError(
#             "Plotly is not installed.\n"
#             "Please install it using `pip install plotly`, "
#             "or install the `[visualize]` extra."
#         )
#     pio.kaleido.scope.mathjax = None
#     return px


# def compute_auc(df: pd.DataFrame) -> float | None:

#     valid_df = df[df.probability.notna()]
#     if valid_df.probability.nunique() <= 1:
#         return None
#     return roc_auc_score(valid_df.activating, valid_df.probability)


# def plot_accuracy_hist(df: pd.DataFrame, out_dir: Path):
#     out_dir.mkdir(exist_ok=True, parents=True)
#     for label in df["score_type"].unique():
#         # Filter out surprisal_intervention as 'accuracy' is not relevant for it
#         if label == "surprisal_intervention":
#             continue
#         fig = px.histogram(
#             df[df["score_type"] == label],
#             x="accuracy",
#             nbins=100,
#             title=f"Accuracy distribution: {label}",
#         )
#         fig.write_image(out_dir / f"{label}_accuracy.pdf")


# def plot_intervention_stats(df: pd.DataFrame, out_dir: Path, model_name: str):
#     """
#     Plots statistics for the surprisal_intervention scorer:
#     1. A histogram of the KL Divergence scores.
#     2. A bar chart of 'Decoder-Live' vs 'Decoder-Dead' features.
#     """
#     out_dir.mkdir(exist_ok=True, parents=True)
    
#     display_name = model_name.split("/")[-1] if "/" in model_name else model_name

#     # 1. KL Divergence Histogram
#     # This shows the distribution of "causal impact" across all features
#     fig_hist = px.histogram(
#         df,
#         x="avg_kl_divergence",
#         nbins=50,
#         title="Distribution of Intervention KL Divergence ({display_name})",
#         labels={"avg_kl_divergence": "Average KL Divergence (Causal Effect)"},
#         log_y=True  # Log scale helps visualize the 'long tail' if many are 0
#     )
#     fig_hist.update_layout(showlegend=False)
#     fig_hist.write_image(out_dir / "intervention_kl_histogram.pdf")

#     # 2. Live vs Dead Bar Chart
#     # We define "Live" as having a KL > 0.01 (non-zero effect)
#     threshold = 0.01
#     df["status"] = df["avg_kl_divergence"].apply(
#         lambda x: "Decoder-Live" if x > threshold else "Decoder-Dead"
#     )
    
#     counts = df["status"].value_counts().reset_index()
#     counts.columns = ["Status", "Count"]
    
#     # Calculate percentage for the title
#     total = counts["Count"].sum()
#     live_count = counts[counts["Status"] == "Decoder-Live"]["Count"].sum()
#     live_pct = (live_count / total) * 100 if total > 0 else 0

#     fig_bar = px.bar(
#         counts,
#         x="Status",
#         y="Count",
#         color="Status",
#         title=f"Causal Relevance: {live_pct:.1f}% Live Features",
#         text="Count",
#         color_discrete_map={"Decoder-Live": "green", "Decoder-Dead": "red"}
#     )
#     fig_bar.update_traces(textposition='auto')
#     fig_bar.write_image(out_dir / "intervention_live_dead_split.pdf")


# def plot_roc_curve(df: pd.DataFrame, out_dir: Path):

#     valid_df = df[df.probability.notna()]
#     if (
#         valid_df.empty
#         or valid_df.activating.nunique() <= 1
#         or valid_df.probability.nunique() <= 1
#     ):
#         return

#     fpr, tpr, _ = roc_curve(valid_df.activating, valid_df.probability)
#     auc = roc_auc_score(valid_df.activating, valid_df.probability)
#     fig = go.Figure(
#         data=[
#             go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"),
#             go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")),
#         ]
#     )
#     fig.update_layout(
#         title="ROC Curve",
#         xaxis_title="FPR",
#         yaxis_title="TPR",
#     )
#     out_dir.mkdir(exist_ok=True, parents=True)
#     fig.write_image(out_dir / "roc_curve.pdf")


# def compute_confusion(df: pd.DataFrame, threshold: float = 0.5) -> dict:
#     df_valid = df[df["prediction"].notna()]
#     if df_valid.empty:
#         return dict(
#             true_positives=0,
#             true_negatives=0,
#             false_positives=0,
#             false_negatives=0,
#             total_examples=0,
#             total_positives=0,
#             total_negatives=0,
#             failed_count=len(df),
#         )

#     act = df_valid["activating"].astype(bool)
#     total = len(df_valid)
#     pos = act.sum()
#     neg = total - pos
#     tp = ((df_valid.prediction >= threshold) & act).sum()
#     tn = ((df_valid.prediction < threshold) & ~act).sum()
#     fp = ((df_valid.prediction >= threshold) & ~act).sum()
#     fn = ((df_valid.prediction < threshold) & act).sum()

#     return dict(
#         true_positives=tp,
#         true_negatives=tn,
#         false_positives=fp,
#         false_negatives=fn,
#         total_examples=total,
#         total_positives=pos,
#         total_negatives=neg,
#         failed_count=len(df) - len(df_valid),
#     )


# def compute_classification_metrics(conf: dict) -> dict:
#     tp, tn, fp, fn = (
#         conf["true_positives"],
#         conf["true_negatives"],
#         conf["false_positives"],
#         conf["false_negatives"],
#     )
#     pos, neg = conf["total_positives"], conf["total_negatives"]

#     balanced_accuracy = (
#         (tp / pos if pos > 0 else 0) + (tn / neg if neg > 0 else 0)
#     ) / 2
#     precision = tp / (tp + fp) if tp + fp > 0 else 0
#     recall = tp / pos if pos > 0 else 0
#     f1 = (
#         2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#     )

#     return dict(
#         precision=precision,
#         recall=recall,
#         f1_score=f1,
#         accuracy=balanced_accuracy,
#         true_positive_rate=tp / pos if pos > 0 else 0,
#         true_negative_rate=tn / neg if neg > 0 else 0,
#         false_positive_rate=fp / neg if neg > 0 else 0,
#         false_negative_rate=fn / pos if pos > 0 else 0,
#     )


# def load_data(scores_path: Path, modules: list[str]):
#     """Load all on-disk data into a single DataFrame."""

#     def parse_score_file(path: Path) -> pd.DataFrame:
#         try:
#             data = orjson.loads(path.read_bytes())
#         except orjson.JSONDecodeError:
#             print(f"Error decoding JSON from {path}. Skipping file.")
#             return pd.DataFrame()

#         if not isinstance(data, list):
#             print(
#                 f"""Warning: Expected a list of results in {path},
#                       but found {type(data)}.
#                       Skipping file."""
#             )
#             return pd.DataFrame()

#         latent_idx = int(path.stem.split("latent")[-1])

#         # Updated to extract all possible keys safely using .get()
#         return pd.DataFrame(
#             [
#                 {
#                     "text": "".join(ex.get("str_tokens", [])),
#                     "distance": ex.get("distance"),
#                     "activating": ex.get("activating"),
#                     "prediction": ex.get("prediction"),
#                     "probability": ex.get("probability"),
#                     "correct": ex.get("correct"),
#                     "activations": ex.get("activations"),
#                     "final_score": ex.get("final_score"),
#                     "avg_kl_divergence": ex.get("avg_kl_divergence"),
#                     "latent_idx": latent_idx,
#                 }
#                 for ex in data
#             ]
#         )

#     counts_file = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
#     counts = torch.load(counts_file, weights_only=True) if counts_file.exists() else {}
#     if not all(module in counts for module in modules):
#         print("Missing firing counts for some modules, setting counts to None.")
#         print(f"Missing modules: {[m for m in modules if m not in counts]}")
#         counts = None

#     latent_dfs = []
#     for score_type_dir in scores_path.iterdir():
#         if not score_type_dir.is_dir():
#             continue
#         for module in modules:
#             for file in score_type_dir.glob(f"*{module}*"):
#                 latent_df = parse_score_file(file)
#                 if latent_df.empty:
#                     continue
#                 latent_df["score_type"] = score_type_dir.name
#                 latent_df["module"] = module
#                 if counts:
#                     latent_idx = latent_df["latent_idx"].iloc[0]
#                     latent_df["firing_count"] = (
#                         counts[module][latent_idx].item()
#                         if module in counts and latent_idx in counts[module]
#                         else None
#                     )
#                 latent_dfs.append(latent_df)

#     if not latent_dfs:
#         return pd.DataFrame(), counts

#     return pd.concat(latent_dfs, ignore_index=True), counts


# def get_agg_metrics(
#     latent_df: pd.DataFrame, counts: Optional[dict[str, torch.Tensor]]
# ) -> pd.DataFrame:
#     processed_rows = []
#     for score_type, group_df in latent_df.groupby("score_type"):
#         # For surprisal_intervention, we don't compute classification metrics
#         if score_type == "surprisal_intervention":
#             continue

#         conf = compute_confusion(group_df)
#         class_m = compute_classification_metrics(conf)
#         auc = compute_auc(group_df)
#         f1_w = frequency_weighted_f1(group_df, counts) if counts else None

#         row = {
#             "score_type": score_type,
#             **conf,
#             **class_m,
#             "auc": auc,
#             "weighted_f1": f1_w,
#         }
#         processed_rows.append(row)

#     return pd.DataFrame(processed_rows)


# def add_latent_f1(latent_df: pd.DataFrame) -> pd.DataFrame:
#     f1s = (
#         latent_df.groupby(["module", "latent_idx"])
#         .apply(
#             lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
#         )
#         .reset_index(name="f1_score")  # <- naive (un-weighted) F1
#     )
#     return latent_df.merge(f1s, on=["module", "latent_idx"])


# def log_results(
#     scores_path: Path, 
#     viz_path: Path, 
#     modules: list[str], 
#     scorer_names: list[str], 
#     model_name: str = "Unknown Model"
# ):
#     import_plotly()

#     latent_df, counts = load_data(scores_path, modules)
#     if latent_df.empty:
#         print("No data to analyze.")
#         return

#     latent_df = latent_df[latent_df["score_type"].isin(scorer_names)]

#     # Separate the dataframes for different processing
#     classification_df = latent_df[latent_df["score_type"] != "surprisal_intervention"]
#     surprisal_df = latent_df[latent_df["score_type"] == "surprisal_intervention"]

#     if not classification_df.empty:
#         classification_df = add_latent_f1(classification_df)
#         if counts:
#             plot_firing_vs_f1(
#                 classification_df,
#                 num_tokens=10_000_000,
#                 out_dir=viz_path,
#                 run_label=scores_path.name,
#             )
#         plot_roc_curve(classification_df, viz_path)
#         processed_df = get_agg_metrics(classification_df, counts)
#         plot_accuracy_hist(processed_df, viz_path)

#     if counts:
#         dead = sum((counts[m] == 0).sum().item() for m in modules)
#         print(f"Number of dead features: {dead}")

#     for score_type in latent_df["score_type"].unique():

#         if score_type == "surprisal_intervention":
#             # Drop duplicates since score is per-latent, not per-example
#             unique_latents = surprisal_df.drop_duplicates(
#                 subset=["module", "latent_idx"]
#             ).copy()

#             avg_score = unique_latents["final_score"].mean()
#             avg_kl = unique_latents["avg_kl_divergence"].mean()

#             # We define "Decoder-Live" as having a KL > 0.01 (non-zero effect)
#             threshold = 0.01
#             n_total = len(unique_latents)
#             n_live = len(unique_latents[unique_latents["avg_kl_divergence"] > threshold])
#             live_pct = (n_live / n_total) * 100 if n_total > 0 else 0.0

#             print(f"\n--- {score_type.title()} Metrics ---")
#             print(f"Average Normalized Score: {avg_score:.3f}")
#             print(f"Average KL Divergence: {avg_kl:.3f}")
#             print(f"Decoder-Live Percentage: {live_pct:.2f}%")
            

#             plot_intervention_stats(unique_latents, viz_path, model_name)

#         else:
#             if not classification_df.empty:
#                 score_type_summary = processed_df[
#                     processed_df.score_type == score_type
#                 ].iloc[0]
#                 print(f"\n--- {score_type.title()} Metrics ---")
#                 print(f"Class-Balanced Accuracy: {score_type_summary['accuracy']:.3f}")
#                 print(f"F1 Score: {score_type_summary['f1_score']:.3f}")

#                 if counts and score_type_summary["weighted_f1"] is not None:
#                     print(
#                         f"""Frequency-Weighted F1 Score:
#                             {score_type_summary['weighted_f1']:.3f}"""
#                     )

#                 print(f"Precision: {score_type_summary['precision']:.3f}")
#                 print(f"Recall: {score_type_summary['recall']:.3f}")

#                 if score_type_summary["auc"] is not None:
#                     print(f"AUC: {score_type_summary['auc']:.3f}")
#                 else:
#                     print("AUC not available.")

# from pathlib import Path
# from typing import Optional

# import orjson
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import torch
# from sklearn.metrics import roc_auc_score, roc_curve


# # --- PLOTTING HELPERS ---

# def import_plotly():
#     """Import plotly with mitigation for MathJax bug."""
#     try:
#         import plotly.express as px
#         import plotly.io as pio
#     except ImportError:
#         raise ImportError(
#             "Plotly is not installed. Please install it via `pip install plotly`."
#         )
#     pio.kaleido.scope.mathjax = None
#     return px

# def plot_firing_vs_f1(latent_df, num_tokens, out_dir, run_label):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     for module, module_df in latent_df.groupby("module"):
#         if "firing_count" not in module_df.columns:
#             continue
#         module_df = module_df[module_df["f1_score"].notna()]
#         if module_df.empty: continue

#         module_df["firing_rate"] = module_df["firing_count"] / num_tokens
#         fig = px.scatter(module_df, x="firing_rate", y="f1_score", log_x=True)
#         fig.update_layout(xaxis_title="Firing rate", yaxis_title="F1 score", xaxis_range=[-5.4, 0])
#         fig.write_image(out_dir / f"{run_label}_{module}_firing_rates.pdf")

# def plot_roc_curve(df, out_dir):
#     valid_df = df[df.probability.notna()]
#     if valid_df.empty or valid_df.activating.nunique() <= 1 or valid_df.probability.nunique() <= 1:
#         return

#     fpr, tpr, _ = roc_curve(valid_df.activating, valid_df.probability)
#     auc = roc_auc_score(valid_df.activating, valid_df.probability)
#     fig = go.Figure(data=[
#         go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"),
#         go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")),
#     ])
#     fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
#     out_dir.mkdir(exist_ok=True, parents=True)
#     fig.write_image(out_dir / "roc_curve.pdf")

# def plot_accuracy_hist(df, out_dir):
#     out_dir.mkdir(exist_ok=True, parents=True)
#     for label in df["score_type"].unique():
#         fig = px.histogram(df[df["score_type"] == label], x="accuracy", nbins=100, title=f"Accuracy: {label}")
#         fig.write_image(out_dir / f"{label}_accuracy.pdf")

# def plot_intervention_stats(df, out_dir, model_name):
#     """Specific plots for Intervention scoring."""
#     out_dir.mkdir(exist_ok=True, parents=True)
#     display_name = model_name.split("/")[-1] if "/" in model_name else model_name
    
#     # 1. KL Histogram
#     fig_hist = px.histogram(
#         df, x="avg_kl_divergence", nbins=50, log_y=True,
#         title=f"KL Divergence ({display_name})",
#         labels={"avg_kl_divergence": "Avg KL Divergence (Causal Effect)"}
#     )
#     fig_hist.write_image(out_dir / "intervention_kl_dist.pdf")

#     # 2. Live/Dead Split
#     threshold = 0.01
#     df["status"] = df["avg_kl_divergence"].apply(lambda x: "Decoder-Live" if x > threshold else "Decoder-Dead")
#     counts = df["status"].value_counts().reset_index()
#     counts.columns = ["Status", "Count"]
    
#     total = counts["Count"].sum()
#     live = counts[counts["Status"] == "Decoder-Live"]["Count"].sum() if "Decoder-Live" in counts["Status"].values else 0
#     pct = (live / total * 100) if total > 0 else 0

#     fig_bar = px.bar(
#         counts, x="Status", y="Count", color="Status", text="Count",
#         title=f"Causal Relevance: {pct:.1f}% Live ({display_name})",
#         color_discrete_map={"Decoder-Live": "green", "Decoder-Dead": "red"}
#     )
#     fig_bar.write_image(out_dir / "intervention_live_dead.pdf")


# # --- METRIC COMPUTATION ---

# def compute_confusion(df, threshold=0.5):
#     df_valid = df[df["prediction"].notna()]
#     if df_valid.empty: return dict(tp=0, tn=0, fp=0, fn=0, pos=0, neg=0)
    
#     act = df_valid["activating"].astype(bool)
#     pred = df_valid["prediction"] >= threshold
    
#     tp = (pred & act).sum()
#     tn = (~pred & ~act).sum()
#     fp = (pred & ~act).sum()
#     fn = (~pred & act).sum()
    
#     return dict(
#         true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn,
#         total_positives=act.sum(), total_negatives=(~act).sum()
#     )

# def compute_classification_metrics(conf):
#     tp, tn, fp, fn = conf["true_positives"], conf["true_negatives"], conf["false_positives"], conf["false_negatives"]
#     pos, neg = conf["total_positives"], conf["total_negatives"]
    
#     acc = ((tp / pos if pos else 0) + (tn / neg if neg else 0)) / 2
#     prec = tp / (tp + fp) if (tp + fp) else 0
#     rec = tp / pos if pos else 0
#     f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0
    
#     return dict(accuracy=acc, precision=prec, recall=rec, f1_score=f1)

# def compute_auc(df):
#     valid = df[df.probability.notna()]
#     if valid.probability.nunique() <= 1: return None
#     return roc_auc_score(valid.activating, valid.probability)

# def get_agg_metrics(df):
#     rows = []
#     for scorer, group in df.groupby("score_type"):
#         conf = compute_confusion(group)
#         metrics = compute_classification_metrics(conf)
#         rows.append({"score_type": scorer, **conf, **metrics, "auc": compute_auc(group)})
#     return pd.DataFrame(rows)

# def add_latent_f1(df):
#     # Calculate F1 per latent for plotting
#     f1s = df.groupby(["module", "latent_idx"]).apply(
#         lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
#     ).reset_index(name="f1_score")
#     return df.merge(f1s, on=["module", "latent_idx"])


# # --- DATA LOADING ---

# def load_data(scores_path, modules):
#     def parse_file(path):
#         try:
#             data = orjson.loads(path.read_bytes())
#             if not isinstance(data, list): return pd.DataFrame()
#             latent_idx = int(path.stem.split("latent")[-1])
#             return pd.DataFrame([{
#                 "text": "".join(ex.get("str_tokens", [])),
#                 "activating": ex.get("activating"),
#                 "prediction": ex.get("prediction"),
#                 "probability": ex.get("probability"),
#                 "final_score": ex.get("final_score"),
#                 "avg_kl_divergence": ex.get("avg_kl_divergence"),
#                 "latent_idx": latent_idx
#             } for ex in data])
#         except Exception:
#             return pd.DataFrame()

#     counts_file = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
#     counts = torch.load(counts_file, weights_only=True) if counts_file.exists() else {}
    
#     dfs = []
#     for scorer_dir in scores_path.iterdir():
#         if not scorer_dir.is_dir(): continue
#         for module in modules:
#             for f in scorer_dir.glob(f"*{module}*"):
#                 df = parse_file(f)
#                 if df.empty: continue
#                 df["score_type"] = scorer_dir.name
#                 df["module"] = module
#                 if module in counts:
#                     idx = df["latent_idx"].iloc[0]
#                     if idx < len(counts[module]):
#                         df["firing_count"] = counts[module][idx].item()
#                 dfs.append(df)
    
#     return (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()), counts


# # --- MAIN HANDLERS ---

# def handle_classification_results(df, counts, viz_path, run_label):
#     """Handles Fuzz, Detection, Simulation."""
#     print(f"\n--- Classification Analysis ({len(df)} examples) ---")
    
#     # Add per-latent F1 for plotting
#     df = add_latent_f1(df)
    
#     # Plots
#     if counts:
#         plot_firing_vs_f1(df, 10_000_000, viz_path, run_label)
#     plot_roc_curve(df, viz_path)
    
#     # Aggregated Metrics (Accuracy, F1, etc.)
#     agg_df = get_agg_metrics(df)
#     plot_accuracy_hist(agg_df, viz_path)
    
#     # Console Output
#     for _, row in agg_df.iterrows():
#         print(f"\n[ {row['score_type'].title()} ]")
#         print(f"Accuracy:  {row['accuracy']:.3f}")
#         print(f"F1 Score:  {row['f1_score']:.3f}")
#         print(f"Precision: {row['precision']:.3f}")
#         print(f"Recall:    {row['recall']:.3f}")
#         if row['auc']: print(f"AUC:       {row['auc']:.3f}")


# def handle_intervention_results(df, viz_path, model_name):
#     """Handles Surprisal Intervention."""
#     # Deduplicate: we only need one row per latent per module
#     unique_latents = df.drop_duplicates(subset=["module", "latent_idx"]).copy()
    
#     avg_score = unique_latents["final_score"].mean()
#     avg_kl = unique_latents["avg_kl_divergence"].mean()
    
#     # Calculate Decoder-Live %
#     total = len(unique_latents)
#     live = len(unique_latents[unique_latents["avg_kl_divergence"] > 0.01])
#     pct = (live / total * 100) if total > 0 else 0
    
#     print(f"\n--- Surprisal Intervention Analysis ({total} latents) ---")
#     print(f"Avg Normalized Score: {avg_score:.3f}")
#     print(f"Avg KL Divergence:    {avg_kl:.3f}")
#     print(f"Decoder-Live %:       {pct:.2f}%")
    
#     plot_intervention_stats(unique_latents, viz_path, model_name)


# # --- ENTRY POINT ---

# def log_results(scores_path: Path, viz_path: Path, modules: list[str], scorer_names: list[str], model_name: str = "Unknown"):
#     import_plotly()
    
#     # 1. Load ALL data (Global Reporting)
#     latent_df, counts = load_data(scores_path, modules)
#     if latent_df.empty:
#         print("No data found to analyze.")
#         return

#     print(f"Generating report for scorers found: {latent_df['score_type'].unique()}")

#     # 2. Split Data
#     classification_mask = latent_df["score_type"] != "surprisal_intervention"
#     classification_df = latent_df[classification_mask]
#     intervention_df = latent_df[~classification_mask]

#     # 3. Dispatch to Handlers
#     if not classification_df.empty:
#         handle_classification_results(classification_df, counts, viz_path, scores_path.name)
    
#     if not intervention_df.empty:
#         handle_intervention_results(intervention_df, viz_path, model_name)


from pathlib import Path
from typing import Optional

import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_auc_score, roc_curve


# --- 1. NEW PLOTTING FUNCTIONS ---

def plot_fuzz_vs_intervention(latent_df: pd.DataFrame, out_dir: Path, run_label: str):
    """
    Replicates the Scatter Plot from the paper (Figure 3/Appendix G).
    Plots Fuzz Score vs. Intervention Score for the same latents.
    """
    # We need to merge the rows for 'fuzz' and 'surprisal_intervention'
    # 1. Pivot the table so we have columns: 'latent_idx', 'fuzz_score', 'intervention_score'
    
    # Extract Fuzz Scores (using F1 or Accuracy as the metric)
    fuzz_df = latent_df[latent_df["score_type"] == "fuzz"].copy()
    if fuzz_df.empty: return
    
    # Calculate per-latent F1 for fuzzing
    fuzz_metrics = fuzz_df.groupby(["module", "latent_idx"]).apply(
        lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
    ).reset_index(name="fuzz_score")

    # Extract Intervention Scores
    int_df = latent_df[latent_df["score_type"] == "surprisal_intervention"].copy()
    if int_df.empty: return
    
    # Deduplicate intervention scores
    int_metrics = int_df.drop_duplicates(subset=["module", "latent_idx"])[
        ["module", "latent_idx", "avg_kl_divergence", "final_score"]
    ]

    # Merge them
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
        labels={"fuzz_score": "Fuzzing Score (Correlation)", "avg_kl_divergence": "Intervention KL (Causation)"},
        trendline="ols" # Adds a regression line to show the negative/zero correlation
    )
    fig_kl.write_image(out_dir / "scatter_fuzz_vs_kl.pdf")

    # Plot 2: Score vs Fuzz (Original Paper Metric)
    fig_score = px.scatter(
        merged, 
        x="fuzz_score", 
        y="final_score",
        hover_data=["latent_idx"],
        title=f"Correlation vs. Causation (Score) - {run_label}",
        labels={"fuzz_score": "Fuzzing Score (Correlation)", "final_score": "Intervention Score (Surprisal)"},
        trendline="ols"
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
    
    # Get percentage
    total = counts["Count"].sum()
    live = counts[counts["Status"] == "Decoder-Live"]["Count"].sum() if "Decoder-Live" in counts["Status"].values else 0
    pct = (live / total * 100) if total > 0 else 0

    fig_bar = px.bar(
        counts, x="Status", y="Count", color="Status", text="Count",
        title=f"Causal Relevance: {pct:.1f}% Live ({display_name})",
        color_discrete_map={"Decoder-Live": "green", "Decoder-Dead": "red"}
    )
    fig_bar.write_image(out_dir / "intervention_live_dead_split.pdf")

    # 2. "Live Features Only" Histogram (The "Pretty" one)
    live_df = df[df["avg_kl_divergence"] > threshold]
    if not live_df.empty:
        fig_live = px.histogram(
            live_df, 
            x="avg_kl_divergence", 
            nbins=20,
            title=f"Distribution of LIVE Features Only ({display_name})",
            labels={"avg_kl_divergence": "KL Divergence (Causal Effect)"}
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
        log_y=True # Log scale to handle the massive spike at 0
    )
    fig_all.write_image(out_dir / "intervention_kl_dist_log_scale.pdf")


# --- 2. STANDARD PLOTTING HELPERS ---

def plot_firing_vs_f1(latent_df, num_tokens, out_dir, run_label):
    out_dir.mkdir(parents=True, exist_ok=True)
    for module, module_df in latent_df.groupby("module"):
        if "firing_count" not in module_df.columns: continue
        module_df = module_df[module_df["f1_score"].notna()]
        if module_df.empty: continue

        module_df["firing_rate"] = module_df["firing_count"] / num_tokens
        fig = px.scatter(module_df, x="firing_rate", y="f1_score", log_x=True)
        fig.update_layout(xaxis_title="Firing rate", yaxis_title="F1 score", xaxis_range=[-5.4, 0])
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
        if label == "surprisal_intervention": continue
        fig = px.histogram(df[df["score_type"] == label], x="accuracy", nbins=100, title=f"Accuracy: {label}")
        fig.write_image(out_dir / f"{label}_accuracy.pdf")

def plot_roc_curve(df, out_dir):
    valid_df = df[df.probability.notna()]
    if valid_df.empty or valid_df.activating.nunique() <= 1: return
    fpr, tpr, _ = roc_curve(valid_df.activating, valid_df.probability)
    auc = roc_auc_score(valid_df.activating, valid_df.probability)
    fig = go.Figure(data=[
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"),
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")),
    ])
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.write_image(out_dir / "roc_curve.pdf")


# --- 3. METRIC COMPUTATION ---

def compute_confusion(df, threshold=0.5):
    df_valid = df[df["prediction"].notna()]
    if df_valid.empty: return dict(true_positives=0, true_negatives=0, false_positives=0, false_negatives=0, total_positives=0, total_negatives=0)
    act = df_valid["activating"].astype(bool)
    pred = df_valid["prediction"] >= threshold
    tp, tn = (pred & act).sum(), (~pred & ~act).sum()
    fp, fn = (pred & ~act).sum(), (~pred & act).sum()
    return dict(true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn, total_positives=act.sum(), total_negatives=(~act).sum())

def compute_classification_metrics(conf):
    tp, tn, fp, fn = conf["true_positives"], conf["true_negatives"], conf["false_positives"], conf["false_negatives"]
    pos, neg = conf["total_positives"], conf["total_negatives"]
    acc = ((tp/pos if pos else 0) + (tn/neg if neg else 0)) / 2
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/pos if pos else 0
    f1 = 2*(prec*rec)/(prec+rec) if (prec+rec) else 0
    return dict(accuracy=acc, precision=prec, recall=rec, f1_score=f1)

def compute_auc(df):
    valid = df[df.probability.notna()]
    if valid.probability.nunique() <= 1: return None
    return roc_auc_score(valid.activating, valid.probability)

def get_agg_metrics(df):
    rows = []
    for scorer, group in df.groupby("score_type"):
        if scorer == "surprisal_intervention": continue
        conf = compute_confusion(group)
        rows.append({"score_type": scorer, **conf, **compute_classification_metrics(conf), "auc": compute_auc(group)})
    return pd.DataFrame(rows)

def add_latent_f1(df):
    f1s = df.groupby(["module", "latent_idx"]).apply(
        lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
    ).reset_index(name="f1_score")
    return df.merge(f1s, on=["module", "latent_idx"])


# --- 4. DATA LOADING ---

def load_data(scores_path, modules):
    def parse_file(path):
        try:
            data = orjson.loads(path.read_bytes())
            if not isinstance(data, list): return pd.DataFrame()
            latent_idx = int(path.stem.split("latent")[-1])
            return pd.DataFrame([{
                "text": "".join(ex.get("str_tokens", [])),
                "activating": ex.get("activating"),
                "prediction": ex.get("prediction"),
                "probability": ex.get("probability"),
                "final_score": ex.get("final_score"),
                "avg_kl_divergence": ex.get("avg_kl_divergence"),
                "latent_idx": latent_idx
            } for ex in data])
        except Exception: return pd.DataFrame()

    counts_file = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
    counts = torch.load(counts_file, weights_only=True) if counts_file.exists() else {}
    
    dfs = []
    for scorer_dir in scores_path.iterdir():
        if not scorer_dir.is_dir(): continue
        for module in modules:
            for f in scorer_dir.glob(f"*{module}*"):
                df = parse_file(f)
                if df.empty: continue
                df["score_type"] = scorer_dir.name
                df["module"] = module
                if module in counts:
                    idx = df["latent_idx"].iloc[0]
                    if idx < len(counts[module]):
                        df["firing_count"] = counts[module][idx].item()
                dfs.append(df)
    
    return (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()), counts


# --- 5. MAIN LOGIC ---

def log_results(scores_path: Path, viz_path: Path, modules: list[str], scorer_names: list[str], model_name: str = "Unknown"):
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
        if counts: plot_firing_vs_f1(class_df, 10_000_000, viz_path, scores_path.name)
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
        
        print(f"\n--- Surprisal Intervention Analysis ---")
        print(f"Avg Normalized Score: {avg_score:.3f}")
        print(f"Avg KL Divergence:    {avg_kl:.3f}")
        print(f"Decoder-Live %:       {pct:.2f}%")
        
        plot_intervention_stats(unique_latents, viz_path, model_name)

    # 3. Generate Scatter Plot (Fuzz vs. Intervention)
    # Only works if we have BOTH types of data
    if not class_df.empty and not int_df.empty:
        plot_fuzz_vs_intervention(latent_df, viz_path, scores_path.name)