import orjson
import pandas as pd
from torch import Tensor
from pathlib import Path
import numpy as np
from scipy import stats
import plotly.express as px


def feature_balanced_score_metrics(df: pd.DataFrame, score_type: str):
    # Calculate weights based on non-errored examples
    valid_examples = df['total_examples']
    weights = valid_examples / valid_examples.sum()

    weighted_mean_metrics = {
        'accuracy': np.average(df['accuracy'], weights=weights),
        'f1_score': np.average(df['f1_score'], weights=weights),
        'precision': np.average(df['precision'], weights=weights),
        'recall': np.average(df['recall'], weights=weights),
        'false_positives': np.average(df['false_positives'], weights=weights),
        'false_negatives': np.average(df['false_negatives'], weights=weights),
        'true_positives': np.average(df['true_positives'], weights=weights),
        'true_negatives': np.average(df['true_negatives'], weights=weights),
        'positive_class_ratio': np.average(df['positive_class_ratio'], weights=weights),
        'negative_class_ratio': np.average(df['negative_class_ratio'], weights=weights),
        'total_positives': np.average(df['total_positives'], weights=weights),
        'total_negatives': np.average(df['total_negatives'], weights=weights),
        'true_positive_rate': np.average(df['true_positive_rate'], weights=weights),
        'true_negative_rate': np.average(df['true_negative_rate'], weights=weights),
        'false_positive_rate': np.average(df['false_positive_rate'], weights=weights),
        'false_negative_rate': np.average(df['false_negative_rate'], weights=weights),
    }

    print(f"\n--- {score_type.title()} Metrics ---")
    print(f"Accuracy: {weighted_mean_metrics['accuracy']:.3f}")
    print(f"F1 Score: {weighted_mean_metrics['f1_score']:.3f}")
    print(f"Precision: {weighted_mean_metrics['precision']:.3f}")
    print(f"Recall: {weighted_mean_metrics['recall']:.3f}")

    fractions_failed = [failed_count / total_examples for failed_count, total_examples in zip(df['failed_count'], df['total_examples'])]
    print(f"Average fraction of failed examples: {sum(fractions_failed) / len(fractions_failed):.3f}")

    print("\nConfusion Matrix:")
    print(f"True Positive Rate:  {weighted_mean_metrics['true_positive_rate']:.3f}")
    print(f"True Negative Rate:  {weighted_mean_metrics['true_negative_rate']:.3f}")
    print(f"False Positive Rate: {weighted_mean_metrics['false_positive_rate']:.3f}")
    print(f"False Negative Rate: {weighted_mean_metrics['false_negative_rate']:.3f}")
    
    print(f"\nClass Distribution:")
    print(f"Positives: {df['total_positives'].sum():.0f} ({weighted_mean_metrics['positive_class_ratio']:.1%})")
    print(f"Negatives: {df['total_negatives'].sum():.0f} ({weighted_mean_metrics['negative_class_ratio']:.1%})")
    print(f"Total: {df['total_examples'].sum():.0f}")

    return weighted_mean_metrics

        
def parse_score_file(file_path):
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())
    
    df = pd.DataFrame([{
        "text": "".join(example["str_tokens"]),
        "distance": example["distance"],
        "ground_truth": example["ground_truth"],
        "prediction": example["prediction"],
        "probability": example["probability"],
        "correct": example["correct"],
        "activations": example["activations"],
        "highlighted": example["highlighted"]
    } for example in data])
    
    # Calculate basic counts
    failed_count = (df['prediction'] == -1).sum()
    df = df[df['prediction'] != -1]
    df.reset_index(drop=True, inplace=True)
    total_examples = len(df)
    total_positives = (df["ground_truth"]).sum()
    total_negatives = (~df["ground_truth"]).sum()
    
    # Calculate confusion matrix elements
    true_positives = ((df["prediction"] == 1) & (df["ground_truth"])).sum()
    true_negatives = ((df["prediction"] == 0) & (~df["ground_truth"])).sum()
    false_positives = ((df["prediction"] == 1) & (~df["ground_truth"])).sum()
    false_negatives = ((df["prediction"] == 0) & (df["ground_truth"])).sum()
    
    # Calculate rates
    true_positive_rate = true_positives / total_positives if total_positives > 0 else 0
    true_negative_rate = true_negatives / total_negatives if total_negatives > 0 else 0
    false_positive_rate = false_positives / total_negatives if total_negatives > 0 else 0
    false_negative_rate = false_negatives / total_positives if total_positives > 0 else 0
    
    # Calculate precision, recall, f1 (using sklearn for verification)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positive_rate  # Same as TPR
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / total_examples
    
    # Add metrics to first row
    metrics = {
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "total_examples": total_examples,
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "positive_class_ratio": total_positives / total_examples,
        "negative_class_ratio": total_negatives / total_examples,
        "failed_count": failed_count,
    }
    
    for key, value in metrics.items():
        df.loc[0, key] = value
    
    return df


def build_scores_df(path: Path, target_modules: list[str], range: Tensor | None = None):
    metrics_cols = [
        "accuracy", "probability", "precision", "recall", "f1_score",
        "true_positives", "true_negatives", "false_positives", "false_negatives",
        "true_positive_rate", "true_negative_rate", "false_positive_rate", "false_negative_rate",
        "total_examples", "total_positives", "total_negatives",
        "positive_class_ratio", "negative_class_ratio", "failed_count"
    ]
    df_data = {
        col: [] 
        for col in ["file_name", "score_type", "feature_idx", "module"] + metrics_cols
    }

    # Get subdirectories in the scores path
    scores_types = [d.name for d in path.iterdir() if d.is_dir()]

    for score_type in scores_types:
        score_type_path = path / score_type

        for module in target_modules:
            for score_file in list(score_type_path.glob(f"*{module}*")) + list(
                score_type_path.glob(f".*{module}*")
            ):
                feature_idx = int(score_file.stem.split("latent")[-1]) # Wrong??
                if range is not None and feature_idx not in range:
                    continue

                df = parse_score_file(score_file)

                # Calculate the accuracy and cross entropy loss for this feature
                df_data["file_name"].append(score_file.stem)
                df_data["score_type"].append(score_type)
                df_data["feature_idx"].append(feature_idx)
                df_data["module"].append(module)
                for col in metrics_cols: df_data[col].append(df.loc[0, col])


    df = pd.DataFrame(df_data)
    assert not df.empty
    return df

def plot_line(df):
    out_path = Path("images")
    out_path.mkdir(parents=True, exist_ok=True)

    for score_type in df["score_type"].unique():
        # Create density curves for accuracies
        plot_data = []
        mask = (df["score_type"] == score_type)
        values = df[mask]["accuracy"]
        if len(values) > 0:
            kernel = stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            density = kernel(x_range)
            plot_data.extend([{"x": x, "density": d} 
                            for x, d in zip(x_range, density)])

        fig = px.line(
            plot_data,
            x="x",
            y="density",
            title=f"Accuracy Distribution - {score_type}"
        )
        fig.write_image(out_path / f"autointerp_accuracies_{score_type}.pdf", format="pdf")

def log_results(scores_path: Path, target_modules: list[str]):
    df = build_scores_df(scores_path, target_modules)
    plot_line(df)
    for score_type in df["score_type"].unique():
        score_df = df[df['score_type'] == score_type]
        feature_balanced_score_metrics(score_df, score_type)

if __name__ == "__main__":
    log_results(Path("results") / "scores", "layers.5.mlp")