# app.py
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

# Add the directory containing the sync_latent_dataset.py to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from delphi.__main__ import load_artifacts, populate_cache
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.latents.sync_dataset import SyncLatentDataset


# Helper functions for token visualization
def escape(t):
    t = t.replace("<bos>", "BOS").replace("<", "&lt;").replace(">", "&gt;")
    return t


def convert_token_array_to_list(array):
    if isinstance(array, torch.Tensor):
        if array.dim() == 1:
            array = [array.tolist()]
        elif array.dim() == 2:
            array = array.tolist()
        else:
            raise NotImplementedError("tokens must be 1 or 2 dimensional")
    elif isinstance(array, list):
        if isinstance(array[0], int):
            array = [array]
    return array


# Initialize session state for persistent storage
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.source_data = {}
    st.session_state.tokenizers = {}
    st.session_state.blinded_order = []
    st.session_state.current_index = 0
    st.session_state.evaluations = {}
    st.session_state.current_source = None
    st.session_state.start_time = None
    st.session_state.hookpoint = None


def cache_latents_if_needed(
    checkpoints_dirs, latent_dirs, model_name, hookpoint, max_latents=100
):
    for checkpoint_dir, latent_dir in zip(checkpoints_dirs, latent_dirs):
        if not os.path.exists(latent_dir):
            run_cfg = RunConfig(
                cache_cfg=CacheConfig(
                    dataset_repo="EleutherAI/SmolLM2-135M-10B",
                ),
                constructor_cfg=ConstructorConfig(),
                sampler_cfg=SamplerConfig(),
                model=model_name,
                sparse_model=str(checkpoint_dir),
                hookpoints=[hookpoint],
                name=latent_dir.stem,
                max_latents=max_latents,
            )
            st.write(f"Caching latents for {latent_dir}")
            model = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")

            hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(
                run_cfg
            )
            tokenizer = AutoTokenizer.from_pretrained(
                run_cfg.model, token=run_cfg.hf_token
            )

            populate_cache(
                run_cfg,
                model,
                hookpoint_to_sparse_encode,
                latent_dir,
                tokenizer,
                transcode,
            )


def initialize_test(model_dirs, hookpoint, features_range, num_examples):
    # Load examples from both sources
    source_data = {}
    tokenizers = {}

    with st.spinner("Loading examples from sources..."):
        for model_name, raw_dir in model_dirs.items():
            # Create synchronous dataset
            sync_dataset = SyncLatentDataset(
                raw_dir,
                SamplerConfig(n_examples_train=num_examples),
                ConstructorConfig(),
                modules=[hookpoint],
                latents={hookpoint: torch.LongTensor(list(range(*features_range)))},
            )

            examples, max_acts, tokenizer = sync_dataset.load_examples()
            source_data[model_name] = {
                "examples": examples,
                "max_activations": max_acts,
            }
            tokenizers[model_name] = tokenizer

    # Create blinded test sequence
    all_features = []
    for model_name in model_dirs.keys():
        features = list(source_data[model_name]["examples"].keys())
        all_features.extend([(model_name, feature) for feature in features])

    # Randomly shuffle features from both sources
    random.shuffle(all_features)
    st.write(f"{len(all_features)} features to evaluate")

    # Store in session state
    st.session_state.source_data = source_data
    st.session_state.tokenizers = tokenizers
    st.session_state.blinded_order = all_features
    st.session_state.initialized = True
    st.session_state.current_index = 0
    st.session_state.evaluations = {}


def display_tokens_with_activations(tokens, activations, tokenizer):
    """Display tokens with activations using Plotly"""
    inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

    all_tokens = []
    all_activations = []
    all_example_ids = []

    for i, (example_tokens, example_activations) in enumerate(zip(tokens, activations)):
        example_id = f"Example {i+1}"
        for token_id, activation in zip(example_tokens, example_activations):
            token_text = (
                inverse_vocab[int(token_id)]
                .replace("Ġ", " ")
                .replace("▁", " ")
                .replace("\n", "\\n")
            )
            all_tokens.append(token_text)
            all_activations.append(float(activation))
            all_example_ids.append(example_id)

    # Create DataFrame for Plotly
    df = pd.DataFrame(
        {"Token": all_tokens, "Activation": all_activations, "Example": all_example_ids}
    )

    # Create heatmap-style visualization
    fig = px.bar(
        df,
        x="Token",
        y="Activation",
        color="Activation",
        facet_row="Example",
        color_continuous_scale="RdBu_r",
        height=120 * len(tokens) + 100,
        template="plotly_dark",
    )

    fig.update_layout(
        title="Token Activations",
        xaxis_title="",
        yaxis_title="Activation",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(tickangle=45)

    return fig


def display_current_feature():
    if st.session_state.current_index >= len(st.session_state.blinded_order):
        st.header("All features evaluated!")
        return False

    source_name, feature_key = st.session_state.blinded_order[
        st.session_state.current_index
    ]
    st.session_state.current_source = source_name

    # Display progress
    st.progress(st.session_state.current_index / len(st.session_state.blinded_order))
    len_items = len(st.session_state.blinded_order)
    st.write(f"Progress: {st.session_state.current_index}/{len_items}")

    st.header(f"Feature ID: {feature_key}")

    # Get examples and tokenizer
    examples = st.session_state.source_data[source_name]["examples"][feature_key]
    max_act = st.session_state.source_data[source_name]["max_activations"][feature_key]
    tokenizer = st.session_state.tokenizers[source_name]

    # Record time when showing a new feature
    st.session_state.start_time = time.time()

    # Prepare tokens and activations
    list_tokens = []
    list_activations = []

    num_examples = len(examples)
    for i, example in enumerate(examples):
        if i >= num_examples:
            break

        example_tokens = example.tokens
        activations = example.activations / max_act
        list_tokens.append(example_tokens)
        list_activations.append(activations.tolist())

    # Create and display the visualization
    fig = display_tokens_with_activations(list_tokens, list_activations, tokenizer)
    st.plotly_chart(fig, use_container_width=True)

    # Also display raw tokens with text below
    for i, (tokens, activations) in enumerate(zip(list_tokens, list_activations)):
        token_texts = [tokenizer.decode([int(t)]) for t in tokens]

        st.write(f"**Example {i+1} Text:**")
        # Create a DataFrame with token texts and activations
        df = pd.DataFrame({"Token": token_texts, "Activation": activations})
        st.dataframe(df, use_container_width=True)

    return True


def submit_evaluation():
    source_name, feature_key = st.session_state.blinded_order[
        st.session_state.current_index
    ]

    # Calculate time spent on this feature
    time_spent = (
        time.time() - st.session_state.start_time if st.session_state.start_time else 0
    )

    # Save evaluation
    evaluation = {
        "source": source_name,
        "feature": feature_key,
        "explanation": st.session_state.explanation,
        "rating": st.session_state.rating,
        "time_spent": round(time_spent, 2),
    }

    st.session_state.evaluations[f"{source_name}_{feature_key}"] = evaluation

    # Move to next
    st.session_state.current_index += 1
    st.rerun()


def skip_feature():
    st.session_state.current_index += 1
    st.rerun()


def calculate_summary():
    """Calculate summary statistics for the evaluations"""
    sources = list(st.session_state.source_data.keys())
    summary = {
        source: {"count": 0, "avg_rating": 0, "total_time": 0} for source in sources
    }

    for eval_id, evaluation in st.session_state.evaluations.items():
        source = evaluation["source"]
        summary[source]["count"] += 1
        summary[source]["avg_rating"] += evaluation["rating"]
        summary[source]["total_time"] += evaluation["time_spent"]

    # Calculate averages
    for source in sources:
        if summary[source]["count"] > 0:
            summary[source]["avg_rating"] = round(
                summary[source]["avg_rating"] / summary[source]["count"], 2
            )
            summary[source]["avg_time_per_feature"] = round(
                summary[source]["total_time"] / summary[source]["count"], 2
            )

    return summary


def display_summary_charts(summary):
    """Display summary charts using Plotly Express"""
    sources = list(summary.keys())

    # Ratings chart
    ratings = [summary[source]["avg_rating"] for source in sources]
    fig_ratings = px.bar(
        x=sources,
        y=ratings,
        title="Average Rating by Source",
        labels={"x": "Source", "y": "Average Rating"},
        color=ratings,
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_ratings)

    # Time chart
    times = [summary[source]["avg_time_per_feature"] for source in sources]
    fig_times = px.bar(
        x=sources,
        y=times,
        title="Average Time per Feature by Source",
        labels={"x": "Source", "y": "Seconds"},
        color=times,
        color_continuous_scale="Cividis",
    )
    st.plotly_chart(fig_times)

    # Count chart
    counts = [summary[source]["count"] for source in sources]
    fig_counts = px.bar(
        x=sources,
        y=counts,
        title="Features Evaluated by Source",
        labels={"x": "Source", "y": "Count"},
        color=counts,
        color_continuous_scale="Plasma",
    )
    st.plotly_chart(fig_counts)


def save_results():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{st.session_state.hookpoint}_blinded_evaluations_{timestamp}.json"

    # Calculate summary statistics
    summary = calculate_summary()

    # Save both evaluations and summary
    output = {"evaluations": st.session_state.evaluations, "summary": summary}

    # Create JSON string
    json_str = json.dumps(output, indent=4)

    # Offer download
    st.download_button(
        label="Download Results JSON",
        data=json_str,
        file_name=filename,
        mime="application/json",
    )

    # Also save to disk
    with open(filename, "w") as f:
        f.write(json_str)

    st.success(f"Results saved to {filename}")


def main():
    st.title("Blinded Feature Interpretability Test")

    # Show setup form if not initialized
    if not st.session_state.initialized:
        with st.form("setup_form"):
            st.header("Test Configuration")

            # Default paths from the original code
            default_checkpoints = [
                "/mnt/ssd-1/lucia/sparsify/checkpoints/SmolLM2-135M-sig-t=2-no-relu/best",
                "/mnt/ssd-1/lucia/sparsify/checkpoints/groupmax-comparison/SmolLM2-135M-skip-adam-5e-3",
            ]

            default_latents = [
                "/mnt/ssd-1/lucia/sparsify/results/sigmoid-best-gemini/latents",
                "/mnt/ssd-1/lucia/delphi/results/SmolLM2-135M-skip-adam-5e-3/latents",
            ]

            # Input fields for configuration
            checkpoint_dirs_str = st.text_area(
                "Sparsify Checkpoint Directories (one per line)",
                "\n".join(default_checkpoints),
            )
            latent_dirs_str = st.text_area(
                "Delphi Latent Directories (one per line)", "\n".join(default_latents)
            )

            hookpoint = st.text_input("Hookpoint", "layers.18.mlp")

            col1, col2 = st.columns(2)
            feature_start = col1.number_input("Feature Range Start", value=0, step=1)
            feature_end = col2.number_input("Feature Range End", value=10, step=1)

            num_examples = st.number_input("Number of Examples", value=20, step=1)

            model_name = st.text_input("Model Name", "HuggingFaceTB/SmolLM2-135M")
            max_latents = st.number_input("Max Latents", value=100, step=10)

            submit_button = st.form_submit_button("Initialize Test")

            if submit_button:
                # Parse input
                checkpoint_dirs = [
                    Path(line.strip())
                    for line in checkpoint_dirs_str.split("\n")
                    if line.strip()
                ]
                latent_dirs = [
                    Path(line.strip())
                    for line in latent_dirs_str.split("\n")
                    if line.strip()
                ]

                if len(checkpoint_dirs) != len(latent_dirs):
                    st.error(
                        "Number of checkpoint directories"
                        "must match number of latent directories"
                    )
                    return

                features_range = (int(feature_start), int(feature_end))

                # Store in session state
                st.session_state.hookpoint = hookpoint

                # Cache latents if needed
                try:
                    with st.spinner("Caching latents if needed..."):
                        cache_latents_if_needed(
                            checkpoint_dirs,
                            latent_dirs,
                            model_name,
                            hookpoint,
                            max_latents,
                        )

                    # Create model dirs dictionary
                    model_dirs = {
                        latent_dir.parent.stem: latent_dir for latent_dir in latent_dirs
                    }

                    # Initialize the test data
                    initialize_test(model_dirs, hookpoint, features_range, num_examples)

                    st.success("Test initialized successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error initializing test: {str(e)}")
                    st.exception(e)
    else:
        # Show the test interface
        has_more_features = display_current_feature()

        if has_more_features:
            # Evaluation form
            with st.form("evaluation_form"):
                st.text_input(
                    "Explanation",
                    key="explanation",
                    help="Describe what you think this feature might represent",
                )

                st.slider(
                    "Interpretability Rating",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="rating",
                    help="1 = Not interpretable, 10 = Clearly interpretable",
                )

                col1, col2 = st.columns(2)
                submit = col1.form_submit_button("Submit")
                skip = col2.form_submit_button("Skip")

                if submit:
                    submit_evaluation()
                if skip:
                    skip_feature()
        else:
            # Show results page
            st.header("Test Results")

            summary = calculate_summary()

            # Display summary stats
            st.subheader("Summary Statistics")
            for source, stats in summary.items():
                st.write(f"Source: {source}")
                st.write(f"- Features evaluated: {stats['count']}")
                st.write(f"- Average rating: {stats['avg_rating']}")

                avg_time = stats["avg_time_per_feature"]
                st.write(f"- Average time per feature: {avg_time} seconds")

            # Visual summary with Plotly charts
            display_summary_charts(summary)

            # Save results button
            save_results()

            # Reset button
            if st.button("Start New Test"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main()
