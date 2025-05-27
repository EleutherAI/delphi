# app.py
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi.__main__ import load_artifacts, populate_cache
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig

from delphi.latents.constructors import constructor
from delphi.latents.loader import ActivationData, LatentData, LatentRecord, TensorBuffer
from delphi.latents.samplers import sampler
from delphi.utils import load_tokenized_data


class SyncLatentDataset:
    """
    Synchronous version of LatentDataset which constructs TensorBuffers
    for each module and latent.
    """

    def __init__(
        self,
        raw_dir: str,
        sampler_cfg: SamplerConfig,
        constructor_cfg: ConstructorConfig,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        modules: Optional[List[str]] = None,
        latents: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize a SyncLatentDataset.

        Args:
            raw_dir: Directory containing raw latent data.
            sampler_cfg: Configuration for sampling examples.
            constructor_cfg: Configuration for constructing examples.
            tokenizer: Tokenizer used to tokenize the data.
            modules: list of module names to include.
            latents: Dictionary of latents per module.
        """
        self.constructor_cfg = constructor_cfg
        self.sampler_cfg = sampler_cfg
        self.buffers: List[TensorBuffer] = []
        self.all_data: Dict[str, Optional[Dict[int, ActivationData]]] = {}
        self.tokens = None

        if modules is None:
            self.modules = os.listdir(raw_dir)
        else:
            self.modules = modules

        # Check that modules exist
        for module in self.modules:
            if not os.path.exists(f"{raw_dir}/{module}"):
                raise FileNotFoundError(
                    f"Could not find {module} in {raw_dir}. "
                    "Please check the parameters passed to the dataset."
                )

        if len(self.modules) == 0:
            raise ValueError("No modules found in the cache folder")

        # Load cache configuration
        cache_config_path = f"{raw_dir}/{self.modules[0]}/config.json"
        if not os.path.exists(cache_config_path):
            raise FileNotFoundError(
                "Each directory in the cache folder must have a config.json file. "
                f"Could not find {cache_config_path} in {raw_dir}."
            )

        with open(cache_config_path, "r") as f:
            cache_config = json.load(f)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(cache_config["model_name"])
        else:
            self.tokenizer = tokenizer

        self.cache_config = cache_config

        # Build dataset
        if latents is None:
            self._build(raw_dir)
        else:
            self._build_selected(raw_dir, latents)

        # Handle neighbours if needed
        if self.constructor_cfg.non_activating_source == "neighbours":
            # path is always going to end with /latents
            split_path = raw_dir.split("/")[:-1]
            neighbours_path = "/".join(split_path) + "/neighbours"
            self.neighbours = self.load_neighbours(
                neighbours_path, self.constructor_cfg.neighbours_type
            )
            # Load all data for neighbours
            self.all_data = self._load_all_data(raw_dir, self.modules)
        else:
            self.neighbours = None

        self.load_tokens()

    def load_tokens(self) -> torch.Tensor:
        """
        Load tokenized data for the dataset.

        Returns:
            torch.Tensor: The tokenized dataset.
        """
        if not hasattr(self, "tokens") or self.tokens is None:
            self.tokens = load_tokenized_data(
                self.cache_config["ctx_len"],
                self.tokenizer,
                self.cache_config["dataset_repo"],
                self.cache_config["dataset_split"],
                self.cache_config["dataset_name"],
                column_name=self.cache_config.get(
                    "dataset_column",
                    self.cache_config.get("dataset_row", "raw_content"),
                ),
            )
        return self.tokens

    def load_neighbours(self, neighbours_path: str, neighbours_type: str) -> Dict:
        """
        Load neighbour data for latents.

        Args:
            neighbours_path: Path to neighbours directory.
            neighbours_type: Type of neighbours to load.

        Returns:
            Dict of neighbours by module and latent index.
        """
        neighbours = {}
        for hookpoint in self.modules:
            with open(
                neighbours_path + f"/{hookpoint}-{neighbours_type}.json", "r"
            ) as f:
                neighbours[hookpoint] = json.load(f)
        return neighbours

    def _edges(self, raw_dir: str, module: str) -> List[Tuple[int, int]]:
        """
        Find edges (start and end indices) for safetensor files.

        Args:
            raw_dir: Directory containing raw latent data.
            module: Module name.

        Returns:
            List of (start, end) tuples for each safetensor file.
        """
        module_dir = Path(raw_dir) / module
        safetensor_files = [f for f in module_dir.glob("*.safetensors")]
        edges = []
        for file in safetensor_files:
            start, end = file.stem.split("_")
            edges.append((int(start), int(end)))
        edges.sort(key=lambda x: x[0])
        return edges

    def _build(self, raw_dir: str):
        """
        Build dataset buffers which load all cached latents.

        Args:
            raw_dir: Directory containing raw latent data.
        """
        for module in self.modules:
            edges = self._edges(raw_dir, module)
            for start, end in edges:
                path = f"{raw_dir}/{module}/{start}_{end}.safetensors"
                tensor_buffer = TensorBuffer(path, module)
                if self.tokens is None:
                    self.tokens = tensor_buffer.tokens
                self.buffers.append(tensor_buffer)
                self.all_data[module] = None
            self.all_data[module] = None

    def _build_selected(
        self,
        raw_dir: str,
        latents: Dict[str, torch.Tensor],
    ):
        """
        Build a dataset buffer which loads only selected latents.

        Args:
            raw_dir: Directory containing raw latent data.
            latents: Dictionary of latents per module.
        """
        for module in self.modules:
            edges = self._edges(raw_dir, module)
            selected_latents = latents[module]
            if len(selected_latents) == 0:
                continue
            if len(edges) == 0:
                raise FileNotFoundError(
                    f"Could not find any safetensor files in {raw_dir}/{module}, "
                    "but latents were selected."
                )
            boundaries = [edges[0][0]] + [edge[1] + 1 for edge in edges]

            bucketized = torch.bucketize(
                selected_latents, torch.tensor(boundaries), right=True
            )
            unique_buckets = torch.unique(bucketized)

            for bucket in unique_buckets:
                mask = bucketized == bucket
                _selected_latents = selected_latents[mask]

                start, end = boundaries[bucket.item() - 1], boundaries[bucket.item()]
                # Adjust end by one as the path avoids overlap
                path = f"{raw_dir}/{module}/{start}_{end-1}.safetensors"
                tensor_buffer = TensorBuffer(
                    path,
                    module,
                    _selected_latents,
                )
                if self.tokens is None:
                    self.tokens = tensor_buffer.tokens
                self.buffers.append(tensor_buffer)
            self.all_data[module] = None

    def _load_all_data(
        self, raw_dir: str, modules: List[str]
    ) -> Dict[str, Dict[int, ActivationData]]:
        """
        For each module, load all locations and activations.

        Args:
            raw_dir: Directory containing raw latent data.
            modules: List of module names to include.

        Returns:
            Dict of activation data by module and latent index.
        """
        all_data = {}
        for buffer in self.buffers:
            module = buffer.module_path
            if module not in all_data:
                all_data[module] = {}
            temp_latents = buffer.latents
            # we remove the filter on latents
            buffer.latents = None
            latents, locations, activations = buffer.load_data_per_latent()
            # we restore the filter on latents
            buffer.latents = temp_latents
            for latent, location, activation in zip(latents, locations, activations):
                all_data[module][latent.item()] = ActivationData(location, activation)
        return all_data

    def _process_sync(self) -> Dict[int, LatentRecord]:
        """
        Process all buffers synchronously.

        Returns:
            Dict mapping latent indices to LatentRecord objects.
        """
        results = {}

        for buffer in self.buffers:
            for latent_data in buffer:
                record = self._process_latent(latent_data)
                if record is not None:
                    results[record.latent.latent_index] = record

        return results

    def _process_latent(self, latent_data: LatentData) -> Optional[LatentRecord]:
        """
        Process a single latent synchronously.

        Args:
            latent_data: Latent data to process.

        Returns:
            Processed latent record or None.
        """
        # This should never happen but we need to type check
        if self.tokens is None:
            raise ValueError("Tokens are not loaded")

        record = LatentRecord(latent_data.latent)

        if self.neighbours is not None:
            record.set_neighbours(
                self.neighbours[latent_data.module][
                    str(latent_data.latent.latent_index)
                ],
            )

        record = constructor(
            record=record,
            activation_data=latent_data.activation_data,
            constructor_cfg=self.constructor_cfg,
            tokens=self.tokens,
            all_data=self.all_data[latent_data.module],
            tokenizer=self.tokenizer,
        )

        # Not enough examples to explain the latent
        if record is None:
            return None

        record = sampler(record, self.sampler_cfg)
        return record

    def load_examples(
        self,
    ) -> Tuple[Dict[str, List], Dict[str, float], PreTrainedTokenizer]:
        """
        Load examples synchronously.

        Returns:
            Tuple of (examples, max_activations, tokenizer).
        """
        all_examples = {}
        maximum_activations = {}

        processed_records = self._process_sync()

        for latent_idx, record in processed_records.items():
            latent_str = str(latent_idx)
            all_examples[latent_str] = record.train
            maximum_activations[latent_str] = record.max_activation

        return all_examples, maximum_activations, self.tokenizer


st.set_page_config(layout="wide")

# Add the directory containing the sync_latent_dataset.py to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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
                constructor_cfg=ConstructorConfig(example_ctx_len=128),
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


def initialize_test(
    model_dirs, hookpoint, num_features, num_examples, max_range_expansion=10_000
):
    # Load examples from both sources
    source_data = {}
    tokenizers = {}

    with st.spinner("Loading examples from sources..."):
        for model_name, raw_dir in model_dirs.items():
            feature_start, feature_end = (0, num_features)
            active_examples = {}
            active_max_acts = {}
            tokenizer = None

            # Keep expanding range until we have enough active features
            # or hit max expansion
            while (
                len(active_examples) < num_features
                and feature_end < feature_start + max_range_expansion
            ):
                # Create current range to check
                current_range = (feature_start, feature_end)

                # Create synchronous dataset for current range
                sync_dataset = SyncLatentDataset(
                    raw_dir,
                    SamplerConfig(n_examples_train=num_examples),
                    ConstructorConfig(),
                    modules=[hookpoint],
                    latents={hookpoint: torch.LongTensor(list(range(*current_range)))},
                )

                examples, max_acts, tokenizer = sync_dataset.load_examples()

                # Filter and add active features
                for feature_key, feature_examples in examples.items():
                    # Skip features we've already processed
                    if feature_key in active_examples:
                        continue

                    # Check if any example has significant activation
                    is_active = False
                    for example in feature_examples:
                        # If any token in any example has activation above threshold
                        if torch.max(torch.abs(example.activations)) > 0:
                            is_active = True
                            break

                    # Keep only active features
                    if is_active:
                        active_examples[feature_key] = feature_examples
                        active_max_acts[feature_key] = max_acts[feature_key]

                    if len(active_examples) >= num_features:
                        break

                # Update progress
                st.write(
                    f"Source {model_name}: Checked features {feature_start}-"
                    f"{feature_end}, found {len(active_examples)} active features"
                )

                # Expand range for next iteration
                feature_start = feature_end
                feature_end += min(100, max_range_expansion // 10)  # Increase in chunks

                # Break if we have enough features
                if len(active_examples) >= num_features:
                    break

            source_data[model_name] = {
                "examples": dict(
                    list(active_examples.items())[:num_features]
                ),  # Limit to target count
                "max_activations": {
                    k: active_max_acts[k]
                    for k in list(active_examples.keys())[:num_features]
                },
            }
            tokenizers[model_name] = tokenizer

            len_examples = len(source_data[model_name]["examples"])
            st.write(
                f"Source {model_name}: Final count - {len_examples} active features"
            )

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
    st.write(
        f"Progress: {st.session_state.current_index}/"
        f"{len(st.session_state.blinded_order)}"
    )

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
    # fig = display_tokens_with_activations(list_tokens, list_activations, tokenizer)
    # st.plotly_chart(fig, use_container_width=True)

    # Display the full text with colored tokens based on activation values
    inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

    for i, (tokens, activations) in enumerate(zip(list_tokens, list_activations)):
        # st.write(f"**Example {i+1}:**")
        example_number_text = f"Example {i+1}: "

        # Create colored HTML for the text
        html = "<div style='font-family: monospace; white-space: pre-wrap;'>"
        html += f"<span style='color: black;'>{example_number_text}</span>"

        for token_id, activation in zip(tokens, activations):
            token_text = (
                inverse_vocab[int(token_id)].replace("Ġ", " ").replace("▁", " ")
            )

            # Color based on activation
            if activation > 0:
                # Blue for positive activations with improved scaling
                intensity = min(255, int(100 + 155 * min(1.0, abs(activation))))
                opacity = min(1.0, 0.3 + 0.7 * min(1.0, abs(activation)))
                bg_color = f"rgba(0, 0, {intensity}, {opacity})"
                text_color = "white" if abs(activation) > 0.4 else "black"
            elif activation < 0:
                # Red for negative activations with improved scaling
                intensity = min(255, int(100 + 155 * min(1.0, abs(activation))))
                opacity = min(1.0, 0.3 + 0.7 * min(1.0, abs(activation)))
                bg_color = f"rgba({intensity}, 0, 0, {opacity})"
                text_color = "white" if abs(activation) > 0.4 else "black"
            else:
                # Neutral
                bg_color = "rgba(220, 220, 220, 0.1)"
                text_color = "black"

            html += (
                f"<span style='background-color: {bg_color}; "
                f"color: {text_color};'>{token_text}</span>"
            )

        html += "</div>"

        # Display the colored HTML
        st.markdown(html, unsafe_allow_html=True)

        # Display only tokens with significant activations in a table
        # activation_threshold = 0.1
        # cleaned_tokens = [
        #     inverse_vocab[int(t)].replace("Ġ", " ").replace("▁", " ")
        #     for t in tokens
        # ]
        # significant_tokens = [(token_text, activations[j])
        #                      for j, token_text in enumerate(cleaned_tokens)
        #                      if abs(activations[j]) > activation_threshold]
        significant_tokens = None

        if significant_tokens:
            # Create DataFrame for display
            df = pd.DataFrame(
                {
                    "Token": [token for token, _ in significant_tokens],
                    "Activation": [f"{act:.3f}" for _, act in significant_tokens],
                    # Placeholder column for color
                    "Color": [""] * len(significant_tokens),
                }
            )

            # Apply color styling based on activation value
            def color_activation(val):
                val = float(val)
                if val > 0:
                    # Blue for positive activations
                    b = min(255, int(200 * val))
                    a = min(1.0, abs(val))
                    return f"background-color: rgba(0, 0, {b}, {a})"
                else:
                    # Red for negative activations
                    r = min(255, int(200 * abs(val)))
                    a = min(1.0, abs(val))
                    return f"background-color: rgba({r}, 0, 0, {a})"

            # Apply the styling
            styled_df = df.style.apply(
                lambda row: ["", "", color_activation(row["Activation"])], axis=1
            )
            st.dataframe(styled_df, use_container_width=True)

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

    # Auto-save current results to JSON file
    auto_save_results()

    if "form_key_counter" not in st.session_state:
        st.session_state.form_key_counter = 1
    else:
        st.session_state.form_key_counter += 1

    # Move to next
    st.session_state.current_index += 1

    st.rerun()


def auto_save_results():
    """Auto-save current evaluation results to a session-specific JSON file"""
    # Create a session-specific filename if it doesn't exist
    if "session_filename" not in st.session_state:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        st.session_state.session_filename = (
            f"{st.session_state.hookpoint}_blinded_evaluations_{timestamp}.json"
        )

    # Calculate summary statistics
    summary = calculate_summary()

    # Save both evaluations and summary
    output = {"evaluations": st.session_state.evaluations, "summary": summary}

    # Save to disk
    with open(st.session_state.session_filename, "w") as f:
        json.dump(output, f, indent=4)


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
                "/mnt/ssd-1/lucia/sparsify/checkpoints/135M-skip-bin/best",
                "/mnt/ssd-1/lucia/sparsify/checkpoints/groupmax-comparison/SmolLM2-135M-skip-adam-5e-3",
            ]

            default_latents = [
                "/mnt/ssd-1/lucia/sparsify/results/135M-skip-binary/latents",
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

            num_features = st.number_input("Number of Features", value=10, step=1)

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
                    initialize_test(model_dirs, hookpoint, num_features, num_examples)
                    st.session_state.form_key_counter = 0
                    st.success("Test initialized successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error initializing test: {str(e)}")
                    st.exception(e)
    else:
        # Show the test interface
        has_more_features = display_current_feature()

        if has_more_features:
            if "form_key_counter" not in st.session_state:
                st.session_state.form_key_counter = 0

            form_key = f"evaluation_form_{st.session_state.form_key_counter}"

            with st.form(form_key):
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
