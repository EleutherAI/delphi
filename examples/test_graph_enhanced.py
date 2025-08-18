import asyncio
import json
import math
import os
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import orjson
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Import Delphi components
from delphi.clients import Offline, OpenRouter
from delphi.config import CacheConfig, ConstructorConfig, SamplerConfig
from delphi.explainers import GraphExplainer
from delphi.latents import LatentCache, LatentDataset
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.sparse_coders.load_sparsify import load_sparsify_hooks
from delphi.utils import load_tokenized_data


def cantor_decode(num: int) -> Tuple[int, int]:
    """Decode a Cantor pairing back to (layer, feature)."""
    w = math.floor((math.sqrt(8 * num + 1) - 1) / 2)
    t = (w * w + w) // 2
    y = num - t
    x = w - y
    return x, y


def load_graph(json_path: str) -> Dict:
    """Load and parse the attribution graph JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def convert_node_names(
    nodes: List[Dict], links: List[Dict]
) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Convert node names to layer_featureidx format with position-aware consolidation.

    Consolidates multiple nodes with same layer+feature but different ctx_idx
    into single logical feature while preserving position-specific influence data.

    Returns:
        - name_mapping: {original_id: consolidated_feature_name}
        - node_info: {consolidated_feature_name: consolidated_data}
    """
    name_mapping = {}
    consolidated_features = {}  # {layer_feature: [list of nodes]}

    # Group nodes by layer+feature
    for node in nodes:
        if node["feature_type"] == "cross layer transcoder":
            layer = int(node["layer"])
            feature_cantor = int(node["feature"])

            # Decode Cantor pairing to get original (layer, feature)
            decoded_layer, feature = cantor_decode(feature_cantor)

            layer_feature = f"{layer}_{feature}"
            name_mapping[node["node_id"]] = layer_feature

            if layer_feature not in consolidated_features:
                consolidated_features[layer_feature] = []
            consolidated_features[layer_feature].append(node)

    # Create consolidated node info
    node_info = {}
    for layer_feature, node_list in consolidated_features.items():
        # Extract position-specific influence data
        influence_by_position = {}
        positions = []
        original_ids = []

        for node in node_list:
            ctx_idx = node.get("ctx_idx", 0)
            influence = node.get("influence", 0.0)

            influence_by_position[ctx_idx] = influence
            positions.append(ctx_idx)
            original_ids.append(node["node_id"])

        # Use first node for basic layer/feature info
        primary_node = node_list[0]
        layer = int(primary_node["layer"])
        decoded_layer, feature = cantor_decode(int(primary_node["feature"]))

        node_info[layer_feature] = {
            "original_ids": original_ids,
            "layer": layer,
            "feature": feature,
            "positions": sorted(positions),
            "influence_by_position": influence_by_position,
            "max_influence": (
                max(influence_by_position.values()) if influence_by_position else 0.0
            ),
            "total_influence": sum(influence_by_position.values()),
            "node_count": len(node_list),
        }

    return name_mapping, node_info


def build_adjacency_list(
    name_mapping: Dict[str, str], links: List[Dict]
) -> Dict[str, List[str]]:
    """
    Build adjacency list mapping consolidated features to their parents.

    Since features are now consolidated by layer+feature, multiple links may exist
    between the same logical parent-child pair. We merge these by deduplicating.

    Returns:
        adjacency_list: {child_feature: [parent1, parent2, ...]}
    """
    # Track unique parent-child relationships
    dependencies = defaultdict(set)

    for link in links:
        source_id = link["source"]
        target_id = link["target"]

        # Only process links between transcoder nodes
        if source_id in name_mapping and target_id in name_mapping:
            source_name = name_mapping[source_id]
            target_name = name_mapping[target_id]

            # target depends on source (source is parent of target)
            # Use set to automatically handle duplicates from consolidation
            dependencies[target_name].add(source_name)

    # Convert sets back to lists
    adjacency_list = {child: list(parents) for child, parents in dependencies.items()}
    return adjacency_list


def topological_sort(adjacency_list: Dict[str, List[str]]) -> List[str]:
    """
    Topological sort using Kahn's algorithm with error handling.

    Returns:
        List of nodes in dependency order (parents before children)

    Raises:
        ValueError: If cycles detected in the graph
    """
    # Get all nodes
    all_nodes = set()
    for child, parents in adjacency_list.items():
        all_nodes.add(child)
        all_nodes.update(parents)

    # Calculate in-degrees
    in_degree = {node: 0 for node in all_nodes}
    for child, parents in adjacency_list.items():
        in_degree[child] = len(parents)

    # Initialize queue with nodes having no dependencies
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # Find children of this node
        for child, parents in adjacency_list.items():
            if node in parents:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    # Check for cycles with better error messaging
    if len(result) != len(all_nodes):
        remaining = [node for node in all_nodes if node not in result]
        remaining_with_deps = []
        for node in remaining:
            deps = adjacency_list.get(node, [])
            unresolved_deps = [dep for dep in deps if dep in remaining]
            remaining_with_deps.append(f"{node} (deps: {unresolved_deps})")

        raise ValueError(
            f"Cycle detected in graph. Could not resolve {len(remaining)} nodes: "
            f"{', '.join(remaining_with_deps[:5])}{'...' if len(remaining) > 5 else ''}"
        )

    return result


class ExplanationPipeline:
    """Manages the explanation workflow with topological ordering."""

    def __init__(
        self,
        topo_order: List[str],
        adjacency_list: Dict[str, List[str]],
        node_info: Dict[str, Dict],
    ):
        self.topo_order = topo_order
        self.adjacency_list = adjacency_list
        self.node_info = node_info
        self.explained = set()
        self.queue = deque()

        # Initialize queue with nodes that have no dependencies
        self._update_queue()

    def _update_queue(self):
        """Update queue with nodes whose dependencies are satisfied."""
        for node in self.topo_order:
            if node not in self.explained and node not in self.queue:
                parents = self.adjacency_list.get(node, [])
                if all(parent in self.explained for parent in parents):
                    self.queue.append(node)

    def get_ready_nodes(self, batch_size: int = 1) -> List[str]:
        """Get next batch of nodes ready for explanation."""
        ready = []
        for _ in range(min(batch_size, len(self.queue))):
            if self.queue:
                ready.append(self.queue.popleft())
        return ready

    def mark_completed(self, nodes: List[str]):
        """Mark nodes as explained and update queue."""
        for node in nodes:
            self.explained.add(node)

        self._update_queue()

    def is_complete(self) -> bool:
        """Check if all nodes have been explained."""
        return len(self.explained) == len(self.topo_order)

    def get_status(self) -> Dict:
        """Get current status of the pipeline."""
        return {
            "total_nodes": len(self.topo_order),
            "explained": len(self.explained),
            "queued": len(self.queue),
            "remaining": len(self.topo_order) - len(self.explained),
            "progress": (
                len(self.explained) / len(self.topo_order) if self.topo_order else 0.0
            ),
        }

    def create_latent_dict_for_nodes(
        self, ready_nodes: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Create a latent_dict containing only the specified ready nodes."""
        latent_dict = {}

        for node_name in ready_nodes:
            if node_name in self.node_info:
                layer = self.node_info[node_name]["layer"]
                feature = self.node_info[node_name]["feature"]
                module = f"layers.{layer}.mlp"

                if module not in latent_dict:
                    latent_dict[module] = []
                latent_dict[module].append(feature)

        # Convert to tensors
        for key in latent_dict.keys():
            latent_dict[key] = torch.tensor(latent_dict[key])

        return latent_dict


def add_parent_connections_to_all_nodes(
    graph_json_path: str, json_directory: str, force_recreate: bool = False
) -> None:
    """
    Add parent connection information to all node JSON files in a directory.

    This function efficiently processes all nodes in an attribution graph and adds
    parent connection information to their corresponding JSON files.
    The parent_connections field contains a list of
    (parent_filename, attribution_strength) tuples.

    Args:
        graph_json_path: Path to the attribution graph JSON file
        json_directory: path containing the feature json files
        force_recreate: If True, recreate parent_connections even if it already exists

    Files must be named using cantor encoding of (layer, feature_idx).
    If a JSON file doesn't exist, it creates a new one with basic structure.

    Example usage:
        add_parent_connections_to_all_nodes(
            "graph.json",
            "node_jsons/",
            force_recreate=False
        )
    """

    def load_graph(json_path: str) -> Dict:
        """Load and parse the attribution graph JSON."""
        with open(json_path, "r") as f:
            return json.load(f)

    print(f"Loading attribution graph from {graph_json_path}...")
    graph = load_graph(graph_json_path)

    # Mapping from cantor-encoded features to (layer, feature) and node IDs
    print("Building node mappings...")
    cantor_to_layer_feature = {}  # {cantor_encoded: (layer, feature)}
    cantor_to_node_ids = defaultdict(set)  # {cantor_encoded: {node_id1, node_id2, ...}}

    for node in graph["nodes"]:
        if node["feature_type"] == "cross layer transcoder":
            layer = int(node["layer"])
            feature_cantor = int(node["feature"])
            node_id = node["node_id"]

            # Decode Cantor pairing to get original (layer, feature)
            decoded_layer, feature = cantor_decode(feature_cantor)

            # Store mappings - use the layer from the node and feature from decoding
            cantor_to_layer_feature[feature_cantor] = (layer, feature)
            cantor_to_node_ids[feature_cantor].add(node_id)

    # Build parent-child relationships for all nodes
    print("Computing parent-child relationships...")
    parent_relationships = defaultdict(
        lambda: defaultdict(float)
    )  # {child_cantor: {parent_cantor: max_weight}}

    for link in graph["links"]:
        source_id = link["source"]
        target_id = link["target"]
        weight = link.get("weight", 0.0)

        # Find cantor encodings for source and target
        source_cantor = None
        target_cantor = None

        for cantor, node_ids in cantor_to_node_ids.items():
            if source_id in node_ids:
                source_cantor = cantor
            if target_id in node_ids:
                target_cantor = cantor

        # Record parent-child relationship if both are transcoder nodes
        if source_cantor is not None and target_cantor is not None:
            # Take maximum weight for consolidated links
            current_weight = parent_relationships[target_cantor][source_cantor]
            parent_relationships[target_cantor][source_cantor] = max(
                current_weight, weight
            )

    # Convert parent relationships to filename format
    print("Converting to filename format...")
    parent_connections_by_cantor = {}

    for child_cantor, parent_weights in parent_relationships.items():
        parent_connections = []

        for parent_cantor, weight in parent_weights.items():
            if parent_cantor in cantor_to_layer_feature:
                layer, feature_idx = cantor_to_layer_feature[parent_cantor]
                filename = f"layers.{layer}.mlp_latent{feature_idx}.txt"
                parent_connections.append((filename, weight))

        parent_connections_by_cantor[child_cantor] = parent_connections

    # Process all JSON files in the directory
    json_dir = Path(json_directory)
    json_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing JSON files in {json_directory}...")
    files_processed = 0
    files_created = 0
    files_skipped = 0

    # Get all unique cantor encodings from the graph
    all_cantor_encodings = set(cantor_to_layer_feature.keys())

    for cantor_encoded in all_cantor_encodings:
        json_filename = f"{cantor_encoded}.json"
        json_path = json_dir / json_filename

        # Load existing JSON or create new structure
        try:
            if json_path.exists():
                with open(json_path, "r") as f:
                    node_data = json.load(f)
            else:
                print(f"JSON file {json_filename} does not exist, creating new file")
                # Create basic structure
                layer, feature_idx = cantor_to_layer_feature[cantor_encoded]
                node_data = {
                    "index": feature_idx,
                    "layer": layer,
                    "cantor_encoding": cantor_encoded,
                }
                files_created += 1
        except json.JSONDecodeError as e:
            print(f"Malformed JSON in {json_filename}: {e}. Creating new file.")
            layer, feature_idx = cantor_to_layer_feature[cantor_encoded]
            node_data = {
                "index": feature_idx,
                "layer": layer,
                "cantor_encoding": cantor_encoded,
            }
            files_created += 1

        # Check if parent_connections already exists and force_recreate is False
        if "parent_connections" in node_data and not force_recreate:
            files_skipped += 1
            continue

        # Add parent connections
        parent_connections = parent_connections_by_cantor.get(cantor_encoded, [])
        node_data["parent_connections"] = parent_connections

        # Save updated JSON
        with open(json_path, "w") as f:
            json.dump(node_data, f, indent=2)

        files_processed += 1

    print("Processing complete!")
    print(f"  Files processed: {files_processed}")
    print(f"  Files created: {files_created}")
    print(f"  Files skipped (already had parent_connections): {files_skipped}")
    print(f"  Total unique features: {len(all_cantor_encodings)}")


def load_model_with_fallback(
    model_path: str, load_in_8bit: bool = False, hf_token: Optional[str] = None
):
    """
    Load model from local directory if it exists, otherwise from HuggingFace.

    Args:
        model_path: Path to model (local directory or HF model name)
        load_in_8bit: Whether to load model in 8-bit quantization
        hf_token: HuggingFace token for private models

    Returns:
        PreTrainedModel: The loaded model
    """
    local_path = Path(model_path)

    # Determine data type
    if load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    # Set up quantization config if needed
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    if local_path.exists() and local_path.is_dir():
        print(f"   Loading model from local directory: {local_path}")
        try:
            model = AutoModel.from_pretrained(
                local_path,
                device_map={"": "cuda"},
                quantization_config=quantization_config,
                torch_dtype=dtype,
                token=hf_token,
            )
            print("   ✅ Successfully loaded model from local directory")
            return model
        except Exception as e:
            print(f"   ⚠️  Failed to load from local directory: {e}")
            print("   🔄 Falling back to HuggingFace...")

    print(f"   Loading model from HuggingFace: {model_path}")
    try:
        model = AutoModel.from_pretrained(
            model_path,
            device_map={"": "cuda"},
            quantization_config=quantization_config,
            torch_dtype=dtype,
            token=hf_token,
        )
        print("   ✅ Successfully loaded model from HuggingFace")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from both local and HuggingFace: {e}")


def load_tokenizer_with_fallback(model_path: str, hf_token: Optional[str] = None):
    """
    Load tokenizer from local directory if it exists, otherwise from HuggingFace.

    Args:
        model_path: Path to model (local directory or HF model name)
        hf_token: HuggingFace token for private models

    Returns:
        AutoTokenizer: The loaded tokenizer
    """
    local_path = Path(model_path)

    if local_path.exists() and local_path.is_dir():
        print(f"   Loading tokenizer from local directory: {local_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path, token=hf_token)
            print("   ✅ Successfully loaded tokenizer from local directory")
            return tokenizer
        except Exception as e:
            print(f"   ⚠️  Failed to load from local directory: {e}")
            print("   🔄 Falling back to HuggingFace...")

    print(f"   Loading tokenizer from HuggingFace: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        print("   ✅ Successfully loaded tokenizer from HuggingFace")
        return tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer from both local and HuggingFace: {e}"
        )


def generate_cache_from_model(
    base_model_path: str,
    sparse_model_path: str,
    hookpoints: list[str],
    cache_dir: str,
    dataset_repo: str = "EleutherAI/fineweb-edu-dedup-10b",
    dataset_split: str = "train[:1%]",
    dataset_column: str = "text",
    n_tokens: int = 1_000_000,
    batch_size: int = 16,
    ctx_len: int = 256,
    n_splits: int = 5,
    load_in_8bit: bool = False,
    filter_bos: bool = True,
    hf_token: Optional[str] = None,
    seed: int = 22,
):
    """
    Generate activation cache from a base model and sparse autoencoder.

    Args:
        base_model_path: Path to base model (local directory or HF model name)
        sparse_model_path: Path to sparse model/SAE (local directory or HF model name)
        hookpoints: List of hookpoints to attach SAEs to (e.g., ["layers.5"])
        cache_dir: Output directory for the cache
        dataset_repo: HuggingFace dataset repository
        dataset_split: Dataset split to use
        dataset_column: Column name containing text data
        n_tokens: Number of tokens to cache
        batch_size: Batch size for processing
        ctx_len: Context length for each sequence
        n_splits: Number of files to split cache into
        load_in_8bit: Whether to load model in 8-bit quantization
        filter_bos: Whether to filter out BOS tokens
        hf_token: HuggingFace token for private models
        seed: Random seed
    """

    print("🏗️  Starting cache generation...")

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_path = cache_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    print(f" Loading base model: {base_model_path}")
    # Load base model
    model = load_model_with_fallback(base_model_path, load_in_8bit, hf_token)

    print(" Loading tokenizer...")
    # Load tokenizer
    tokenizer = load_tokenizer_with_fallback(base_model_path, hf_token)

    print(f" Loading sparse autoencoders: {sparse_model_path}")
    print(f"   Hookpoints: {hookpoints}")
    # Load sparse autoencoders and create encoding hooks
    hookpoint_to_sparse_encode, transcode = load_sparsify_hooks(
        model,
        sparse_model_path,
        hookpoints,
        compile=True,
    )

    print(f"   Loading and tokenizing dataset: {dataset_repo}")
    print(f"   Split: {dataset_split}, Column: {dataset_column}")
    print(f"   Context length: {ctx_len}, Tokens to process: {n_tokens:,}")
    # Load and tokenize dataset
    tokens = load_tokenized_data(
        ctx_len,
        tokenizer,
        dataset_repo,
        dataset_split,
        dataset_name="",  # Usually empty for most datasets
        column_name=dataset_column,
        seed=seed,
    )

    # Filter BOS tokens if requested
    if filter_bos:
        if tokenizer.bos_token_id is None:
            print("   Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            print("   Filtering BOS tokens...")
            original_tokens = len(tokens.flatten())
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, ctx_len)
            filtered_tokens = len(tokens.flatten())
            print(f"   Filtered {original_tokens - filtered_tokens:,} BOS tokens")

    print("⚡ Creating activation cache...")
    print(f"   Batch size: {batch_size}")
    print(f"   Transcode mode: {transcode}")
    # Create and populate cache
    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=batch_size,
        transcode=transcode,
        log_path=log_path,
    )

    # Run the caching process
    cache.run(n_tokens, tokens)

    print("  Generating cache statistics...")
    # Generate statistics
    cache.generate_statistics_cache()

    print("  Saving cache to disk...")
    print(f"   Cache directory: {cache_dir}")
    print(f"   Number of splits: {n_splits}")
    # Save cache splits
    cache.save_splits(
        n_splits=n_splits,
        save_dir=cache_path,
    )

    # Save configuration
    cache_cfg = CacheConfig(
        dataset_repo=dataset_repo,
        dataset_split=dataset_split,
        dataset_column=dataset_column,
        batch_size=batch_size,
        cache_ctx_len=ctx_len,
        n_tokens=n_tokens,
        n_splits=n_splits,
    )
    cache.save_config(save_dir=cache_path, cfg=cache_cfg, model_name=base_model_path)

    print("  Cache generation complete!")
    print(f"   Cache saved to: {cache_dir}")
    print(f"   Hookpoints processed: {list(hookpoint_to_sparse_encode.keys())}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()


async def generate_explanations_from_cache(
    cache_dir: str,
    output_dir: str,
    model_name: str = "google/gemma-2-2b-it",
    hookpoints: list[str] = ["layers.5"],
    max_latents: int = 10,
    explainer_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    use_openrouter: bool = False,
):
    """
    Generate explanations from a pre-existing activation cache.

    Args:
        cache_dir: Path to the directory containing cached activations
        output_dir: Path where explanations will be saved
        model_name: Name of the base model (used for tokenizer)
        hookpoints: List of hookpoints to process
        max_latents: Maximum number of latents to explain per hookpoint
        explainer_model: Model to use for generating explanations
        use_openrouter: Whether to use OpenRouter API (True) or local model (False)
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for {model_name}...")
    # Load tokenizer (needed to convert tokens back to text)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Creating dataset from cache at {cache_dir}...")
    # Create dataset from cached activations
    # These configs control how examples are constructed and sampled
    sampler_cfg = SamplerConfig(
        n_examples_train=30,  # Number of examples to use for explanation generation
        n_quantiles=4,  # Number of activation quantiles to sample from
        train_type="quantiles",  # Sample across different activation strengths
        test_type="quantiles",
    )

    constructor_cfg = ConstructorConfig(
        example_ctx_len=32,  # Length of each example (in tokens)
        min_examples=50,  # Minimum activatioons needed to explain a latent
        n_non_activating=0,  # Number of non-activating examples
        center_examples=True,  # Center examples on the activating token
        non_activating_source="random",
    )

    # Load attribution graph and extract features to explain
    logits_dir = (
        "/mnt/ssd-1/soar-automated_interpretability/graphs/"
        "pawan/delphi-env/attribute/attribution-graphs-frontend/"
        "features/gemmascope-transcoders-sparsify-1m"
    )
    graph_dir = (
        "/mnt/ssd-1/soar-automated_interpretability/graphs/"
        "pawan/delphi-env/attribute/attribution-graphs-frontend/"
        "graph_data/gemma-bball.json"
    )

    start = time.time()
    add_parent_connections_to_all_nodes(graph_dir, logits_dir)
    print(f"Parent connections added in {time.time() - start:.2f} seconds")

    print(f"Loading attribution graph from {graph_dir}...")
    graph_data = load_graph(graph_dir)

    # Extract prompt for context
    prompt = graph_data["metadata"]["prompt"]
    print(f"Graph prompt: {prompt}")

    # Process graph to get node names and topology
    print("Processing attribution graph nodes...")
    name_mapping, node_info = convert_node_names(
        graph_data["nodes"], graph_data["links"]
    )
    print(f"Found {len(name_mapping)} transcoder nodes")

    # Build adjacency list and topological order
    adjacency_list = build_adjacency_list(name_mapping, graph_data["links"])
    topo_order = topological_sort(adjacency_list)
    print(f"Computed topological order: {len(topo_order)} nodes")

    # Create explanation pipeline
    explanation_pipeline = ExplanationPipeline(topo_order, adjacency_list, node_info)
    print(f"Pipeline initialized with {len(explanation_pipeline.queue)} ready nodes")

    # Create latent dict from the graph nodes (limit by max_latents if specified)
    latent_dict = {}
    nodes_to_process = topo_order[:max_latents] if max_latents else topo_order

    for node_name in nodes_to_process:
        if node_name in node_info:
            layer = node_info[node_name]["layer"]
            feature = node_info[node_name]["feature"]
            module = f"layers.{layer}.mlp"

            if module not in latent_dict:
                latent_dict[module] = []
            latent_dict[module].append(feature)

    # Convert to tensors
    for key in latent_dict.keys():
        latent_dict[key] = torch.tensor(latent_dict[key])

    print(
        f"LatentData created with"
        f"{sum(len(v) for v in latent_dict.values())} features and"
        f"{len(latent_dict)} modules"
    )
    # Note: LatentDataset will be created per iteration with only the ready nodes

    print(f"Setting up explainer model: {explainer_model}...")
    # Create client for the explainer model
    if use_openrouter:
        if "OPENROUTER_API_KEY" not in os.environ:
            raise ValueError(
                "set OPENROUTER_API_KEY environment variable to use OpenRouter"
            )

        client = OpenRouter(explainer_model, api_key=os.environ["OPENROUTER_API_KEY"])
        print("   Using OpenRouter API")
    else:
        client = Offline(
            explainer_model,
            max_memory=0.9,  # Use 80% of available GPU memory
            max_model_len=4096,  # Context length for the explainer model
            num_gpus=torch.cuda.device_count(),  # Use all available GPUs
        )
        print(f"   Using local model with {torch.cuda.device_count()} GPU(s)")

    # Create the explainer
    explainer = GraphExplainer(
        client,
        threshold=0.5,  # Activation threshold for highlighting tokens
        verbose=True,  # Print explanations as they're generated
        graph_info_path=graph_dir,
        explanations_dir=output_path,
        graph_prompt=prompt,
        max_parent_explanations=2,
        max_examples=20,
        cot=True,  # Use chain-of-thought prompting
    )

    # Create pipeline for this iteration
    def explanation_postprocess(result):
        """Save each explanation to a separate file and track it."""
        output_file = output_path / f"{result.record.latent}.txt"
        with open(output_file, "wb") as f:
            f.write(orjson.dumps(result.explanation))
        return result

    explainer_pipe = Pipe(
        process_wrapper(explainer, postprocess=explanation_postprocess)
    )

    print("Starting pipeline-based explanation generation...")

    # Create a pipeline-based explanation process
    async def explain_with_pipeline():
        """Process features in topological order using delphi Pipeline architecture."""
        iteration = 0

        while not explanation_pipeline.is_complete():
            iteration += 1
            # Get all ready nodes (don't limit by batch_size as requested)
            ready_nodes = explanation_pipeline.get_ready_nodes(
                batch_size=len(explanation_pipeline.queue)
            )

            if not ready_nodes:
                print(
                    "No ready nodes, but pipeline not complete - checking for issues..."
                )
                status = explanation_pipeline.get_status()
                print(f"Status: {status}")
                break

            print(f"Iteration {iteration}: Processing {len(ready_nodes)} features")

            # Create latent_dict for this iteration's nodes
            iteration_latent_dict = explanation_pipeline.create_latent_dict_for_nodes(
                ready_nodes
            )
            print("Created latent dict")

            # Create LatentDataset for this batch
            iteration_dataset = LatentDataset(
                raw_dir=cache_dir,
                sampler_cfg=sampler_cfg,
                constructor_cfg=constructor_cfg,
                modules=list(iteration_latent_dict.keys()),
                latents=iteration_latent_dict,
                tokenizer=tokenizer,
                logits_directory=logits_dir,
            )
            pipeline = Pipeline(iteration_dataset, explainer_pipe)

            # Run pipeline
            await pipeline.run(max_concurrent=10)

            # Mark nodes as completed
            explanation_pipeline.mark_completed(ready_nodes)

            # Show progress
            status = explanation_pipeline.get_status()
            print(
                f"Progress: {status['explained']}/{status['total_nodes']} "
                f"({status['progress']:.1%}) - {status['queued']} queued"
            )

    # Run the pipeline-based explanation process
    await explain_with_pipeline()

    print(f"Explanation generation complete! Results saved to {output_dir}")
    print(f"   Generated {len(all_explanations)} explanations")
    print("   Processing followed topological order to ensure dependencies")


async def main():
    """
    Main function demonstrating complete end-to-end workflow.

    This script can either:
    1. Generate a cache from scratch using a base model + SAE
    2. Generate explanations from an existing cache
    3. Do both in sequence

    Modify the parameters below to match your setup.
    """

    # ====== CONFIGURATION SECTION ======
    # Set these parameters to match your models and desired setup

    # MODELS BEING ANALYZED (the subject of interpretability)
    base_model_path = "google/gemma-2-2b-it"
    sparse_model_path = (
        "/mnt/ssd-1/soar-automated_interpretability/graphs/pawan/"
        "delphi-env/models/gemmascope-transcoders-sparsify-1m"
    )

    # Hookpoints to process (which layers to attach replacement models to)
    hookpoints = ["layers.5.mlp"]

    # EXPLAINER MODEL (the model that generates explanations)
    explainer_model = "meta-llama/Llama-3.1-8B-Instruct"
    # explainer_model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    # explainer_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    explainer_type = "default"
    # explainer_type = "graph"

    # Paths
    cache_dir = (
        "/mnt/ssd-1/soar-automated_interpretability/graphs/"
        "pawan/delphi-env/delphi/results/gemma2b/latents"
    )
    output_dir = (
        f"results/graph_enhanced_{explainer_model.split('/')[-1]}_{explainer_type}"
    )

    # What to do - set these flags based on what you want to run
    GENERATE_CACHE = False  # Set to True to create cache from models
    GENERATE_EXPLANATIONS = True  # Set to True to generate explanations

    # Cache generation parameters
    cache_params = {
        "n_tokens": 500_000,  # Number of tokens to cache (smaller for demo)
        "batch_size": 8,  # Batch size (adjust based on GPU memory)
        "ctx_len": 256,  # Context length
        "dataset_repo": "EleutherAI/fineweb-edu-dedup-10b",
        "dataset_split": "train[:1%]",
        "load_in_8bit": False,  # Set to True if running out of GPU memory
        "filter_bos": True,
        "hf_token": None,  # Set if using private models
    }

    # Explanation generation parameters
    explanation_params = {
        "max_latents": 20,  # Number of features to explain
        "explainer_model": explainer_model,  # Model that writes the explanations
        "use_openrouter": False,  # Set to True to use OpenRouter API
    }

    # ====== EXECUTION SECTION ======

    if GENERATE_CACHE:
        print("🏗️  Phase 1: Generating activation cache...")
        print(f"   Base model: {base_model_path}")
        print(f"   Sparse model: {sparse_model_path}")
        print(f"   Hookpoints: {hookpoints}")
        print(f"   Output: {cache_dir}")

        try:
            generate_cache_from_model(
                base_model_path=base_model_path,
                sparse_model_path=sparse_model_path,
                hookpoints=hookpoints,
                cache_dir=cache_dir,
                **cache_params,
            )
            print("✅ Cache generation completed successfully!")
        except Exception as e:
            print(f"❌ Cache generation failed: {e}")
            print("📋 Full stack trace:")
            traceback.print_exc()
            if not GENERATE_EXPLANATIONS:
                return
            print("Continuing to explanation generation if cache already exists...")

    if GENERATE_EXPLANATIONS:
        # Check if cache exists
        if not Path(cache_dir).exists():
            print(f"❌ Cache directory not found: {cache_dir}")
            return

        print("\nPhase 2: Generating explanations from cache...")
        print(f"   Cache directory: {cache_dir}")
        print(f"   Output directory: {output_dir}")

        try:
            await generate_explanations_from_cache(
                cache_dir=cache_dir,
                output_dir=output_dir,
                model_name=base_model_path,  # Use same model for tokenizer
                hookpoints=hookpoints,
                **explanation_params,
            )
            print("Explanation generation completed successfully!")
        except Exception as e:
            print(f"❌ Explanation generation failed: {e}")
            print("📋 Full stack trace:")
            traceback.print_exc()
            return

    if not GENERATE_CACHE and not GENERATE_EXPLANATIONS:
        print("set either GENERATE_CACHE=True or GENERATE_EXPLANATIONS=True")
        return

    print("\n  Script completed successfully!")
    if GENERATE_CACHE:
        print(f"   Cache saved to: {cache_dir}")
    if GENERATE_EXPLANATIONS:
        print(f"   Explanations saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
