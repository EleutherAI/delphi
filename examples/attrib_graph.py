import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import torch


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


def get_prompt(json_path):
    """Extract prompt information from a JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            return data.get("metadata", {}).get("prompt", "")
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return ""
