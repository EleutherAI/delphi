import argparse
import asyncio
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import torch
import yaml
from attrib_graph import (
    ExplanationPipeline,
    add_parent_connections_to_all_nodes,
    build_adjacency_list,
    convert_node_names,
    load_graph,
    topological_sort,
)
from transformers import AutoTokenizer

# Import Delphi components
from delphi.clients import Offline, OpenRouter
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.explainers import (
    DefaultExplainer,
    GraphExplainer,
)
from delphi.latents import LatentDataset
from delphi.pipeline import Pipe, Pipeline, process_wrapper


@dataclass
class GraphExplainerConfig:
    """Simplified configuration for graph explainer experiments."""

    # Model settings
    model_name: str = "google/gemma-2-2b-it"
    sparse_model: str = "../models/gemmascope-sparsify-1m"
    explainer_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    hookpoints: List[str] = field(default_factory=lambda: ["layers.5"])

    # Experiment identification
    prompt_name: str = "bees"
    experiment_name: str = "experiment_1"

    # Directory structure
    base_dir: str = "."
    results_dir: str = "results"
    cache_name: str = "gemma2b_transcoder-sparsify-1m_cache"

    # Explainer settings
    explainer_type: str = "graph"
    threshold: float = 0.3
    temperature: float = 0.0
    max_latents: int = 50
    max_examples: int = 20
    max_parent_explanations: int = 2
    verbose: bool = False
    activations: bool = True
    cot: bool = True
    top_logits: bool = False
    bot_logits: bool = False
    graph_prompt: bool = False

    # Client settings
    client_type: str = "offline"
    max_memory: float = 0.9
    max_model_len: int = 4096

    # Sampling settings
    n_examples_train: int = 10
    n_examples_test: int = 0
    n_quantiles: int = 10
    train_type: str = "quantiles"
    test_type: str = "quantiles"

    # Constructor settings
    example_ctx_len: int = 16
    min_examples: int = 10
    n_non_activating: int = 0
    center_examples: bool = True
    non_activating_source: str = "random"

    @property
    def prompt_dir(self) -> str:
        """Directory for this prompt's graphs and shared files."""
        return f"{self.results_dir}/{self.prompt_name}"

    @property
    def experiment_dir(self) -> str:
        """Directory for this specific experiment's outputs."""
        return f"{self.prompt_dir}/{self.experiment_name}"

    @property
    def cache_dir(self) -> str:
        """Directory for model cache."""
        return f"{self.results_dir}/{self.cache_name}"

    @property
    def attribute_graph_path(self) -> str:
        """Path to attribution graph file."""
        return f"{self.prompt_dir}/attribute.json"

    @property
    def neuronpedia_graph_path(self) -> str:
        """Path to neuronpedia graph file."""
        return f"{self.prompt_dir}/neuronpedia.json"

    @property
    def logits_dir(self) -> str:
        """Directory for logits files."""
        return f"{self.prompt_dir}/logits"

    def resolve_paths(self) -> "GraphExplainerConfig":
        """Convert relative paths to absolute based on base_dir."""
        resolved = copy.deepcopy(self)
        base = Path(self.base_dir).resolve()
        resolved.results_dir = str(base / self.results_dir)
        return resolved

    @classmethod
    def from_yaml(cls, path: str) -> "GraphExplainerConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


async def generate_explanations_from_cache(config: GraphExplainerConfig):
    """
    Generate explanations from a pre-existing activation cache
    using the simplified config.
    """

    # Create output directory
    output_path = Path(config.experiment_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create configs from dataclass fields
    sampler_cfg = SamplerConfig(
        n_examples_train=config.n_examples_train,
        n_examples_test=config.n_examples_test,
        n_quantiles=config.n_quantiles,
        train_type=config.train_type,
        test_type=config.test_type,
    )
    constructor_cfg = ConstructorConfig(
        example_ctx_len=config.example_ctx_len,
        min_examples=config.min_examples,
        n_non_activating=config.n_non_activating,
        center_examples=config.center_examples,
        non_activating_source=config.non_activating_source,
    )

    # Create client
    if config.client_type == "offline":
        client = Offline(
            config.explainer_model,
            max_memory=config.max_memory,
            max_model_len=config.max_model_len,
            num_gpus=torch.cuda.device_count(),
        )
    elif config.client_type == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set")
        client = OpenRouter(config.explainer_model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported client type: {config.client_type}")

    # Create explainer
    if config.explainer_type == "graph":
        explainer = GraphExplainer(
            client=client,
            threshold=config.threshold,
            verbose=config.verbose,
            activations=config.activations,
            cot=config.cot,
            temperature=config.temperature,
            max_parent_explanations=config.max_parent_explanations,
            max_examples=config.max_examples,
            graph_info_path=config.attribute_graph_path,
            explanations_dir=config.experiment_dir,
            graph_prompt="",  # Will be loaded from graph data
        )
    else:
        explainer = DefaultExplainer(
            client=client,
            threshold=config.threshold,
            verbose=config.verbose,
            activations=config.activations,
            cot=config.cot,
            temperature=config.temperature,
        )

    def post_process(result):
        """Save each explanation to a separate file and track it."""
        latent = result.record.latent
        feature = f"{latent.module_name}_{latent.latent_index}"
        log_entry = {
            "feature": feature,
            "prompt": result.prompt,
            "response": result.response,
            "explanation": result.explanation,
        }
        with open(f"{output_path}/explanations.jsonl", "a+") as f:
            f.write(json.dumps(log_entry) + "\n")

        with open(f"{output_path}/{result.record.latent}.txt", "w+") as f:
            f.write(json.dumps(result.explanation))
        return result

    if config.explainer_type == "graph":
        start = time.time()
        add_parent_connections_to_all_nodes(
            config.attribute_graph_path, config.logits_dir
        )
        print(f"Parent connections added in {time.time() - start:.2f} seconds")

        print(f"Loading attribution graph from {config.attribute_graph_path}")
        graph_data = load_graph(config.attribute_graph_path)

        # Extract prompt for context
        prompt = ""
        if config.graph_prompt:
            prompt = graph_data["metadata"]["prompt"]
            print(f"Using Graph prompt: {prompt}")

        # Process graph to get node names and topology
        print("Processing attribution graph nodes")
        name_mapping, node_info = convert_node_names(
            graph_data["nodes"], graph_data["links"]
        )
        print(f"Found {len(name_mapping)} transcoder nodes")

        # Build adjacency list and topological order
        adjacency_list = build_adjacency_list(name_mapping, graph_data["links"])
        topo_order = topological_sort(adjacency_list)
        print(f"Computed topological order: {len(topo_order)} nodes")

        # Create explanation pipeline
        explanation_pipeline = ExplanationPipeline(
            topo_order, adjacency_list, node_info
        )
        print(
            f"Pipeline initialized with {len(explanation_pipeline.queue)} ready nodes"
        )

        # Create latent dict from the graph nodes
        latent_dict = {}
        nodes_to_process = topo_order

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
            f"LatentData created with "
            f"{sum(len(v) for v in latent_dict.values())} features and "
            f"{len(latent_dict)} modules"
        )

        # Create pipeline for this iteration

        explainer_pipe = Pipe(process_wrapper(explainer, postprocess=post_process))

        print("Starting pipeline-based explanation generation")

        # Create a pipeline-based explanation process
        async def explain_with_pipeline():
            """Process features in topological order"""
            iteration = 0

            while not explanation_pipeline.is_complete():
                iteration += 1
                # Get all ready nodes (don't limit by batch_size as requested)
                ready_nodes = explanation_pipeline.get_ready_nodes(
                    batch_size=len(explanation_pipeline.queue)
                )

                if not ready_nodes:
                    status = explanation_pipeline.get_status()
                    print(f"Status: {status}")
                    break

                print(f"Iteration {iteration}: Processing {len(ready_nodes)} features")

                # Create latent_dict for this iteration's nodes
                iteration_latent_dict = (
                    explanation_pipeline.create_latent_dict_for_nodes(ready_nodes)
                )
                print("Created latent dict")

                # Create LatentDataset for this batch
                iteration_dataset = LatentDataset(
                    raw_dir=config.cache_dir,
                    sampler_cfg=sampler_cfg,
                    constructor_cfg=constructor_cfg,
                    modules=list(iteration_latent_dict.keys()),
                    latents=iteration_latent_dict,
                    tokenizer=tokenizer,
                    logits_directory=config.logits_dir,
                    graph_prompt=prompt,
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
    else:
        # Create latent range (which features to explain)
        latent_range = torch.arange(config.max_latents) if config.max_latents else None
        latent_dict = (
            {hook: latent_range for hook in config.hookpoints}
            if latent_range is not None
            else None
        )

        # Create the dataset
        dataset = LatentDataset(
            raw_dir=config.cache_dir,
            sampler_cfg=sampler_cfg,
            constructor_cfg=constructor_cfg,
            modules=config.hookpoints,
            latents=latent_dict,
            tokenizer=tokenizer,
        )

        # Wrap the explainer with post-processing
        explainer_pipe = Pipe(process_wrapper(explainer, postprocess=post_process))

        print("Starting explanation generation pipeline")
        # Create and run the pipeline
        pipeline = Pipeline(
            dataset,  # Load latent records from cache
            explainer_pipe,  # Generate and save explanations
        )

        # Run with limited concurrency to avoid overwhelming the model
        await pipeline.run(max_concurrent=5)

    print(f"Explanations saved at {config.experiment_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple Graph Explainer Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python examples/test_graph_explainer.py

  # Override experiment name
  python examples/test_graph_explainer.py --experiment-name my_test

        """,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        default="examples/config.yaml",
        help="Path to YAML configuration file",
    )

    # Key overrides
    parser.add_argument(
        "--base-dir",
        help="Override base directory for all paths",
    )
    parser.add_argument(
        "--prompt",
        help="Override prompt name (graph subfolder)",
    )
    parser.add_argument(
        "--experiment",
        help="Override experiment name",
    )
    parser.add_argument(
        "--explainer_type",
        choices=["graph", "default"],
        help="Type of explainer to use",
    )

    parser.add_argument(
        "--n_examples",
        type=int,
        help="Override number of examples to use",
    )
    parser.add_argument(
        "--logits",
        action="store_true",
        help="Override whether to use logits",
    )

    parser.add_argument(
        "--parent_explanations",
        type=int,
        help="Override number of parent explanations to use",
    )
    parser.add_argument(
        "--graph_prompt",
        action="store_true",
        help="Override whether to use graph prompt",
    )
    parser.add_argument(
        "--all_enabled",
        action="store_true",
        help="Enable graph prompt, logits, and parent explanations",
    )

    return parser


async def main():
    """Main function with simplified config support."""
    parser = parse_args()
    args = parser.parse_args()

    # Load config from YAML
    config = GraphExplainerConfig.from_yaml(args.config)

    # Apply command line overrides
    if args.base_dir:
        config.base_dir = args.base_dir
    if args.prompt:
        config.prompt_name = args.prompt
    if args.experiment:
        config.experiment_name = args.experiment

    if args.n_examples:
        config.n_examples_train = int(args.n_examples)
        config.min_examples = int(args.n_examples)
        config.max_examples = int(args.n_examples)
    if args.logits:
        config.top_logits = args.logits
        config.bot_logits = args.logits

    if args.parent_explanations is not None:
        config.max_parent_explanations = args.parent_explanations

    if args.explainer_type:
        config.explainer_type = args.explainer_type

    if args.graph_prompt:
        config.graph_prompt = args.graph_prompt

    if args.all_enabled:
        config.graph_prompt = True
        config.top_logits = True
        config.bot_logits = True
        config.max_parent_explanations = 3
        config.cot = True

    # Resolve all paths
    config = config.resolve_paths()

    # Create directories
    Path(config.experiment_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    config.to_yaml(f"{config.experiment_dir}/resolved_config.yaml")

    print("Experiment Configuration Summary:")
    print(f"Prompt: {config.prompt_name}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Explainer: {config.explainer_type}")

    if not Path(config.cache_dir).exists():
        print(f"Cache directory not found: {config.cache_dir}")
        return

    print(f"Generating {config.explainer_type} explanations")
    await generate_explanations_from_cache(config)

    # Remove all .txt explanation files
    for file in Path(config.experiment_dir).glob("*.txt"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Failed to remove {file}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
