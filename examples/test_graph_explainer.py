import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from attrib_graph import (
    ExplanationPipeline,
    add_parent_connections_to_all_nodes,
    build_adjacency_list,
    convert_node_names,
    get_prompt,
    load_graph,
    topological_sort,
)
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Import Delphi components
from delphi.clients import Offline, OpenRouter
from delphi.config import CacheConfig, ConstructorConfig, SamplerConfig
from delphi.explainers import (
    DefaultExplainer,
    GraphExplainer,
)
from delphi.latents import LatentCache, LatentDataset
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.sparse_coders.load_sparsify import load_sparsify_hooks
from delphi.utils import load_tokenized_data


def load_config(config_path: str = "examples/explainer_configs.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_explainer_factory(config: Dict[str, Any], tokenizer=None):
    """
    Factory function to create explainer instances based on configuration.

    Args:
        config: Configuration dictionary from YAML
        tokenizer: Required for SingleTokenExplainer

    Returns:
        Function that creates explainer instances given a client
    """

    def create_explainer(explainer_type: str, client, **override_params):
        """Create an explainer instance of the specified type."""
        if explainer_type not in config["explainers"]:
            raise ValueError(
                f"bad explainer. Use one of these: {list(config['explainers'].keys())}"
            )

        explainer_config = config["explainers"][explainer_type]

        if explainer_type == "default":
            return DefaultExplainer(client=client, **explainer_config)

        elif explainer_type == "graph":
            # Graph explainer needs special handling for paths and graph information
            dirs = config.get("dirs", {})

            # Add required graph-specific parameters
            graph_params = explainer_config.copy()
            graph_params["graph_info_path"] = dirs.get("attribute_graph", "")
            graph_params["explanations_dir"] = override_params.get(
                "explanations_dir", "results/graph_explanations"
            )

            # Load graph prompt from the graph data if available
            graph_prompt = None
            if graph_params["graph_info_path"]:
                try:
                    with open(graph_params["graph_info_path"], "r") as f:
                        graph_data = json.load(f)
                        graph_prompt = graph_data.get("metadata", {}).get("prompt", "")
                except Exception as e:
                    print(f"Failed to load prompt: {e}")
                    graph_prompt = ""

            graph_params["graph_prompt"] = graph_prompt or ""

            return GraphExplainer(client=client, **graph_params)

        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")

    return create_explainer


def create_client(client_type: str, model_name: str, config: Dict[str, Any]):
    """Create a client based on configuration."""
    if client_type not in config["client_configs"]:
        raise ValueError(
            f"Unknown client type. Use: {list(config['client_configs'].keys())}"
        )

    client_config = config["client_configs"][client_type]

    if client_type == "offline":
        return Offline(
            model_name,
            max_memory=client_config.get("max_memory", 0.8),
            max_model_len=client_config.get("max_model_len", 4096),
            num_gpus=torch.cuda.device_count(),
            statistics=client_config.get("statistics", True),
        )

    elif client_type == "openrouter":
        api_key_env = client_config.get("api_key_env", "OPENROUTER_API_KEY")
        if api_key_env not in os.environ:
            raise ValueError(
                f"{api_key_env} environment variable must be set when using OpenRouter"
            )

        return OpenRouter(model_name, api_key=os.environ[api_key_env])

    else:
        raise ValueError(f"Unsupported client type: {client_type}")


def get_constructor_config(
    config_name: str, config: Dict[str, Any]
) -> ConstructorConfig:
    """Create ConstructorConfig from YAML configuration."""
    if config_name not in config["constructor_configs"]:
        raise ValueError(f"Unknown constructor config: {config_name}")

    cfg = config["constructor_configs"][config_name]
    return ConstructorConfig(
        example_ctx_len=cfg.get("example_ctx_len", 32),
        min_examples=cfg.get("min_examples", 50),
        n_non_activating=cfg.get("n_non_activating", 25),
        center_examples=cfg.get("center_examples", True),
        non_activating_source=cfg.get("non_activating_source", "random"),
    )


def get_sampler_config(config_name: str, config: Dict[str, Any]) -> SamplerConfig:
    """Create SamplerConfig from YAML configuration."""
    if config_name not in config["sampler_configs"]:
        raise ValueError(f"Unknown sampler config: {config_name}")

    cfg = config["sampler_configs"][config_name]
    return SamplerConfig(
        n_examples_train=cfg.get("n_examples_train", 20),
        n_examples_test=cfg.get("n_examples_test", 30),
        n_quantiles=cfg.get("n_quantiles", 10),
        train_type=cfg.get("train_type", "quantiles"),
        test_type=cfg.get("test_type", "quantiles"),
    )


def load_model_with_fallback(
    model_path: str, load_in_8bit: bool = False, hf_token: Optional[str] = None
):
    """Load model from local directory if it exists, otherwise from HuggingFace."""
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
            print("   [OK] Successfully loaded model from local directory")
            return model
        except Exception as e:
            print(f"   [!]  Failed to load from local directory: {e}")
            print("   [>>] Falling back to HuggingFace...")

    print(f"   Loading model from HuggingFace: {model_path}")
    try:
        model = AutoModel.from_pretrained(
            model_path,
            device_map={"": "cuda"},
            quantization_config=quantization_config,
            torch_dtype=dtype,
            token=hf_token,
        )
        print("   [OK] Successfully loaded model from HuggingFace")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from both local and HuggingFace: {e}")


def load_tokenizer_with_fallback(model_path: str, hf_token: Optional[str] = None):
    """Load tokenizer from local directory if it exists, otherwise from HuggingFace."""
    local_path = Path(model_path)

    if local_path.exists() and local_path.is_dir():
        print(f"   Loading tokenizer from local directory: {local_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path, token=hf_token)
            print("   [OK] Successfully loaded tokenizer from local directory")
            return tokenizer
        except Exception as e:
            print(f"   [!]  Failed to load from local directory: {e}")
            print("   [>>] Falling back to HuggingFace...")

    print(f"   Loading tokenizer from HuggingFace: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        print("   [OK] Successfully loaded tokenizer from HuggingFace")
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
    """Generate activation cache from a base model and sparse autoencoder."""

    print("[>>]  Starting cache generation...")

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_path = cache_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"[AI] Loading base model: {base_model_path}")
    model = load_model_with_fallback(base_model_path, load_in_8bit, hf_token)

    print("[TK] Loading tokenizer...")
    tokenizer = load_tokenizer_with_fallback(base_model_path, hf_token)

    print(f"[SAE] Loading sparse autoencoders: {sparse_model_path}")
    print(f"   Hookpoints: {hookpoints}")
    hookpoint_to_sparse_encode, transcode = load_sparsify_hooks(
        model,
        sparse_model_path,
        hookpoints,
        compile=True,
    )

    print(f"[DATA] Loading and tokenizing dataset: {dataset_repo}")
    print(f"   Split: {dataset_split}, Column: {dataset_column}")
    print(f"   Context length: {ctx_len}, Tokens to process: {n_tokens:,}")
    tokens = load_tokenized_data(
        ctx_len,
        tokenizer,
        dataset_repo,
        dataset_split,
        dataset_name="",
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

    print("[CACHE] Creating activation cache...")
    print(f"   Batch size: {batch_size}")
    print(f"   Transcode mode: {transcode}")
    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=batch_size,
        transcode=transcode,
        log_path=log_path,
    )

    # Run the caching process
    cache.run(n_tokens, tokens)

    print("[STATS] Generating cache statistics...")
    cache.generate_statistics_cache()

    print("[SAVE] Saving cache to disk...")
    print(f"   Cache directory: {cache_dir}")
    print(f"   Number of splits: {n_splits}")
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

    print("[OK] Cache generation complete!")
    print(f"   Cache saved to: {cache_dir}")
    print(f"   Hookpoints processed: {list(hookpoint_to_sparse_encode.keys())}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()


async def generate_explanations_from_cache(
    cache_dir: str,
    output_dir: str,
    model_name: str,
    hookpoints: list[str],
    explainer_type: str,
    client_type: str,
    explainer_model: str,
    max_latents: int,
    config: Dict[str, Any],
    sampler_config_name: str = "default",
    constructor_config_name: str = "basic",
):
    """
    Generate explanations
    from a pre-existing activation cache
    using configurable explainers.
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create configs from YAML
    sampler_cfg = get_sampler_config(sampler_config_name, config)
    constructor_cfg = get_constructor_config(constructor_config_name, config)

    # Validate explainer and constructor compatibility
    explainer_config = config["explainers"][explainer_type]
    required_constructor = explainer_config.get("required_constructor_cfg", {})

    for key, required_value in required_constructor.items():
        actual_value = getattr(constructor_cfg, key)
        if actual_value != required_value:
            print(
                f"Explainer requires {key}='{required_value}' but got '{actual_value}'"
            )
            print("   Updating constructor config to match explainer requirements...")
            setattr(constructor_cfg, key, required_value)

    # Create client
    client = create_client(client_type, explainer_model, config)

    # Create explainer factory and explainer instance
    explainer_factory = create_explainer_factory(config, tokenizer)
    explainer = explainer_factory(explainer_type, client, explanations_dir=output_dir)

    def post_process(result):
        """Save each explanation to a separate file and track it."""
        latent = result.record.latent
        feature = f"{latent.module_name}_{latent.latent_index}"
        log_entry = {
            "feature": feature,
            "prompt": result.prompt,
            "response": result.response_text,
            "explanation": result.explanation,
        }
        with open(f"{output_path}/explanations.jsonl", "a+") as f:
            f.write(json.dumps(log_entry) + "\n")

        with open(f"{output_path}/{result.record.latent}.txt", "w+") as f:
            f.write(json.dumps(result.explanation))
        return result

    if explainer_type == "graph":
        start = time.time()
        graph_dir = config.get("dirs", {}).get("attribute_graph", "")
        logits_dir = config.get("dirs", {}).get("logits", "")
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
        explanation_pipeline = ExplanationPipeline(
            topo_order, adjacency_list, node_info
        )
        print(
            f"Pipeline initialized with {len(explanation_pipeline.queue)} ready nodes"
        )

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

        # Create pipeline for this iteration

        explainer_pipe = Pipe(process_wrapper(explainer, postprocess=post_process))

        print("Starting pipeline-based explanation generation...")

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
                    raw_dir=cache_dir,
                    sampler_cfg=sampler_cfg,
                    constructor_cfg=constructor_cfg,
                    modules=list(iteration_latent_dict.keys()),
                    latents=iteration_latent_dict,
                    tokenizer=tokenizer,
                    logits_directory=logits_dir,
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

        print(f"Explanation generation complete. Results saved to {output_dir}")
    else:
        # Create latent range (which features to explain)
        latent_range = torch.arange(max_latents) if max_latents else None
        latent_dict = (
            {hook: latent_range for hook in hookpoints}
            if latent_range is not None
            else None
        )

        # Create the dataset
        dataset = LatentDataset(
            raw_dir=cache_dir,
            sampler_cfg=sampler_cfg,
            constructor_cfg=constructor_cfg,
            modules=hookpoints,
            latents=latent_dict,
            tokenizer=tokenizer,
        )

        # Wrap the explainer with post-processing
        explainer_pipe = Pipe(process_wrapper(explainer, postprocess=post_process))

        print("[RUN] Starting explanation generation pipeline...")
        # Create and run the pipeline
        pipeline = Pipeline(
            dataset,  # Load latent records from cache
            explainer_pipe,  # Generate and save explanations
        )

        # Run with limited concurrency to avoid overwhelming the model
        await pipeline.run(max_concurrent=5)

        print(f"   Explanation generation complete. Results saved to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Configurable Explainer Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a predefined experiment preset
  python examples/configurable_explainer_demo.py --preset quick_test

  # Use specific explainer type with custom parameters
  python examples/configurable_explainer_demo.py --explainer graph

        """,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        default="examples/explainer_configs.yaml",
        help="Path to YAML configuration file",
    )

    # Experiment presets vs custom configuration
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--preset",
        choices=None,  # Will be populated after loading config
        help="Use predefined experiment preset",
    )

    # Custom configuration options
    parser.add_argument(
        "--explainer",
        choices=None,  # Will be populated after loading config
        help="Explainer type to use",
    )
    parser.add_argument(
        "--client",
        choices=None,  # Will be populated after loading config
        help="Client type (offline/openrouter)",
    )
    parser.add_argument(
        "--sampler_config", default="default", help="Sampler configuration name"
    )
    parser.add_argument(
        "--constructor_config", default="default", help="Constructor configuration name"
    )

    # Model configuration
    parser.add_argument(
        "--model_preset",
        choices=None,  # Will be populated after loading config
        help="Predefined model configuration",
    )
    parser.add_argument("--base_model", help="Base model path/name")
    parser.add_argument("--sparse_model", help="Sparse model path/name")
    parser.add_argument("--explainer_model", help="Explainer model path/name")
    parser.add_argument("--hookpoints", nargs="+", help="Hookpoints to process")

    # Execution options
    parser.add_argument(
        "--generate_cache",
        action="store_true",
        help="Generate activation cache from models",
    )
    parser.add_argument(
        "--generate_explanations",
        action="store_true",
        default=True,
        help="Generate explanations from cache (default: True)",
    )

    # Dataset and processing options
    parser.add_argument(
        "--max_latents",
        type=int,
        default=20,
        help="Maximum number of latents to explain",
    )
    parser.add_argument("--cache_dir", help="Cache directory path")
    parser.add_argument("--output_dir", help="Output directory for explanations")

    # Cache generation options
    parser.add_argument(
        "--n_tokens", type=int, default=1_000_000, help="Number of tokens to cache"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for cache generation"
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load models in 8-bit quantization"
    )

    return parser


async def main():
    """Main function with configurable explainer support."""
    # First, load config to populate choices
    config = load_config()

    # Set up argument parser with dynamic choices
    parser = parse_args()

    # Update choices based on config
    for action in parser._actions:
        if action.dest == "preset":
            action.choices = list(config.get("experiment_presets", {}).keys())
        elif action.dest == "explainer":
            action.choices = list(config.get("explainers", {}).keys())
        elif action.dest == "client":
            action.choices = list(config.get("client_configs", {}).keys())
        elif action.dest == "model_preset":
            action.choices = list(config.get("model_presets", {}).keys())

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    # Load experiment preset if specified
    if args.preset:
        if args.preset not in config["experiment_presets"]:
            print(f"[X] Unknown preset: {args.preset}")
            print(f"Available presets: {list(config['experiment_presets'].keys())}")
            return

        preset = config["experiment_presets"][args.preset]

        explainer_type = preset["explainer"]
        client_type = preset["client"]
        sampler_config = preset.get("sampler_configs", "default")
        constructor_config = preset.get("constructor_configs", "default")
        max_latents = preset.get("max_latents", 20)
        model_preset = config["model_presets"].get(
            preset.get("model_preset", "gemma2s")
        )
    else:
        # Use command line arguments or defaults
        explainer_type = args.explainer or "default"
        client_type = args.client or "offline"
        sampler_config = args.sampler_config
        constructor_config = args.constructor_config
        max_latents = args.max_latents
        model_preset = None
    # Load model configuration
    if model_preset:
        base_model_path = model_preset.get("base_model", "google/gemma-2-2b-it")
        sparse_model_path = model_preset.get(
            "sparse_model", "EleutherAI/gemmascope-sparsify-1m"
        )
        explainer_model = model_preset.get("explainer_model", "")
        hookpoints = model_preset.get("hookpoints", [])
    else:
        base_model_path = args.base_model or "google/gemma-2-2b-it"
        sparse_model_path = args.sparse_model or "EleutherAI/gemmascope-sparsify-1m"
        explainer_model = args.explainer_model or "meta-llama/Llama-3.1-8B-Instruct"
        hookpoints = args.hookpoints or ["layers.5.mlp"]

    # Set up paths
    config = load_config(args.config)
    cache_dir = config.get("dirs", {}).get("cache", "cache")
    prompt = get_prompt(config["dirs"]["attribute_graph"])
    output_dir = Path(config.get("dirs", {}).get("output", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[CFG] Configuration Summary")
    print(f"   Explainer: {explainer_type}")
    print(f"   Client: {client_type}")
    print(f"   Base model: {base_model_path}")
    print(f"   Sparse model: {sparse_model_path}")
    print(f"   Explainer model: {explainer_model}")
    print(f"   Cache directory: {cache_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Graph prompt: {prompt}")

    # ====== EXECUTION SECTION ======

    try:
        if args.generate_cache:
            print("\n[>>]  Phase 1: Generating activation cache...")

            generate_cache_from_model(
                base_model_path=base_model_path,
                sparse_model_path=sparse_model_path,
                hookpoints=hookpoints,
                cache_dir=cache_dir,
                n_tokens=args.n_tokens,
                batch_size=args.batch_size,
                load_in_8bit=args.load_in_8bit,
                filter_bos=True,
            )
            print("[OK] Cache generation completed successfully!")

        if args.generate_explanations:
            # Check if cache exists
            if not Path(cache_dir).exists():
                print(f"[X] Cache directory not found: {cache_dir}")
                print("Please run cache generation first with --generate_cache")
                return
            # Save resolved config as JSON in output directory
            # Precompute resolved_config in shorter, split lines
            resolved_config = {}
            resolved_config["preset"] = args.preset if args.preset else None
            resolved_config["explainer"] = explainer_type
            resolved_config["client"] = client_type
            resolved_config["sampler_config"] = sampler_config
            resolved_config["sampler_config_values"] = config.get(
                "sampler_configs"
            ).get(sampler_config)
            resolved_config["constructor_config"] = constructor_config
            resolved_config["constructor_config_values"] = config.get(
                "constructor_configs"
            ).get(constructor_config)
            resolved_config["explainer_config"] = config.get("explainers", {}).get(
                explainer_type, {}
            )
            resolved_config["client_config"] = config.get("client_configs", {}).get(
                client_type, {}
            )
            resolved_config["model_preset"] = (
                config.get("experiment_presets", {})
                .get(args.preset, {})
                .get("model_preset", None)
                if args.preset
                else getattr(args, "model_preset", None)
            )
            resolved_config["model_preset_values"] = (
                model_preset if model_preset else {}
            )
            resolved_config["base_model"] = base_model_path
            resolved_config["sparse_model"] = sparse_model_path
            resolved_config["explainer_model"] = explainer_model
            resolved_config["hookpoints"] = hookpoints
            resolved_config["max_latents"] = max_latents
            resolved_config["n_tokens"] = getattr(args, "n_tokens", None)
            resolved_config["batch_size"] = getattr(args, "batch_size", None)
            resolved_config_path = f"{output_dir}/experiment_config.json"
            with open(resolved_config_path, "w+") as f:
                json.dump(resolved_config, f, indent=2)
            print(f"    Resolved experiment config saved to: {resolved_config_path}")

            print(
                f"    Phase 2: Generating {explainer_type} explanations from cache..."
            )

            await generate_explanations_from_cache(
                cache_dir=cache_dir,
                output_dir=output_dir,
                model_name=base_model_path,
                hookpoints=hookpoints,
                explainer_type=explainer_type,
                client_type=client_type,
                explainer_model=explainer_model,
                max_latents=max_latents,
                config=config,
                sampler_config_name=sampler_config,
                constructor_config_name=constructor_config,
            )
            print("[OK] Explanation generation completed successfully")

        if not args.generate_cache and not args.generate_explanations:
            print("Specify --generate_cache and/or --generate_explanations")
            return

        if args.generate_cache:
            print(f"   Cache saved to: {cache_dir}")
        if args.generate_explanations:
            print(f"   {explainer_type.title()} explanations saved to: {output_dir}")

    except Exception as e:
        print(f"[X] Error occurred: {e}")
        print("[DEBUG] Full stack trace:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
