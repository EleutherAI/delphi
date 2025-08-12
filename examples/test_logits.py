#!/usr/bin/env python3
"""
Basic Explanation Generation Script

This script demonstrates how to generate explanations from a pre-existing cache
without any fancy tricks. It's a minimal example showing the core workflow:
1. Load cached activations
2. Create a dataset from the cache
3. Generate explanations using a simple explainer
4. Save the results

Usage:
    python examples/basic_explanation_generation.py
    
Prerequisites:
    - You must have a pre-existing cache directory (created by running the main pipeline first)
    - Set OPENROUTER_API_KEY environment variable if using OpenRouter
"""

import asyncio
import os
import traceback
from pathlib import Path
from functools import partial
from typing import Optional
import json
import math

import orjson
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Import Delphi components
from delphi.clients import Offline, OpenRouter
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.explainers import DefaultExplainer
from delphi.explainers.explainer import ExplainerResult
from delphi.latents import LatentCache, LatentDataset
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.sparse_coders.load_sparsify import load_sparsify_hooks
from delphi.utils import load_tokenized_data


def load_model_with_fallback(model_path: str, load_in_8bit: bool = False, hf_token: Optional[str] = None):
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
        raise RuntimeError(f"Failed to load tokenizer from both local and HuggingFace: {e}")


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
    
    print(f"🤖 Loading base model: {base_model_path}")
    # Load base model
    model = load_model_with_fallback(base_model_path, load_in_8bit, hf_token)
    
    print(f"🔤 Loading tokenizer...")
    # Load tokenizer
    tokenizer = load_tokenizer_with_fallback(base_model_path, hf_token)
    
    print(f"🧠 Loading sparse autoencoders: {sparse_model_path}")
    print(f"   Hookpoints: {hookpoints}")
    # Load sparse autoencoders and create encoding hooks
    hookpoint_to_sparse_encode, transcode = load_sparsify_hooks(
        model,
        sparse_model_path,
        hookpoints,
        compile=True,
    )
    
    print(f"📚 Loading and tokenizing dataset: {dataset_repo}")
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
    
    print(f"⚡ Creating activation cache...")
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
    
    print("📊 Generating cache statistics...")
    # Generate statistics
    cache.generate_statistics_cache()
    
    print(f"💾 Saving cache to disk...")
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
    
    print("✅ Cache generation complete!")
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
    
    print(f"🔧 Loading tokenizer for {model_name}...")
    # Load tokenizer (needed to convert tokens back to text)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"📚 Creating dataset from cache at {cache_dir}...")
    # Create dataset from cached activations
    # These configs control how examples are constructed and sampled
    sampler_cfg = SamplerConfig(
        n_examples_train=20,  # Number of examples to use for explanation generation
        n_examples_test=30,   # Number of examples to use for testing (not used here)
        n_quantiles=10,       # Number of activation quantiles to sample from
        train_type="quantiles",  # Sample across different activation strengths
        test_type="quantiles"
    )
    
    constructor_cfg = ConstructorConfig(
        example_ctx_len=32,              # Length of each example (in tokens)
        min_examples=50,                 # Minimum activatioons needed to explain a latent
        n_non_activating=25,             # Number of non-activating examples
        center_examples=True,            # Center examples on the activating token
        non_activating_source="random"   # Use random non-activating examples (no fancy tricks)
    )
    
    # Create latent range (which features to explain)
    latent_range = torch.arange(max_latents) if max_latents else None
    latent_dict = {hook: latent_range for hook in hookpoints} if latent_range is not None else None
    print(latent_dict)
    logits_dir = "/mnt/ssd-1/soar-automated_interpretability/graphs/pawan/delphi-env/attribute/attribution-graphs-frontend/features/gemmascope-transcoders-sparsify-1m"
    def cantor_decode(num):
        w = math.floor((math.sqrt(8 * num + 1) - 1) / 2)
        t = (w * w + w) // 2
        y = num - t
        x = w - y
        return x, y

    latent_dict = {}
    for filename in os.listdir(logits_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(logits_dir, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                layer, index = cantor_decode(int(data['index']))
                module  = f"layers.{layer}.mlp"
                if module not in latent_dict:
                    latent_dict[module] = []
                latent_dict[module].append(index)
    for key in latent_dict.keys():
        latent_dict[key] = torch.tensor(latent_dict[key])
    # Create the dataset that will load and construct examples from cache
    dataset = LatentDataset(
        raw_dir=cache_dir,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        modules=list(latent_dict.keys()),
        latents=latent_dict,
        tokenizer=tokenizer,
        logits_directory = logits_dir
    )
    
    print(f"Setting up explainer model: {explainer_model}...")
    # Create client for the explainer model
    if use_openrouter:
        if "OPENROUTER_API_KEY" not in os.environ:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set when using OpenRouter")
        
        client = OpenRouter(
            explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"]
        )
        print("   Using OpenRouter API")
    else:
        client = Offline(
            explainer_model,
            max_memory=0.8,      # Use 80% of available GPU memory
            max_model_len=4096,  # Context length for the explainer model
            num_gpus=torch.cuda.device_count()  # Use all available GPUs
        )
        print(f"   Using local model with {torch.cuda.device_count()} GPU(s)")
    
    # Create the explainer
    explainer = DefaultExplainer(
        client,
        threshold=0.3,  # Activation threshold for highlighting tokens
        verbose=True    # Print explanations as they're generated
    )
    
    print("Setting up output processing...")
    # Define what to do after each explanation is generated
    def explanation_postprocess(result: ExplainerResult) -> ExplainerResult:
        """Save each explanation to a separate file"""
        output_file = output_path / f"{result.record.latent}.txt"
        
        with open(output_file, "wb") as f:
            f.write(orjson.dumps(result.explanation))
        
        print(f"   Saved explanation for {result.record.latent}")
        return result
    
    # Wrap the explainer with post-processing
    explainer_pipe = Pipe(
        process_wrapper(explainer, postprocess=explanation_postprocess)
    )
    
    print("Starting explanation generation pipeline...")
    # Create and run the pipeline
    pipeline = Pipeline(
        dataset,        # Load latent records from cache
        explainer_pipe  # Generate and save explanations
    )
    
    # Run with limited concurrency to avoid overwhelming the model
    await pipeline.run(max_concurrent=5)
    
    print(f"Explanation generation complete! Results saved to {output_dir}")
    print(f"   Generated explanations for latents in hookpoints: {hookpoints}")


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
    sparse_model_path = "/mnt/ssd-1/soar-automated_interpretability/graphs/pawan/delphi-env/models/gemmascope-transcoders-sparsify-1m" 
    
    # Hookpoints to process (which layers to attach replacement models to)
    hookpoints = ["layers.5.mlp"]
    
    # EXPLAINER MODEL (the model that generates explanations)
    explainer_model = "meta-llama/Llama-3.1-8B-Instruct"
    # explainer_model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    
    # Paths
    cache_dir = "/mnt/ssd-1/soar-automated_interpretability/graphs/pawan/delphi-env/delphi/results/gemma2b/latents"
    output_dir = "results/basic_explanations"
    
    # What to do - set these flags based on what you want to run
    GENERATE_CACHE = False   # Set to True to create cache from models
    GENERATE_EXPLANATIONS = True   # Set to True to generate explanations
    
    # Cache generation parameters
    cache_params = {
        "n_tokens": 500_000,                    # Number of tokens to cache (smaller for demo)
        "batch_size": 8,                        # Batch size (adjust based on GPU memory)
        "ctx_len": 256,                        # Context length
        "dataset_repo": "EleutherAI/fineweb-edu-dedup-10b",
        "dataset_split": "train[:1%]",
        "load_in_8bit": False,                 # Set to True if running out of GPU memory
        "filter_bos": True,
        "hf_token": None,                      # Set if using private models
    }
    
    # Explanation generation parameters  
    explanation_params = {
        "max_latents": 20,                     # Number of features to explain
        "explainer_model": explainer_model,    # Model that writes the explanations
        "use_openrouter": False,               # Set to True to use OpenRouter API
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
                **cache_params
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
                **explanation_params
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