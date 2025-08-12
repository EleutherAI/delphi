import gc
from typing import Tuple, List, Dict, Any
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM

from sae_dashboard.components import (
    ActsHistogramData,
    DecoderWeightsDistribution,
    FeatureTablesData,
    LogitsHistogramData,
    SequenceData,
    SequenceGroupData,
    SequenceMultiGroupData,
)
from sae_dashboard.data_parsing_fns import get_logits_table_data
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.utils_fns import FeatureStatistics, ASYMMETRIC_RANGES_AND_PRECISIONS
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset
from delphi.sparse_coders.sparse_model import load_sparsify_sparse_coders

try:
    from itertools import batched
except ImportError:
    from more_itertools import chunked as batched


def parse_args() -> Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed arguments with all configuration options.
    """
    parser = ArgumentParser(description="Convert SAE data for dashboard visualization")
    parser.add_argument(
        "--module", type=str, default="model.layers.9", help="Model module to analyze"
    )
    parser.add_argument(
        "--latents", type=int, default=5, help="Number of latents to process"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="../results/sae_pkm/baseline",
        help="Path to cached activations",
    )
    parser.add_argument(
        "--sae-path",
        type=str,
        default="../halutsae/sae-pkm/smollm/baseline",
        help="Path to trained SAE model",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="results/latent_dashboard.html",
        help="Path to save the dashboard",
    )
    parser.add_arguments(ConstructorConfig, dest="constructor_cfg")
    parser.add_arguments(
        SamplerConfig,
        dest="sampler_cfg",
        default=SamplerConfig(n_examples_train=25, n_quantiles=5, train_type="quantiles"),
    )
    return parser.parse_args()


def get_lm_head(cache_lm: AutoModelForCausalLM) -> torch.nn.Sequential:
    """
    Get the LM head for different HuggingFace architectures.

    Args:
        cache_lm (AutoModelForCausalLM): Loaded language model.

    Returns:
        torch.nn.Sequential: The LM head module for logits computation.
    """
    if hasattr(cache_lm, "model") and hasattr(cache_lm.model, "norm") and hasattr(cache_lm, "lm_head"):
        # llama models
        return torch.nn.Sequential(cache_lm.model.norm, cache_lm.lm_head)
    elif hasattr(cache_lm, "gpt_neox") and hasattr(cache_lm, "embed_out"):
        # pythia models
        return torch.nn.Sequential(cache_lm.gpt_neox.final_layer_norm, cache_lm.embed_out)
    else:
        raise ValueError("Unknown model architecture for extracting lm_head.")


def compute_quantiles() -> Tuple[List[float], torch.Tensor, List[Any]]:
    """
    Compute quantile bins for activation statistics.

    Returns:
        tuple[list[float], torch.Tensor, list[tuple]]: (quantiles, quantiles_tensor, ranges_and_precisions)
    """
    ranges_and_precisions = ASYMMETRIC_RANGES_AND_PRECISIONS
    quantiles = []
    for r, p in ranges_and_precisions:
        start, end = r
        step = 10**-p
        quantiles.extend(np.arange(start, end - 0.5 * step, step))
    quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)
    return quantiles, quantiles_tensor, ranges_and_precisions


def process_latents(
    raw_loader: LatentDataset,
    lm_head: torch.nn.Module,
    latent_to_resid: torch.Tensor,
    layout: Any,
    quantiles: List[float],
    quantiles_tensor: torch.Tensor,
    ranges_and_precisions: List[Any],
    n_sequences: int,
    max_seq_len: int,
) -> Tuple[Dict[int, FeatureData], FeatureStatistics]:
    """
    Process each latent, compute statistics and visualization data.

    Args:
        raw_loader (LatentDataset): Loader for latent activations.
        lm_head (torch.nn.Module): LM head for logits computation.
        latent_to_resid (torch.Tensor): Decoder weights.
        layout: Dashboard layout config.
        quantiles (list[float]): Quantile bins.
        quantiles_tensor (torch.Tensor): Quantile bins as tensor.
        ranges_and_precisions (list[tuple]): Ranges and precisions for quantiles.
        n_sequences (int): Number of sequences.
        max_seq_len (int): Maximum sequence length.

    Returns:
        tuple[dict[int, FeatureData], FeatureStatistics]:
            (latent_data_dict, latent_stats)
    """
    latent_data_dict = {}
    latent_stats = FeatureStatistics()
    bar = tqdm(total=len(raw_loader.buffers))
    for _, record in enumerate(raw_loader):
        assert record.activation_data is not None
        latent_id = record.activation_data.locations[0, 2].item()
        decoder_resid = latent_to_resid[latent_id].to(record.activation_data.activations.device)
        logit_vector = lm_head(decoder_resid)

        activations = record.activation_data.activations
        _max = activations.max()
        nonzero_mask = activations.abs() > 1e-6
        nonzero_acts = activations[nonzero_mask]
        frac_nonzero = nonzero_mask.sum() / (n_sequences * max_seq_len)
        quantile_data = torch.quantile(activations.float(), quantiles_tensor)
        skew = torch.mean((activations - activations.mean()) ** 3) / (activations.std() ** 3)
        kurtosis = torch.mean((activations - activations.mean()) ** 4) / (activations.std() ** 4)
        latent_stats.update(
            FeatureStatistics(
                max=[_max.item()],
                frac_nonzero=[frac_nonzero.item()],
                skew=[skew.item()],
                kurtosis=[kurtosis.item()],
                quantile_data=[quantile_data.unsqueeze(0).tolist()],
                quantiles=quantiles + [1.0],
                ranges_and_precisions=ranges_and_precisions,
            )
        )

        latent_data = FeatureData()
        latent_data.feature_tables_data = FeatureTablesData()
        latent_data.logits_histogram_data = LogitsHistogramData.from_data(
            data=logit_vector.to(torch.float32),
            n_bins=layout.logits_hist_cfg.n_bins,
            tickmode="5 ticks",
            title=None,
        )
        latent_data.acts_histogram_data = ActsHistogramData.from_data(
            data=nonzero_acts.to(torch.float32),
            n_bins=layout.act_hist_cfg.n_bins,
            tickmode="5 ticks",
            title=f"ACTIVATIONS<br>DENSITY = {frac_nonzero:.3%}",
        )
        latent_data.logits_table_data = get_logits_table_data(
            logit_vector=logit_vector,
            n_rows=layout.logits_table_cfg.n_rows,
        )
        latent_data.decoder_weights_data = DecoderWeightsDistribution(
            len(decoder_resid), decoder_resid.tolist()
        )
        latent_data_dict[latent_id] = latent_data
        bar.update(1)
        bar.refresh()
    bar.close()
    return latent_data_dict, latent_stats


def process_sequences(
    sequence_loader: LatentDataset,
    latent_data_dict: Dict[int, FeatureData],
    n_quantiles: int,
) -> None:
    """
    Group sequence examples by quantile for each latent.

    Args:
        sequence_loader (LatentDataset): Loader for sequence data.
        latent_data_dict (dict[int, FeatureData]): Dictionary of latent feature data.
        n_quantiles (int): Number of quantile groups.

    Returns:
        None
    """
    bar = tqdm(total=len(sequence_loader.buffers))
    for record in sequence_loader:
        groups = []
        for quantile_index, quantile_data in enumerate(
            list(batched(record.train, len(record.train) // n_quantiles))[::-1]
        ):
            group = []
            for example in quantile_data:
                default_list = [0.0] * len(example.tokens)
                logit_list = [[0.0]] * len(default_list)
                token_list = [[0]] * len(default_list)
                default_attrs = dict(
                    loss_contribution=default_list,
                    token_logits=default_list,
                    top_token_ids=token_list,
                    top_logits=logit_list,
                    bottom_token_ids=token_list,
                    bottom_logits=logit_list,
                )
                group.append(
                    SequenceData(
                        token_ids=example.tokens.tolist(),
                        feat_acts=example.activations.tolist(),
                        **default_attrs,
                    )
                )
            groups.append(
                SequenceGroupData(
                    title=f"Quantile {quantile_index/n_quantiles:1%}-{(quantile_index+1)/n_quantiles:1%}",
                    seq_data=group,
                )
            )
        latent_data_dict[record.latent.latent_index].sequence_data = SequenceMultiGroupData(
            seq_group_data=groups
        )
        bar.update(1)
        bar.refresh()
    bar.close()


def main() -> None:
    """
    Main function to run SAE dashboard conversion.

    Returns:
        None
    """
    # Setup configurations and arguments
    torch.set_grad_enabled(False)
    args = parse_args()
    constructor_cfg = args.constructor_cfg
    sampler_cfg = args.sampler_cfg
    out_path = args.out_path
    module = args.module
    n_latents = args.latents
    start_latent = 0
    latent_dict = {f"{module}": torch.arange(start_latent, start_latent + n_latents)}
    kwargs = dict(
        raw_dir=args.cache_path,
        modules=[module],
        latents=latent_dict,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
    )

    # Load latent dataset
    raw_loader = LatentDataset(
        **(
            kwargs
            | {"constructor_cfg": replace(constructor_cfg, save_activation_data=True)}
        )
    )
    print("Loading model")
    model_name = raw_loader.cache_config["model_name"]
    cache_lm = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="cpu"
    )
    lm_head = get_lm_head(cache_lm)

    # Load SAE model
    hookpoint_to_sparse_model = load_sparsify_sparse_coders(
        args.sae_path,
        [args.module],
        "cuda",
        compile=False,
    )
    transcoder = hookpoint_to_sparse_model[args.module]
    w_dec = transcoder.W_dec.data
    latent_to_resid = w_dec

    del cache_lm
    gc.collect()

    tokens = raw_loader.buffers[0].load()[-1]
    n_sequences, max_seq_len = tokens.shape

    cfg = SaeVisConfig(
        hook_point=args.module,
        minibatch_size_tokens=raw_loader.cache_config["cache_ctx_len"],
        features=[],
    )
    layout = cfg.feature_centric_layout

    quantiles, quantiles_tensor, ranges_and_precisions = compute_quantiles()

    # Process latents
    latent_data_dict, latent_stats = process_latents(
        raw_loader, lm_head, latent_to_resid, layout, quantiles, quantiles_tensor, ranges_and_precisions, n_sequences, max_seq_len
    )
    latent_list = latent_dict[module].tolist()
    cfg.features = latent_list

    # Process sequences
    n_quantiles = sampler_cfg.n_quantiles
    sequence_loader = LatentDataset(
        **kwargs | dict(sampler_cfg=sampler_cfg)
    )
    process_sequences(sequence_loader, latent_data_dict, n_quantiles)

    # Save dashboard
    latent_list = list(latent_data_dict.keys())
    tokenizer = raw_loader.tokenizer
    model = Namespace(tokenizer=tokenizer)
    sae_vis_data = SaeVisData(
        cfg=cfg,
        feature_data_dict=latent_data_dict,
        feature_stats=latent_stats,
        model=model,
    )
    print(f"Saving dashboard to {out_path}")
    save_feature_centric_vis(sae_vis_data=sae_vis_data, filename=out_path)


if __name__ == "__main__":
    main()