#%%
%cd ..
#%%
%load_ext autoreload
%autoreload 2
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.components import FeatureTablesData, LogitsHistogramData, ActsHistogramData
from sae_dashboard.components import SequenceMultiGroupData, SequenceGroupData, SequenceData
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.utils_fns import FeatureStatistics
from sae_dashboard.components import DecoderWeightsDistribution
from sae_dashboard.data_parsing_fns import get_logits_table_data
try:
    from itertools import batched
except ImportError:
    from more_itertools import chunked as batched
from argparse import Namespace
from delphi.config import ExperimentConfig, LatentConfig
from delphi.latents import LatentDataset, LatentRecord
from delphi.latents.constructors import constructor
from delphi.latents.samplers import sampler
from functools import partial
import torch
torch.set_grad_enabled(False)
import numpy as np
from sae_dashboard.utils_fns import ASYMMETRIC_RANGES_AND_PRECISIONS
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoConfig
import gc
import re
args = Namespace(
    module="model.layers.9",
    latent_options=LatentConfig(),
    latents=100,
    model="sae_pkm/baseline",
    sae_path="../halutsae/sae-pkm/smollm/baseline"
)
#%%
module = args.module
latent_cfg = args.latent_options
n_latents = args.latents
start_latent = 0
sae_model = args.model
latent_dict = {f"{module}": torch.arange(start_latent,start_latent+n_latents)}
kwargs = dict(
    raw_dir=f"results/{args.model}",
    latent_cfg=latent_cfg,
    modules=[module],
    latents=latent_dict,
    experiment_cfg=ExperimentConfig(),
)


def set_record_buffer(record, *, latent_data):
    record.buffer = latent_data.activation_data
    return record
loader = LatentDataset(**kwargs, constructor=set_record_buffer, sampler=lambda x: x)
#%%
model_name = loader.cache_config["model_name"]
cache_lm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cpu")
#%%
lm_head = torch.nn.Sequential(
    cache_lm.model.norm,
    cache_lm.lm_head
)
#%%
lm_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
#%%
from pathlib import Path
from delphi.sparse_coders.sparse_model import load_sparsify_sparse_coders
if (router_match := re.match(r"\.model\.layers\.(\d+)\.router", args.module)):
    cache_lm
    # monet model
    layer = int(router_match.group(1))
    affected_layers = list(range(layer, layer+lm_config.moe_groups))
    total_bias = 0
    for l in affected_layers:
        moe = cache_lm.model.layers[l].moe
        b1, b2 = moe.b1, moe.b2
        x1_est = moe.u1.bias.reshape(lm_config.moe_experts, lm_config.moe_dim // 2)
        x2_est = moe.u2.bias.reshape(lm_config.moe_experts, lm_config.moe_dim // 2)
        b1 = b1 + moe.v11(x1_est.flatten(-2))
        b2 = b2 + moe.v21(x2_est.flatten(-2))
        b1 = torch.nn.functional.pad(b1, (0, lm_config.hidden_size // 2))
        b2 = torch.nn.functional.pad(b2, (lm_config.hidden_size // 2, 0))
        bias = (b1[:, None, :] + b2[None, :, :]).reshape(-1, lm_config.hidden_size)
        total_bias = total_bias + bias
    latent_to_resid = total_bias
if (transcoder_match := re.match(r"model\.(layers\.(\d+))", args.module)):
    module_raw = transcoder_match.group(1)
    hookpoint_to_sparse_model = load_sparsify_sparse_coders(
        Path(args.sae_path),
        [module_raw],
        "cuda",
        compile=False,
    )
    transcoder = hookpoint_to_sparse_model[module_raw]
    w_dec = transcoder.W_dec.data
    latent_to_resid = w_dec
#%%
del cache_lm
gc.collect()
#%%
tokens = loader.buffers[0].load()[-1]
n_sequences, max_seq_len = tokens.shape
#%%

cfg = SaeVisConfig(
    hook_point=args.module,
    minibatch_size_tokens=loader.cache_config["ctx_len"],
    features=[],
    # batch_size=dataset.cache_config["batch_size"],
)
layout = cfg.feature_centric_layout

ranges_and_precisions = ASYMMETRIC_RANGES_AND_PRECISIONS
quantiles = []
for r, p in ranges_and_precisions:
    start, end = r
    step = 10**-p
    quantiles.extend(np.arange(start, end - 0.5 * step, step))
quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)

#%%
latent_data_dict = {}

latent_stats = FeatureStatistics()
# supposed_latent = 0
bar = tqdm(total=args.latents)
i = -1
async for record in loader:
    i += 1
    # https://github.com/jbloomAus/SAEDashboard/blob/main/sae_dashboard/utils_fns.py
    latent_id = record.buffer.locations[0, 2].item()
    decoder_resid = latent_to_resid[latent_id].to(record.buffer.activations.device)
    logit_vector = lm_head(decoder_resid)

    buffer = record.buffer
    activations, locations = buffer.activations, buffer.locations
    _max = activations.max()
    nonzero_mask = activations.abs() > 1e-6
    nonzero_acts = activations[nonzero_mask]
    frac_nonzero = nonzero_mask.sum() / (n_sequences * max_seq_len)
    quantile_data = torch.quantile(activations.float(), quantiles_tensor)
    skew = torch.mean((activations - activations.mean())**3) / (activations.std()**3)
    kurtosis = torch.mean((activations - activations.mean())**4) / (activations.std()**4)
    latent_stats.update(FeatureStatistics(
        max=[_max.item()],
        frac_nonzero=[frac_nonzero.item()],
        skew=[skew.item()],
        kurtosis=[kurtosis.item()],
        quantile_data=[quantile_data.unsqueeze(0).tolist()],
        quantiles=quantiles + [1.0],
        ranges_and_precisions=ranges_and_precisions
    ))

    latent_data = FeatureData()
    latent_data.feature_tables_data = FeatureTablesData()
    latent_data.logits_histogram_data = LogitsHistogramData.from_data(
        data=logit_vector.to(
            torch.float32
        ),  # need this otherwise fails on MPS
        n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
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
        n_rows=layout.logits_table_cfg.n_rows,  # type: ignore
    )
    latent_data.decoder_weights_data = (
        DecoderWeightsDistribution(
            len(decoder_resid), decoder_resid.tolist()
        )
    )
    latent_data_dict[latent_id] = latent_data
    # supposed_latent += 1
    bar.update(1)
    bar.refresh()
bar.close()

latent_list = latent_dict[module].tolist()
cfg.features = latent_list
#%%
n_quantiles = 5
experiment_cfg = ExperimentConfig(
    n_examples_train=25,
    example_ctx_len=16,
    n_quantiles=n_quantiles,
    train_type="quantiles"
)
sequence_loader = LatentDataset(**kwargs | dict(experiment_cfg=experiment_cfg))
bar = tqdm(total=args.latents)
async for record in sequence_loader:
    groups = []
    for quantile_index, quantile_data in enumerate(
        list(batched(record.train, len(record.train) // n_quantiles))[::-1]):
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
            group.append(SequenceData(
                token_ids=example.tokens.tolist(),
                feat_acts=example.activations.tolist(),
                **default_attrs
            ))
        groups.append(SequenceGroupData(
            title=f"Quantile {quantile_index/n_quantiles:1%}-{(quantile_index+1)/n_quantiles:1%}",
            seq_data=group,
        ))
    latent_data_dict[record.latent.latent_index].sequence_data = SequenceMultiGroupData(
        seq_group_data=groups
    )
    bar.update(1)
    bar.refresh()
bar.close()
# %%
latent_list = list(latent_data_dict.keys())
tokenizer = loader.tokenizer
model = Namespace(
    tokenizer=tokenizer,
)

sae_vis_data = SaeVisData(
    cfg=cfg,
    feature_data_dict=latent_data_dict,
    feature_stats=latent_stats,
    model=model,
)
from sae_dashboard.data_writing_fns import save_feature_centric_vis
save_feature_centric_vis(sae_vis_data=sae_vis_data, filename="results/latent_dashboard.html")
# %%
