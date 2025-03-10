# WARNING: removes activations with total probability < 1e-2
# %%
from collections import defaultdict
from tqdm.auto import tqdm
from transformers import AutoConfig
from safetensors.numpy import load_file
from safetensors.torch import save_file
from pathlib import Path
from glob import glob
from tqdm.auto import trange
import numpy as np
import gc
import json
import os
import re
import torch
import numba as nb
from math import ceil

from opt_einsum import contract as einsum

torch.set_grad_enabled(False)


import sys

# base_dir = "../.."
base_dir = "."
sys.path.append(base_dir)
from delphi.utils import generate_split_indices

source_dir = base_dir / Path("results/monet_cache_separate")
target_dir = base_dir / Path("results/monet_cache_converted")
for monet_size in source_dir.glob("*"):
    print("Processing", monet_size)
    size = monet_size.name
    model_config = AutoConfig.from_pretrained(
        f"MonetLLM/monet-vd-{size.upper()}-100BT-hf", trust_remote_code=True
    )
    for layer_path in monet_size.glob("*"):
        target_layer_dir = target_dir / layer_path.relative_to(source_dir)
        if target_layer_dir.exists():
            print("Skipping", target_layer_dir)
            continue
        print("Writing to", target_layer_dir)
        config_path = layer_path / "config.json"
        config = json.loads(config_path.read_text())
        layer = int(
            re.match(r"\.model\.layers\.(\d+)\.router", layer_path.name).group(1)
        )
        st_files = []
        for cache_file in layer_path.glob("*.safetensors"):
            start, end = map(int, cache_file.stem.split("_"))
            st_files.append(((start, end), cache_file))
        st_files.sort()
        n_features = max(x[0][1] for x in st_files) + 1
        st_files = [
            (load_file(y), x) for x, y in tqdm(st_files, desc="Loading cache files")
        ]
        ##%%
        n_sequences, max_seq_len = st_files[0][0]["tokens"].shape
        n_experts = model_config.moe_experts**2
        ##%%
        combined_activations = np.zeros(
            (
                n_sequences,
                max_seq_len,
                model_config.moe_heads * model_config.moe_experts * 2,
            ),
            dtype=np.float32,
        )

        @nb.njit(parallel=True)
        def scatter_activations(combined_activations, activations, locations, st_start):
            for i in nb.prange(locations.shape[0]):
                batch, seq, dim = locations[i]
                combined_activations[batch, seq, dim + st_start] = activations[i]

        for st_file, (st_start, st_end) in tqdm(st_files, desc="Combining activations"):
            activations = st_file["activations"].astype(np.float32)
            locations = st_file["locations"]
            scatter_activations(combined_activations, activations, locations, st_start)
        ##%%
        reshaped_activations = combined_activations.reshape(
            (
                n_sequences,
                max_seq_len,
                model_config.moe_heads,
                model_config.moe_experts,
                2,
            )
        )
        g1, g2 = reshaped_activations[..., 0], reshaped_activations[..., 1]

        ##%%
        computed_activations = []
        batch_size = 8
        max_dim = max(n_sequences, max_seq_len, n_experts)
        max_fidelity = (
            torch.int64
            if np.log2(max_dim) > 32
            else (torch.int32 if np.log2(max_dim) > 16 else torch.int16)
        )
        for seq_start in trange(
            0, n_sequences, batch_size, desc="Computing activations"
        ):
            g1_gpu = (
                torch.from_numpy(g1[seq_start : seq_start + batch_size]).half().cuda()
            )
            g2_gpu = (
                torch.from_numpy(g2[seq_start : seq_start + batch_size]).half().cuda()
            )
            einsummed = einsum("bshx,bshy->bsxy", g1_gpu, g2_gpu)
            einsummed = einsummed.reshape(*einsummed.shape[:2], -1)
            # einsummed = torch.cat([g1_gpu, g2_gpu], dim=-1).view(*g1_gpu.shape[:2], -1)
            threshold = 1e-2
            locations_gpu = torch.nonzero(einsummed > threshold, as_tuple=True)
            activations_gpu = einsummed[
                locations_gpu[0], locations_gpu[1], locations_gpu[2]
            ]
            locations_gpu = torch.stack(
                [locations_gpu[0] + seq_start, *locations_gpu[1:]], dim=-1
            )
            computed_activations.append(
                (locations_gpu.to(max_fidelity).cpu(), activations_gpu.half().cpu())
            )
            del g1_gpu, g2_gpu, einsummed, locations_gpu, activations_gpu
            gc.collect()
            torch.cuda.empty_cache()
        ##%%
        all_locations, all_activations = zip(*computed_activations)
        activations, locations = torch.cat(all_activations), torch.cat(
            all_locations, dim=0
        )
        ##%%
        tokens = torch.from_numpy(st_files[0][0]["tokens"])
        target_layer_dir.mkdir(parents=True, exist_ok=True)
        n_splits = 8
        target_config = config | dict(
            n_splits=n_splits,
            ctx_len=max_seq_len,
            width=n_experts,
        )
        (target_layer_dir / "config.json").write_text(json.dumps(target_config))
        split_indices = generate_split_indices(n_experts, n_splits)
        for feature_min, feature_max in tqdm(split_indices, desc="Saving cache files"):
            feature_mask = (locations[:, 2] >= feature_min) & (
                locations[:, 2] <= feature_max
            )
            combined_st_file = dict(
                activations=activations[feature_mask],
                locations=locations[feature_mask],
                tokens=tokens,
            )
            print(
                "Saving", target_layer_dir / f"{feature_min}_{feature_max}.safetensors"
            )
            save_file(
                combined_st_file,
                target_layer_dir / f"{feature_min}_{feature_max}.safetensors",
            )
# %%
